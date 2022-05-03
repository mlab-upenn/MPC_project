import time
import numpy as np
import casadi as cs
from scipy.linalg import solve_triangular


def CasadiRBF(X, Y, model):

    sX = X.shape[0]
    sY = Y.shape[0]
    length_scale = model.kernel_.get_params()['k1__k2__length_scale'].reshape(1, -1)
    constant = model.kernel_.get_params()['k1__k1__constant_value']
    X = X / cs.repmat(length_scale, sX, 1)
    Y = Y / cs.repmat(length_scale, sY, 1)
    dist = cs.repmat(cs.sum1(X.T ** 2).T, 1, sY) + cs.repmat(cs.sum1(Y.T ** 2), sX, 1) - 2 * cs.mtimes(X, Y.T)
    K = constant * cs.exp(-.5 * dist)
    return K


def CasadiConstant(X, Y, model):
    """ Constant kernel in CasADi
    """
    constant = model.kernel_.get_params()['k2__constant_value']
    sX = X.shape[0]
    sY = Y.shape[0]
    K = constant * cs.SX.ones((sX, sY))
    return K



def loadGPModel(name, model, xscaler, yscaler):
    X = model.X_train_
    x = cs.SX.sym('x', 1, X.shape[1])

    K1 = CasadiRBF(x, X, model)
    K2 = CasadiConstant(x, X, model)
    K = K1 + K2

    y_mu = cs.mtimes(K, model.alpha_) + model._y_train_mean
    y_mu = y_mu * yscaler.scale_ + yscaler.mean_

    # variance
    L_inv = solve_triangular(model.L_.T,np.eye(model.L_.shape[0]))
    K_inv = L_inv.dot(L_inv.T)

    K1_ = CasadiRBF(x, x, model)
    K2_ = CasadiConstant(x, x, model)
    K_ = K1_ + K2_

    y_var = cs.diag(K_) - cs.sum2(cs.mtimes(K, K_inv)*K)
    y_var = cs.fmax(y_var, 0)
    y_std = cs.sqrt(y_var)
    y_std *= yscaler.scale_

    gpmodel = cs.Function(name, [x], [y_mu, y_std])
    return gpmodel
