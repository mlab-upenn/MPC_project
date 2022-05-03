from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from cvxopt.solvers import qp
import numpy as np
import datetime
import pdb


def curvature(s, PointAndTangent):
    """curvature computation
    s: curvilinear abscissa at which the curvature has to be evaluated
    PointAndTangent: points and tangent vectors defining the map (these quantities are initialized in the map object)
    """
    TrackLength = PointAndTangent[-1, 3] + PointAndTangent[-1, 4]

    # In case on a lap after the first one
    while (s > TrackLength):
        s = s - TrackLength

    # Given s \in [0, TrackLength] compute the curvature
    # Compute the segment in which system is evolving
    index = np.all([[s >= PointAndTangent[:, 3]], [s < PointAndTangent[:, 3] + PointAndTangent[:, 4]]], axis=0)

    i = int(np.where(np.squeeze(index))[0])
    curvature = PointAndTangent[i, 5]

    return curvature


def Regression(x, u, lamb):
    """Estimates linear system dynamics
    x, u: date used in the regression
    lamb: regularization coefficient
    """

    # Want to solve W^* = argmin sum_i ||W^T z_i - y_i ||_2^2 + lamb ||W||_F,
    # with z_i = [x_i u_i] and W \in R^{n + d} x n
    Y = x[2:x.shape[0], :]
    X = np.hstack((x[1:(x.shape[0] - 1), :], u[1:(x.shape[0] - 1), :]))

    Q = np.linalg.inv(np.dot(X.T, X) + lamb * np.eye(X.shape[1]))
    b = np.dot(X.T, Y)
    W = np.dot(Q, b)

    A = W.T[:, 0:6]
    B = W.T[:, 6:8]

    ErrorMatrix = np.dot(X, W) - Y
    ErrorMax = np.max(ErrorMatrix, axis=0)
    ErrorMin = np.min(ErrorMatrix, axis=0)
    Error = np.vstack((ErrorMax, ErrorMin))

    return A, B, Error


class ClosedLoopData():
    def __init__(self, dt, Time, v0):
        """Initialization
        Arguments:
            dt: discretization time
            Time: maximum time [s] which can be recorded
            v0: velocity initial condition
        """
        self.dt = dt
        self.Points = int(Time / dt)  # Number of points in the simulation
        self.u = np.zeros((self.Points, 2))  # Initialize the input vector
        self.x = np.zeros((self.Points + 1, 6))  # Initialize state vector (In curvilinear abscissas)
        self.x_glob = np.zeros((self.Points + 1, 6))  # Initialize the state vector in absolute reference frame
        self.SimTime = 0.0
        self.x[0, 0] = v0
        self.x_glob[0,0] = v0

    def updateInitialConditions(self, x, x_glob):
        """Clears memory and resets initial condition
        x: initial condition is the curvilinear reference frame
        x_glob: initial condition in the inertial reference frame
        """
        self.x[0, :] = x
        self.x_glob[0, :] = x_glob

        self.x[1:, :] = 0*self.x[1:, :]
        self.x_glob[1:, :] = 0*self.x_glob[1:, :]


class PID:
    """Create the PID controller used for path following at constant speed
    Attributes:
        solve: given x0 computes the control action
    """
    def __init__(self, vt):
        """Initialization
        Arguments:
            vt: target velocity
        """
        self.vt = vt
        self.uPred = np.zeros([1,2])

        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer
        self.feasible = 1

    def solve(self, x0):
        """Computes control action
        Arguments:
            x0: current state position
        """
        vt = self.vt
        self.uPred[0, 0] = - 0.6 * x0[5] - 0.9 * x0[3] + np.max([-0.9, np.min([np.random.randn() * 0.25, 0.9])])
        self.uPred[0, 1] = 1.5 * (vt - x0[0]) + np.max([-0.2, np.min([np.random.randn() * 0.10, 0.2])])


class PredictiveModel():
    """Object collecting the predictions and SS at eath time step
    """
    def __init__(self, N, n, d, TimeLMPC, numSS_Points, Laps):
        """
        Initialization:
            N: horizon length
            n, d: input and state dimensions
            TimeLMPC: maximum simulation time length [s]
            num_SSpoints: number used to buils SS at each time step
        """
        self.PredictedStates = np.zeros((N+1, n, TimeLMPC, Laps))
        self.PredictedInputs = np.zeros((N, d, TimeLMPC, Laps))

        self.SSused   = np.zeros((n , numSS_Points, TimeLMPC, Laps))
        self.Qfunused = np.zeros((numSS_Points, TimeLMPC, Laps))

# class PredictiveModel():
#     def __init__(self, n, d, map, trToUse):
#         self.map = map
#         self.n = n  # state dimension
#         self.d = d  # input dimention
#         self.xStored = []
#         self.uStored = []
#         self.MaxNumPoint = 7  # max number of point per lap to use
#         self.h = 5  # bandwidth of the Kernel for local linear regression
#         self.lamb = 0.0  # regularization
#         self.dt = 0.1
#         self.scaling = np.array([[0.1, 0.0, 0.0, 0.0, 0.0],
#                                  [0.0, 1.0, 0.0, 0.0, 0.0],
#                                  [0.0, 0.0, 1.0, 0.0, 0.0],
#                                  [0.0, 0.0, 0.0, 1.0, 0.0],
#                                  [0.0, 0.0, 0.0, 0.0, 1.0]])
#
#         self.stateFeatures = [0, 1, 2]
#         self.inputFeaturesVx = [1]
#         self.inputFeaturesLat = [0]
#         self.usedIt = [i for i in range(trToUse)]
#         self.lapTime = []
#
#     def addTrajectory(self, x, u):
#         if self.lapTime == [] or x.shape[0] >= self.lapTime[-1]:
#             self.xStored.append(x)
#             self.uStored.append(u)
#             self.lapTime.append(x.shape[0])
#         else:
#             for i in range(0, len(self.xStored)):
#                 if x.shape[0] < self.lapTime[i]:
#                     self.xStored.insert(i, x)
#                     self.uStored.insert(i, u)
#                     self.lapTime.insert(i, x.shape[0])
#                     break
#
#     def regressionAndLinearization(self, x, u):
#         Ai = np.zeros((self.n, self.n))
#         Bi = np.zeros((self.n, self.d))
#         Ci = np.zeros(self.n)
#
#         # Compute Index to use for each stored lap
#         xuLin = np.hstack((x[self.stateFeatures], u[:]))
#         self.indexSelected = []
#         self.K = []
#         for ii in self.usedIt:
#             indexSelected_i, K_i = self.computeIndices(xuLin, ii)
#             self.indexSelected.append(indexSelected_i)
#             self.K.append(K_i)
#         # print("xuLin: ",xuLin)
#         # print("aaa indexSelected: ", self.indexSelected)
#
#         # =========================
#         # ====== Identify vx ======
#         Q_vx, M_vx = self.compute_Q_M(self.inputFeaturesVx, self.usedIt)
#
#         yIndex = 0
#         b_vx = self.compute_b(yIndex, self.usedIt, M_vx)
#         Ai[yIndex, self.stateFeatures], Bi[yIndex, self.inputFeaturesVx], Ci[yIndex] = self.LMPC_LocLinReg(Q_vx, b_vx,
#                                                                                                            self.inputFeaturesVx)
#
#         # =======================================
#         # ====== Identify Lateral Dynamics ======
#         Q_lat, M_lat = self.compute_Q_M(self.inputFeaturesLat, self.usedIt)
#
#         yIndex = 1  # vy
#         b_vy = self.compute_b(yIndex, self.usedIt, M_lat)
#         Ai[yIndex, self.stateFeatures], Bi[yIndex, self.inputFeaturesLat], Ci[yIndex] = self.LMPC_LocLinReg(Q_lat, b_vy,
#                                                                                                             self.inputFeaturesLat)
#
#         yIndex = 2  # wz
#         b_wz = self.compute_b(yIndex, self.usedIt, M_lat)
#         Ai[yIndex, self.stateFeatures], Bi[yIndex, self.inputFeaturesLat], Ci[yIndex] = self.LMPC_LocLinReg(Q_lat, b_wz,
#                                                                                                             self.inputFeaturesLat)
#
#         # ===========================
#         # ===== Linearization =======
#         vx = x[0];
#         vy = x[1]
#         wz = x[2];
#         epsi = x[3]
#         s = x[4];
#         ey = x[5]
#         dt = self.dt
#
#         if s < 0:
#             print("s is negative, here the state: \n", x)
#
#         startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
#         cur = self.map.curvature(s)
#         cur = self.map.curvature(s)
#         den = 1 - cur * ey
#
#         # ===========================
#         # ===== Linearize epsi ======
#         # epsi_{k+1} = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
#         depsi_vx = -dt * np.cos(epsi) / den * cur
#         depsi_vy = dt * np.sin(epsi) / den * cur
#         depsi_wz = dt
#         depsi_epsi = 1 - dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den * cur
#         depsi_s = 0  # Because cur = constant
#         depsi_ey = dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * cur * (-cur)
#
#         Ai[3, :] = [depsi_vx, depsi_vy, depsi_wz, depsi_epsi, depsi_s, depsi_ey]
#         Ci[3] = epsi + dt * (wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur) - np.dot(Ai[3, :], x)
#         # ===========================
#         # ===== Linearize s =========
#         # s_{k+1} = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) )
#         ds_vx = dt * (np.cos(epsi) / den)
#         ds_vy = -dt * (np.sin(epsi) / den)
#         ds_wz = 0
#         ds_epsi = dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den
#         ds_s = 1  # + Ts * (Vx * cos(epsi) - Vy * sin(epsi)) / (1 - ey * rho) ^ 2 * (-ey * drho);
#         ds_ey = -dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * (-cur)
#
#         Ai[4, :] = [ds_vx, ds_vy, ds_wz, ds_epsi, ds_s, ds_ey]
#         Ci[4] = s + dt * ((vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey)) - np.dot(Ai[4, :], x)
#
#         # ===========================
#         # ===== Linearize ey ========
#         # ey_{k+1} = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi))
#         dey_vx = dt * np.sin(epsi)
#         dey_vy = dt * np.cos(epsi)
#         dey_wz = 0
#         dey_epsi = dt * (vx * np.cos(epsi) - vy * np.sin(epsi))
#         dey_s = 0
#         dey_ey = 1
#
#         Ai[5, :] = [dey_vx, dey_vy, dey_wz, dey_epsi, dey_s, dey_ey]
#         Ci[5] = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi)) - np.dot(Ai[5, :], x)
#
#         endTimer = datetime.datetime.now();
#         deltaTimer_tv = endTimer - startTimer
#
#         return Ai, Bi, Ci
#
#     def compute_Q_M(self, inputFeatures, usedIt):
#         Counter = 0
#         X0 = np.empty((0, len(self.stateFeatures) + len(inputFeatures)))
#         Ktot = np.empty((0))
#
#         for it in usedIt:
#             X0 = np.append(X0, np.hstack((self.xStored[it][np.ix_(self.indexSelected[Counter], self.stateFeatures)],
#                                           self.uStored[it][np.ix_(self.indexSelected[Counter], inputFeatures)])),
#                            axis=0)
#             Ktot = np.append(Ktot, self.K[Counter])
#             Counter += 1
#
#         M = np.hstack((X0, np.ones((X0.shape[0], 1))))
#         Q0 = np.dot(np.dot(M.T, np.diag(Ktot)), M)
#         Q = matrix(Q0 + self.lamb * np.eye(Q0.shape[0]))
#
#         return Q, M
#
#     def compute_b(self, yIndex, usedIt, M):
#         Counter = 0
#         y = np.empty((0))
#         Ktot = np.empty((0))
#
#         for it in usedIt:
#             y = np.append(y, np.squeeze(self.xStored[it][self.indexSelected[Counter] + 1, yIndex]))
#             Ktot = np.append(Ktot, self.K[Counter])
#             Counter += 1
#
#         b = matrix(-np.dot(np.dot(M.T, np.diag(Ktot)), y))
#         return b
#
#     def LMPC_LocLinReg(self, Q, b, inputFeatures):
#         # Solve QP
#         res_cons = qp(Q, b)  # This is ordered as [A B C]
#         # Unpack results
#         result = np.squeeze(np.array(res_cons['x']))
#         A = result[0:len(self.stateFeatures)]
#         B = result[len(self.stateFeatures):(len(self.stateFeatures) + len(inputFeatures))]
#         C = result[-1]
#         return A, B, C
#
#     def computeIndices(self, x, it):
#         oneVec = np.ones((self.xStored[it].shape[0] - 1, 1))
#         xVec = (np.dot(np.array([x]).T, oneVec.T)).T
#         DataMatrix = np.hstack((self.xStored[it][0:-1, self.stateFeatures], self.uStored[it][0:-1, :]))
#
#         diff = np.dot((DataMatrix - xVec), self.scaling)
#         norm = la.norm(diff, 1, axis=1)
#         indexTot = np.squeeze(np.where(norm < self.h))
#         if (indexTot.shape[0] >= self.MaxNumPoint):
#             index = np.argsort(norm)[0:self.MaxNumPoint]
#         else:
#             index = indexTot
#
#         K = (1 - (norm[index] / self.h) ** 2) * 3 / 4
#         # if norm.shape[0]<500:
#         #     print("norm: ", norm, norm.shape)
#
#         return index, K