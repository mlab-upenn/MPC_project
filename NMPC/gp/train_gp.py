import time
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_true_predicted_variance(y_true, y_mu, y_std, x=None, xlabel=None, ylabel=None, figsize=(8, 6),
                                 plot_title=None):
    y_true = y_true.flatten()
    y_mu = y_mu.flatten()
    y_std = y_std.flatten()

    l = y_true.shape[0]
    if x is None:
        x = range(l)

    plt.figure(figsize=figsize)
    plt.title(plot_title)
    gs = gridspec.GridSpec(3, 1)

    # mean variance
    plt.subplot(gs[:-1, :])
    plt.plot(x, y_mu, '#990000', ls='-', lw=1.5, zorder=9,
             label='predicted')
    plt.fill_between(x, (y_mu + 2 * y_std), (y_mu - 2 * y_std),
                     alpha=0.2, color='g', label='+-2sigma')
    plt.plot(x, y_true, '#e68a00', ls='--', lw=1, zorder=9,
             label='true')
    plt.legend(loc='upper right')
    plt.title('Ground truth vs predicted results')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=0)

    plt.subplot(gs[2, :])
    plt.plot(x, np.abs(np.array(y_true).flatten() - y_mu), '#990000',
             ls='-', lw=0.5, zorder=9)
    plt.fill_between(x, np.zeros([l, 1]).flatten(), 2 * y_std,
                     alpha=0.2, color='g')
    plt.title("Model error and predicted variance")
    plt.xlabel(xlabel)
    plt.ylabel('error ' + ylabel)
    plt.tight_layout()


SAVE_MODELS = True

N_SAMPLES = 400
VARIDX = 2
state_names = ['x', 'y', 'yaw', 'vx', 'vy', 'omega']
filename = 'set/{}gp.pickle'.format(state_names[VARIDX])


def load_data(CTYPE, TRACK_NAME, TRACK_NAME1, VARIDX, xscaler=None, yscaler=None):
    data_dyn = np.load('/home/mlab/Dynamic_bic_mpc/gp/set/DYN-{}-{}.npz'.format(CTYPE, TRACK_NAME))
    data_kin = np.load('/home/mlab/Dynamic_bic_mpc/gp/set/DYN-{}-{}.npz'.format(CTYPE, TRACK_NAME1))
    y_all = data_dyn['states'][:6, 1:N_SAMPLES + 1] - data_kin['states'][:6, 1:N_SAMPLES + 1]
    x = np.concatenate([
        data_kin['inputs'][:, :N_SAMPLES].T,
        data_kin['states'][:6, :N_SAMPLES].T,
        data_dyn['states'][:6, :N_SAMPLES].T], axis=1)
    y = y_all[VARIDX].reshape(-1, 1)

    if xscaler is None or yscaler is None:
        xscaler = StandardScaler()
        yscaler = StandardScaler()
        xscaler.fit(x)
        yscaler.fit(y)
        return xscaler.transform(x), yscaler.transform(y), xscaler, yscaler
    else:
        return xscaler.transform(x), yscaler.transform(y)


x_train, y_train, xscaler, yscaler = load_data('PP', 'cf120', 'cf125', VARIDX)

#####################################################################
# train GP model

k1 = 1.0 * RBF(
    length_scale=np.ones(x_train.shape[1]),
    length_scale_bounds=(1e-5, 1e5),
)
k2 = ConstantKernel(0.1)
kernel = k1 + k2
model = GaussianProcessRegressor(
    alpha=1e-6,
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=10,
)
start = time.time()
model.fit(x_train, y_train)
end = time.time()
print('training time: %ss' % (end - start))
print('final kernel: %s' % (model.kernel_))

if SAVE_MODELS:
    with open(filename, 'wb') as f:
        pickle.dump((model, xscaler, yscaler), f)

#####################################################################
# test GP model on training data

y_train_mu, y_train_std = model.predict(x_train, return_std=True)
y_train = yscaler.inverse_transform(y_train)
y_train_mu = yscaler.inverse_transform(y_train_mu)
y_train_std *= yscaler.scale_

MSE = mean_squared_error(y_train, y_train_mu, multioutput='raw_values')
R2Score = r2_score(y_train, y_train_mu, multioutput='raw_values')
EV = explained_variance_score(y_train, y_train_mu, multioutput='raw_values')

print('Root mean square error (RMSE): %s' % (np.sqrt(MSE)))
print('Normalized mean square error (NMSE): %s' % (np.sqrt(MSE) / np.array(np.abs(y_train.mean()))))
print('R2 score: %s' % (R2Score))
print('Explained variance: %s' % (EV))

#####################################################################
# test GP model on validation data

N_SAMPLES = 400
x_test, y_test = load_data('NMPC', 'dycf120ETHZ', 'dycf125ETHZ', VARIDX, xscaler=xscaler, yscaler=yscaler)
y_test_mu, y_test_std = model.predict(x_test, return_std=True)
y_test = yscaler.inverse_transform(y_test)
y_test_mu = yscaler.inverse_transform(y_test_mu)
y_test_std *= yscaler.scale_

MSE = mean_squared_error(y_test, y_test_mu, multioutput='raw_values')
R2Score = r2_score(y_test, y_test_mu, multioutput='raw_values')
EV = explained_variance_score(y_test, y_test_mu, multioutput='raw_values')

print('Root mean square error (RMSE): %s' % (np.sqrt(MSE)))
print('Normalized mean square error (NMSE): %s' % (np.sqrt(MSE) / np.array(np.abs(y_test.mean()))))
print('R2 score: %s' % (R2Score))
print('Explained variance: %s' % (EV))

#####################################################################
# plot results

plot_true_predicted_variance(
    y_train, y_train_mu, y_train_std,
    ylabel='{} '.format(state_names[VARIDX]), xlabel='Sample index'
)

plot_true_predicted_variance(
    y_test, y_test_mu, y_test_std,
    ylabel='{} '.format(state_names[VARIDX]), xlabel='Sample index'
)

plt.show()
