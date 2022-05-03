### Tracks in Automatic Control Lab at ETH Zurich.
### Source: https://github.com/alexliniger/MPCC/tree/master/Matlab/Tracks

import os
import numpy as np
from numpy import loadtxt
import bisect
import math
import matplotlib.pyplot as plt


def Projection(point, line):
    assert len(point) == 1
    assert len(line) == 2

    x = np.array(point[0])
    x1 = np.array(line[0])
    x2 = np.array(line[len(line) - 1])

    dir1 = x2 - x1
    dir1 /= np.linalg.norm(dir1, 2)
    proj = x1 + dir1 * np.dot(x - x1, dir1)

    dir2 = (proj - x1)
    dir3 = (proj - x2)

    if np.linalg.norm(dir2, 2) > 0 and np.linalg.norm(dir3, 2) > 0:
        dir2 /= np.linalg.norm(dir2)
        dir3 /= np.linalg.norm(dir3)
        is_on_line = np.linalg.norm(dir2 - dir3, 2) > 1e-10
        if not is_on_line:
            if np.linalg.norm(x1 - proj, 2) < np.linalg.norm(x2 - proj, 2):
                proj = x1
            else:
                proj = x2
    dist = np.linalg.norm(x - proj, 2)
    return proj, dist


class Spline:

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        u"""
        Calc position

        if t is outside of the input x, return None

        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h):
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        #  print(B)
        return B


class Spline2D:
    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = [math.sqrt(idx ** 2 + idy ** 2)
                   for (idx, idy) in zip(dx, dy)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
        return k

    def calc_yaw(self, s):
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw


def calc_spline_course(x, y, ds=0.1):
    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


class Track:
    def __init__(self):
        self._calc_center_line()
        self._calc_track_length()
        self._calc_theta_track()

    def _calc_center_line(self):
        self.center_line = np.concatenate([
            self.x_center.reshape(1, -1),
            self.y_center.reshape(1, -1)
        ])

    def _calc_track_length(self):
        center = self.center_line
        # connect first and last point
        center = np.concatenate([center, center[:, 0].reshape(-1, 1)], axis=1)
        diff = np.diff(center)
        self.track_length = np.sum(np.linalg.norm(diff, 2, axis=0))

    def _calc_raceline_length(self, raceline):
        raceline = np.concatenate([raceline, raceline[:, 0].reshape(-1, 1)], axis=1)
        diff = np.diff(raceline)
        return np.sum(np.linalg.norm(diff, 2, axis=0))

    def _calc_theta_track(self):
        diff = np.diff(self.center_line)
        theta_track = np.cumsum(np.linalg.norm(diff, 2, axis=0))
        self.theta_track = np.concatenate([np.array([0]), theta_track])

    def _load_raceline(self, wx, wy, n_samples, v=None, t=None):
        self.spline = Spline2D(wx, wy)
        x, y = wx, wy
        theta = self.spline.s

        self.x_raceline = np.array(x)
        self.y_raceline = np.array(y)
        self.raceline = np.array([x, y])

        if v is not None:
            self.v_raceline = v
            self.t_raceline = t
            self.spline_v = Spline(theta, v)

    def _fit_cubic_splines(self, wx, wy, n_samples):
        sp = Spline2D(wx, wy)
        self.spline = sp
        s = np.linspace(0, sp.s[-1] - 0.001, n_samples)
        x, y = [], []
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            x.append(ix)
            y.append(iy)
        return x, y, s

    def _param2xy(self, theta):
        """	finds (x,y) coordinate on center line for a given theta
        """
        theta_track = self.theta_track
        idt = 0
        while idt < theta_track.shape[0] - 1 and theta_track[idt] <= theta:
            idt += 1
        deltatheta = (theta - theta_track[idt - 1]) / (theta_track[idt] - theta_track[idt - 1])
        x = self.x_center[idt - 1] + deltatheta * (self.x_center[idt] - self.x_center[idt - 1])
        y = self.y_center[idt - 1] + deltatheta * (self.y_center[idt] - self.y_center[idt - 1])
        return x, y

    def _xy2param(self, x, y):
        """	finds theta on center line for a given (x,y) coordinate
        """
        center_line = self.center_line
        theta_track = self.theta_track

        optxy, optidx = self.project(x, y, center_line)
        distxy = np.linalg.norm(optxy - center_line[:, optidx], 2)
        dist = np.linalg.norm(center_line[:, optidx + 1] - center_line[:, optidx], 2)
        deltaxy = distxy / dist
        if optidx == -1:
            theta = theta_track[optidx] + deltaxy * (self.track_length - theta_track[optidx])
        else:
            theta = theta_track[optidx] + deltaxy * (theta_track[optidx + 1] - theta_track[optidx])
        theta = theta % self.track_length
        return theta

    def project(self, x, y, raceline):
        """	finds projection for (x,y) on a raceline
        """
        point = [(x, y)]
        n_waypoints = raceline.shape[1]

        proj = np.empty([2, n_waypoints])
        dist = np.empty([n_waypoints])
        for idl in range(-1, n_waypoints - 1):
            line = [raceline[:, idl], raceline[:, idl + 1]]
            proj[:, idl], dist[idl] = Projection(point, line)
        optidx = np.argmin(dist)
        if optidx == n_waypoints - 1:
            optidx = -1
        optxy = proj[:, optidx]
        return optxy, optidx

    def project_fast(self, x, y, raceline):
        """	finds projection for (x,y) on a raceline
        """
        point = [(x, y)]
        n_waypoints = raceline.shape[1]

        proj = np.empty([2, n_waypoints - 1])
        dist = np.empty([n_waypoints - 1])
        for idl in range(n_waypoints - 1):
            line = [raceline[:, idl], raceline[:, idl + 1]]
            proj[:, idl], dist[idl] = Projection(point, line)
        optidx = np.argmin(dist)
        optxy = proj[:, optidx]
        return optxy, optidx

    def _plot(self, color='g', grid=True, figsize=(6.4, 4.8)):
        """ plot center, inner and outer track lines
        """
        fig = plt.figure(figsize=figsize)
        plt.grid(grid)
        # plt.plot(self.x_center, self.y_center, '--'+color, lw=0.75, alpha=0.5)
        plt.plot(self.x_outer, self.y_outer, color, lw=0.75, alpha=0.5)
        plt.plot(self.x_inner, self.y_inner, color, lw=0.75, alpha=0.5)
        plt.scatter(0, 0, color='k', alpha=0.2)
        plt.axis('equal')
        return fig

    def plot_raceline(self):
        """ plot center, inner and outer track lines
        """
        fig = self._plot()
        plt.plot(self.x_raceline, self.y_raceline, 'b', lw=1)
        plt.show()

    def param_to_xy(self, theta, **kwargs):
        """	convert distance along the track to x, y coordinates
        """
        raise NotImplementedError

    def xy_to_param(self, x, y):
        """	convert x, y coordinates to distance along the track
        """
        raise NotImplementedError




class ETHZTrack(Track):
    """ base class for ETHZ tracks"""

    def __init__(self, track_id, track_width, reference, longer):

        loadstr = 'src/ethz' + track_id
        path = os.path.join(os.path.dirname(__file__), loadstr)
        self.inner = loadtxt(
            path + '_inner.txt',
            comments='#',
            delimiter=',',
            unpack=False
        )
        self.center = loadtxt(
            path + '_center.txt',
            comments='#',
            delimiter=',',
            unpack=False
        )
        self.outer = loadtxt(
            path + '_outer.txt',
            comments='#',
            delimiter=',',
            unpack=False
        )
        self.x_inner, self.y_inner = self.inner[0, :], self.inner[1, :]
        self.x_center, self.y_center = self.center[0, :], self.center[1, :]
        self.x_outer, self.y_outer = self.outer[0, :], self.outer[1, :]
        self.track_width = track_width
        super(ETHZTrack, self).__init__()
        self.load_raceline(reference, track_id, longer)

    def param_to_xy(self, theta):
        """	convert distance along the track to x, y coordinates
        """
        return self._param2xy(theta)

    def xy_to_param(self, x, y):
        """	convert x, y coordinates to distance along the track
        """
        theta = self._xy2param(x, y)
        return theta

    def load_raceline(self, reference, track_id, longer):
        """	load raceline stored in npz file with keys 'x', 'y', 'speed', 'inputs'
        """
        if longer:
            suffix = '_long'
        else:
            suffix = ''
        if reference is 'center':
            n_samples = 2 * self.x_center.shape[0] - 1
            self._load_raceline(
                wx=self.x_center,
                wy=self.y_center,
                n_samples=n_samples
            )
        elif reference is 'optimal':
            file_name = 'ethz{}_raceline{}.npz'.format(track_id, suffix)
            file_path = os.path.join(os.path.dirname(__file__), 'src', file_name)
            raceline = np.load(file_path)
            n_samples = raceline['x'].size
            self._load_raceline(
                wx=raceline['x'],
                wy=raceline['y'],
                n_samples=n_samples,
                v=raceline['speed'],
                t=raceline['time'],
            )
        else:
            raise NotImplementedError

    def plot(self, **kwargs):
        """ plot center, inner and outer track lines
        """
        fig = self._plot(**kwargs)
        return fig


class ETHZ(ETHZTrack):
    def __init__(self, reference='center', longer=False):
        track_width = 0.37
        super(ETHZ, self).__init__(
            track_id='',
            track_width=track_width,
            reference=reference,
            longer=longer,
        )
        self.psi_init = -np.pi / 4
        self.x_init = self.x_raceline[0]
        self.y_init = self.y_raceline[0]
        self.vx_init = 0.1


class ETHZMobil(ETHZTrack):
    """ETHZ Mobil track"""

    def __init__(self, reference='center', longer=False):
        track_width = 0.46
        super(ETHZMobil, self).__init__(
            track_id='Mobil',
            track_width=track_width,
            reference=reference,
            longer=longer,
        )
        self.psi_init = 0.
        self.x_init = self.x_raceline[0]
        self.y_init = self.y_raceline[0]
        self.vx_init = 0.1
