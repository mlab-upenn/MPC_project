import numpy as np
import pdb
import math


class systemdy(object):

	def __init__(self, x0, dt):
		self.x = [x0]
		self.u = []
		self.w = []
		self.x0 = x0
		self.dt = dt

	def applyInput(self, ut):
		self.u.append(ut)

		xt = self.x[-1]

		lf = 0.125
		lr = 0.125
		a = ut[0]
		steer = ut[1]
		u_next = steer

		x_next = xt[0] + self.dt * (xt[3] * np.cos(xt[2]) - xt[4] * np.sin(xt[2]))
		y_next = xt[1] + self.dt * (xt[3] * np.sin(xt[2]) + xt[4] * np.cos(xt[2]))
		theta_next = xt[2] + self.dt * xt[5]
		vx_next = xt[3] + self.dt * a
		vy_next = xt[4] + self.dt * (lr / (lr + lf) * (steer * vx_next + xt[3] * u_next))
		yaw_next = xt[5] + self.dt * (1 / (lr + lf) * (steer * vx_next + xt[3] * u_next))

		state_next = np.array([x_next, y_next, theta_next, vx_next, vy_next, yaw_next])

		self.x.append(state_next)

	def reset_IC(self):
		self.x = [self.x0]
		self.u = []
		self.w = []
