import numpy as np
import pdb

# class system(object):
# 	"""docstring for system"""
# 	def __init__(self, x0, dt):
# 		self.x 	   = [x0]
# 		self.u 	   = []
# 		self.w 	   = []
# 		self.x0    = x0
# 		self.dt    = dt
#
# 	def applyInput(self, ut):
# 		self.u.append(ut)
#
# 		xt = self.x[-1]
# 		x_next      = xt[0] + self.dt * np.cos(xt[3]) * xt[2]
# 		y_next      = xt[1] + self.dt * np.sin(xt[3]) * xt[2]
# 		v_next      = xt[2] + self.dt * ut[0]
# 		theta_next  = xt[3] + self.dt * ut[1]
#
# 		state_next = np.array([x_next, y_next, v_next, theta_next])
#
# 		self.x.append(state_next)
#
# 	def reset_IC(self):
# 		self.x = [self.x0]
# 		self.u = []
# 		self.w = []


class system(object):
	"""docstring for system"""

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
		beta = np.arctan(lr / (lr + lf) * np.tan(ut[1]))

		x_next = xt[0] + self.dt * xt[2] * np.cos(xt[3] + beta)
		y_next = xt[1] + self.dt * xt[2] * np.sin(xt[3] + beta)
		v_next = xt[2] + self.dt * ut[0]
		theta_next = xt[3] + self.dt * xt[2] / lr * np.sin(beta)

		state_next = np.array([x_next, y_next, v_next, theta_next])

		self.x.append(state_next)

	def reset_IC(self):
		self.x = [self.x0]
		self.u = []
		self.w = []
