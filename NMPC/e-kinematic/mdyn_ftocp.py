from casadi import *
from numpy import *
import numpy as np
from cvxpy import *
import time


class NFTOCPNLP(object):

    def __init__(self, N, Q, R, Qf, goal, dt, bx, bu):
        # Define variables
        self.N = N
        self.n = Q.shape[1]
        self.d = R.shape[1]
        self.bx = bx
        self.bu = bu
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.goal = goal
        self.dt = dt
        self.optCost = np.inf
        self.bx = bx
        self.bu = bu

        self.buildFTOCP()
        self.solverTime = []

    def solve(self, x0, verbose=False):
        # Set initial condition + state and input box constraints
        self.lbx = x0.tolist() + (-self.bx).tolist() * (self.N) + (-self.bu).tolist() * self.N
        self.ubx = x0.tolist() + (self.bx).tolist() * (self.N) + (self.bu).tolist() * self.N

        # Solve the NLP
        start = time.time()
        sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics)
        end = time.time()
        delta = end - start
        self.solverTime.append(delta)

        # Check if the solution is feasible
        if (self.solver.stats()['success']):
            self.feasible = 1
            x = sol["x"]
            self.qcost = sol["f"]
            self.xPred = np.array(x[0:(self.N + 1) * self.n].reshape((self.n, self.N + 1))).T
            self.uPred = np.array(
                x[(self.N + 1) * self.n:((self.N + 1) * self.n + self.d * self.N)].reshape((self.d, self.N))).T
            self.mpcInput = self.uPred[0][0]

            print("xPredicted:")
            print(self.xPred)
            print("uPredicted:")
            print(self.uPred)
            print("Cost:")
            print(self.qcost)

            print("NLP Solver Time: ", delta, " seconds.")
        else:
            self.xPred = np.zeros((self.N + 1, self.n))
            self.uPred = np.zeros((self.N, self.d))
            self.mpcInput = []
            self.feasible = 0
            print("Unfeasible")

        return self.uPred[0]

    def buildFTOCP(self):

        n = self.n
        d = self.d

        # Define variables
        X = SX.sym('X', n * (self.N + 1))
        U = SX.sym('U', d * self.N)

        # Define dynamic constraints
        self.constraint = []

        for i in range(0, self.N):
            X_next = self.ekinematic(X[n * i:n * (i + 1)], U[d * i:d * (i + 1)])
            for j in range(0, self.n):
                self.constraint = vertcat(self.constraint, X_next[j] - X[n * (i + 1) + j])

        self.cost = 0
        for i in range(0, self.N):
            self.cost = self.cost + (X[n * i:n * (i + 1)] - self.goal).T @ self.Q @ (X[n * i:n * (i + 1)] - self.goal)
            self.cost = self.cost + U[d * i:d * (i + 1)].T @ self.R @ U[d * i:d * (i + 1)]

        self.cost = self.cost + (X[n * self.N:n * (self.N + 1)] - self.goal).T @ self.Qf @ (
                    X[n * self.N:n * (self.N + 1)] - self.goal)

        # Standard NLP
        opts = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        nlp = {'x': vertcat(X, U), 'f': self.cost, 'g': self.constraint}
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)

        # Set lower bound of inequality constraint to zero to force n*N state dynamics
        self.lbg_dyanmics = [0] * (n * self.N)
        self.ubg_dyanmics = [0] * (n * self.N)

    def ekinematic(self, x, u):
        lf = 0.125
        lr = 0.125
        a = u[0]
        steer = u[1]
        u_next = steer

        x_next = x[0] + self.dt * (x[3] * np.cos(x[2]) - x[4] * np.sin(x[2]))
        y_next = x[1] + self.dt * (x[3] * np.sin(x[2]) + x[4] * np.cos(x[2]))
        theta_next = x[2] + self.dt * x[5]
        vx_next = x[3] + self.dt * a
        vy_next = x[4] + self.dt * (lr / (lr + lf) * (steer * vx_next + x[3] * u_next))
        yaw_next = x[5] + self.dt * (1 / (lr + lf) * (steer * vx_next + x[3] * u_next))

        state_next = [x_next, y_next, theta_next, vx_next, vy_next, yaw_next]

        return state_next
