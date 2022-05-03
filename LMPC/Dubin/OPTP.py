import numpy as np
from scipy import linalg, sparse
from scipy.sparse import vstack
from osqp import OSQP
from numpy import linalg as la
from cvxopt.solvers import qp
from dataclasses import dataclass, field
from cvxopt import spmatrix, matrix, solvers
from casadi import sin, cos, SX, vertcat, Function, jacobian
import pdb
import datetime
from utils import PredictiveModel



class OCTP(object):

    def __init__(self, dim_state, dim_input, N, Q, R, Qm, Fx, bx, Fu, bu, Qslack, slacks, A, B, xRef, timevarying, dR):
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.N = N
        self.Q = Q
        self.R = R
        self.Qm = Qm
        self.Fx = Fx
        self.bx = bx
        self.Fu = Fu
        self.bu = bu
        self.Qslack = Qslack
        self.slacks = slacks
        self.A = A
        self.B = B
        self.xRef = xRef
        self.timevarying = timevarying
        self.ts = 0
        self.dR = dR

        if self.timevarying == True:
            self.xLin = PredictiveModel.xStored[-1][0:self.N + 1, :]
            self.uLin = PredictiveModel.uStored[-1][0:self.N, :]
            self.computeLTVdynamics()

        self.OldInput = np.zeros((1, 2))

        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now();
        deltaTime = endTimer - startTimer
        self.solverTime = deltaTime
        self.linearizationTime = deltaTime

        self.buildIneqConstr()
        self.buildCost()
        self.buildEqConstr()

        self.xPred = []

    def solve(self, x0):
        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L
        startTime = datetime.datetime.now()
        self.osqp_solve_qp(self.H_FTOCP, self.q_FTOCP, self.F_FTOCP, self.b_FTOCP, self.G_FTOCP,
                           np.add(np.dot(self.E_FTOCP, x0), self.L_FTOCP))
        self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(self.dim_state * (self.N + 1))]), (self.N + 1, self.dim_state)))).T
        self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[self.dim_input * (self.N + 1) + np.arange(self.dim_input * self.N)]), (self.N, self.dim_input)))).T
        endTime = datetime.datetime.now()
        deltaTime = endTime - startTime
        self.solverTime = deltaTime
        # print("Solving time (s): ", deltaTime)

        self.zt = self.xPred[-1, :]
        self.zt_u = self.uPred[-1, :]

        if self.timevarying == True:
            self.xLin = np.vstack((self.xPred[1:, :], self.zt))
            self.uLin = np.vstack((self.uPred[1:, :], self.zt_u))

        self.OldInput = self.uPred[0, :]
        self.ts += 1

    def buildIneqConstr(self):
        # The inequality constraint is Fz<=b
        # Let's start by computing the submatrix of F relates with the state
        rep_a = [self.Fx] * (self.N)
        Mat = linalg.block_diag(*rep_a)
        NoTerminalConstr = np.zeros((np.shape(Mat)[0], self.dim_state))
        Fxtot = np.hstack((Mat, NoTerminalConstr))
        bxtot = np.tile(np.squeeze(self.bx), self.N)

        # Let's start by computing the submatrix of F relates with the input
        rep_b = [self.Fu] * (self.N)
        Futot = linalg.block_diag(*rep_b)
        butot = np.tile(np.squeeze(self.bu), self.N)

        # Let's stack all together
        F_hard = linalg.block_diag(Fxtot, Futot)

        # Add slack if need
        if self.slacks == True:
            nc_x = self.Fx.shape[0]  # add slack only for state constraints
            # Fist add add slack to existing constraints
            addSlack = np.zeros((F_hard.shape[0], nc_x * self.N))
            addSlack[0:nc_x * (self.N), 0:nc_x * (self.N)] = -np.eye(nc_x * (self.N))
            # Now constraint slacks >= 0
            I = - np.eye(nc_x * self.N)
            Zeros = np.zeros((nc_x * self.N, F_hard.shape[1]))
            Positivity = np.hstack((Zeros, I))

            # Let's stack all together
            self.F = np.vstack((np.hstack((F_hard, addSlack)), Positivity))
            self.b = np.hstack((bxtot, butot, np.zeros(nc_x * self.N)))
        else:
            self.F = F_hard
            self.b = np.hstack((bxtot, butot))

    def computeLTVdynamics(self):
        # Estimate system dynamics
        self.A = []
        self.B = []
        self.C = []
        for i in range(0, self.N):
            Ai, Bi, Ci = PredictiveModel.regressionAndLinearization(self.xLin[i], self.uLin[i])
            self.A.append(Ai)
            self.B.append(Bi)
            self.C.append(Ci)

    def buildEqConstr(self):
        # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
        # The equality constraint is: G*z = E * x(t) + L
        Gx = np.eye(self.dim_state * (self.N + 1))
        Gu = np.zeros((self.dim_state * (self.N + 1), self.dim_input * (self.N)))

        E = np.zeros((self.dim_state * (self.N + 1), self.dim_state))
        E[np.arange(self.dim_state)] = np.eye(self.dim_state)

        L = np.zeros(self.dim_state * (self.N + 1))

        for i in range(0, self.N):
            Gx[(self.dim_state + i * self.dim_state):(self.dim_state + i * self.dim_state + self.dim_state), (i * self.dim_state):(i * self.dim_state + self.dim_state)] = -self.A
            Gu[(self.dim_state + i * self.dim_state):(self.dim_state + i * self.dim_state + self.dim_state), (i * self.dim_input):(i * self.dim_input + self.dim_input)] = -self.B

        if self.slacks == True:
            self.G = np.hstack((Gx, Gu, np.zeros((Gx.shape[0], self.Fx.shape[0] * self.N))))
        else:
            self.G = np.hstack((Gx, Gu))

        self.E = E
        self.L = L

    def buildCost(self):
        # The cost is: (1/2) * z' H z + q' z
        listQ = [self.Q] * (self.N)
        Hx = linalg.block_diag(*listQ)

        listTotR = [self.R + 2 * np.diag(self.dR)] * (self.N)  # Need to add dR for the derivative input cost
        Hu = linalg.block_diag(*listTotR)
        # Need to condider that the last input appears just once in the difference
        for i in range(0, self.dim_input):
            Hu[i - self.dim_input, i - self.dim_input] = Hu[i - self.dim_input, i - self.dim_input] - self.dR[i]

        # Derivative Input Cost
        OffDiaf = -np.tile(self.dR, self.N - 1)
        np.fill_diagonal(Hu[self.dim_input:], OffDiaf)
        np.fill_diagonal(Hu[:, self.dim_input:], OffDiaf)

        # Cost linear term for state and input
        q = - 2 * np.dot(np.append(np.tile(self.xRef, self.N + 1), np.zeros(self.R.shape[0] * self.N)),
                         linalg.block_diag(Hx, self.Qm, Hu))
        # Derivative Input (need to consider input at previous time step)
        q[self.dim_state * (self.N + 1):self.dim_state * (self.N + 1) + self.dim_input] = -2 * np.dot(self.OldInput, np.diag(self.dR))
        if self.slacks == True:
            quadSlack = self.Qslack[0] * np.eye(self.Fx.shape[0] * self.N)
            linSlack = self.Qslack[1] * np.ones(self.Fx.shape[0] * self.N)
            self.H = linalg.block_diag(Hx, self.Qm, Hu, quadSlack)
            self.q = np.append(q, linSlack)
        else:
            self.H = linalg.block_diag(Hx, self.Qm, Hu)
            self.q = q

        self.H = 2 * self.H  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    def osqp_solve_qp(self, P, q, G=None, h=None, A=None, b=None, initvals=None):
        """
        Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
        using OSQP <https://github.com/oxfordcontrol/osqp>.
        """
        self.osqp = OSQP()
        qp_A = vstack([G, A]).tocsc()
        l = -np.inf * np.ones(len(h))
        qp_l = np.hstack([l, b])
        qp_u = np.hstack([h, b])

        self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
        if initvals is not None:
            self.osqp.warm_start(x=initvals)
        res = self.osqp.solve()
        if res.info.status_val == 1:
            self.feasible = 1
        else:
            self.feasible = 0
        self.Solution = res.x


    # class LMPC(OCTP):
    #     def __init__(self, numSS_Points, numSS_it, QterminalSlack, mpcPrameters, predictiveModel, dt=0.1):
    #         super().__init__(mpcPrameters, predictiveModel)
    #         self.numSS_Points = numSS_Points
    #         self.numSS_it = numSS_it
    #         self.QterminalSlack = QterminalSlack
    #
    #         self.OldInput = np.zeros((1, 2))
    #         self.xPred = []
    #
    #         # Initialize the following quantities to avoid dynamic allocation
    #         self.LapTime = []  # Time at which each j-th iteration is completed
    #         self.SS = []  # Sampled Safe SS
    #         self.uSS = []  # Input associated with the points in SS
    #         self.Qfun = []  # Qfun: cost-to-go from each point in SS
    #         self.SS_glob = []  # SS in global (X-Y) used for plotting
    #
    #         self.xStoredPredTraj = []
    #         self.xStoredPredTraj_it = []
    #         self.uStoredPredTraj = []
    #         self.uStoredPredTraj_it = []
    #         self.SSStoredPredTraj = []
    #         self.SSStoredPredTraj_it = []
    #
    #         self.zt = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0])
    #
    #         # Initialize the controller iteration
    #         self.it = 0
    #         self.buildIneqConstr()
    #         self.buildCost()
    #         self.addSafeSetIneqConstr()
    #
    #     def solvelmpc(self, x0):
    #         if (self.zt[4] - x0[4] > self.predictiveModel.map.TrackLength / 2):
    #             self.zt[4] = np.max([self.zt[4] - self.predictiveModel.map.TrackLength, 0])
    #             self.xLin[4, -1] = self.xLin[4, -1] - self.predictiveModel.map.TrackLength
    #         sortedLapTime = np.argsort(np.array(self.LapTime))
    #
    #         # Select Points from historical data. These points will be used to construct the terminal cost function and constraint set
    #         SS_PointSelectedTot = np.empty((self.n, 0))
    #         Succ_SS_PointSelectedTot = np.empty((self.n, 0))
    #         Succ_uSS_PointSelectedTot = np.empty((self.d, 0))
    #         Qfun_SelectedTot = np.empty((0))
    #         for jj in sortedLapTime[0:self.numSS_it]:
    #             SS_PointSelected, uSS_PointSelected, Qfun_Selected = self.selectPoints(jj, self.zt,
    #                                                                                    self.numSS_Points / self.numSS_it + 1)
    #             Succ_SS_PointSelectedTot = np.append(Succ_SS_PointSelectedTot, SS_PointSelected[:, 1:], axis=1)
    #             Succ_uSS_PointSelectedTot = np.append(Succ_uSS_PointSelectedTot, uSS_PointSelected[:, 1:], axis=1)
    #             SS_PointSelectedTot = np.append(SS_PointSelectedTot, SS_PointSelected[:, 0:-1], axis=1)
    #             Qfun_SelectedTot = np.append(Qfun_SelectedTot, Qfun_Selected[0:-1], axis=0)
    #
    #         self.Succ_SS_PointSelectedTot = Succ_SS_PointSelectedTot
    #         self.Succ_uSS_PointSelectedTot = Succ_uSS_PointSelectedTot
    #         self.SS_PointSelectedTot = SS_PointSelectedTot
    #         self.Qfun_SelectedTot = Qfun_SelectedTot
    #
    #         self.addSafeSetEqConstr()
    #         self.addSafeSetCost()
    #
    #         startTimer = datetime.datetime.now()
    #         self.osqp_solve_qp(self.H, self.q, self.G_in, np.add(self.w_in, np.dot(self.E_in, x0)), self.G_eq,
    #                            np.add(np.dot(self.E_eq, x0), self.C_eq))
    #         endTimer = datetime.datetime.now();
    #         deltaTimer = endTimer - startTimer
    #         self.solverTime = deltaTimer
    #
    #         # Unpack Solution
    #         self.unpackSolution(x0)
    #         self.time += 1
    #
    #         return self.uPred[0, :]
    #
    #
    #     def addSafeSetIneqConstr(self):
    #         # Add positiviti constraints for lambda_{SafeSet}. Note that no constraint is enforced on slack_{SafeSet} ---> add np.hstack(-np.eye(self.numSS_Points), np.zeros(self.n))
    #         self.F_FTOCP = sparse.csc_matrix(
    #             linalg.block_diag(self.F,
    #                               np.hstack((-np.eye(self.numSS_Points), np.zeros((self.numSS_Points, self.n))))))
    #         self.b_FTOCP = np.append(self.b, np.zeros(self.numSS_Points))
    #
    #     def addSafeSetEqConstr(self):
    #         # Add constrains for x, u, slack
    #         xTermCons = np.zeros((self.n, self.G.shape[1]))
    #         xTermCons[:, self.N * self.n:(self.N + 1) * self.n] = np.eye(self.n)
    #         G_x_u_slack = np.vstack((self.G, xTermCons))
    #         # Constraint for lambda_{SaFeSet, slack_{safeset}} to enforce safe set
    #         G_lambda_slackSafeSet = np.vstack((np.zeros((self.G.shape[0], self.SS_PointSelectedTot.shape[1] + self.n)),
    #                                            np.hstack((-self.SS_PointSelectedTot, np.eye(self.n)))))
    #         # Constraints on lambda = 1
    #         G_lambda = np.append(np.append(np.zeros(self.G.shape[1]), np.ones(self.SS_PointSelectedTot.shape[1])),
    #                              np.zeros(self.n))
    #         # Put all together
    #         self.G_FTOCP = sparse.csc_matrix(np.vstack((np.hstack((G_x_u_slack, G_lambda_slackSafeSet)), G_lambda)))
    #         self.E_FTOCP = np.vstack(
    #             (self.E, np.zeros((self.n + 1, self.n))))  # adding n for terminal constraint and 1 for lambda = 1
    #         self.L_FTOCP = np.append(np.append(self.L, np.zeros(self.n)), 1)
    #
    #     def addSafeSetCost(self):
    #         # need to multiply the quadratic term as cost is (1/2) z'*Q*z
    #         self.H_FTOCP = sparse.csc_matrix(
    #             linalg.block_diag(self.H,
    #                               np.zeros((self.SS_PointSelectedTot.shape[1], self.SS_PointSelectedTot.shape[1])),
    #                               2 * self.QterminalSlack))
    #         self.q_FTOCP = np.append(np.append(self.q, self.Qfun_SelectedTot), np.zeros(self.n))
    #
    #     def unpackSolution(self):
    #         stateIdx = self.n * (self.N + 1)
    #         inputIdx = stateIdx + self.d * self.N
    #         slackIdx = inputIdx + self.Fx.shape[0] * self.N
    #         lambdIdx = slackIdx + self.SS_PointSelectedTot.shape[1]
    #         sTermIdx = lambdIdx + self.n
    #
    #         self.xPred = np.squeeze(
    #             np.transpose(np.reshape((self.Solution[np.arange(self.n * (self.N + 1))]), (self.N + 1, self.n)))).T
    #         self.uPred = np.squeeze(np.transpose(
    #             np.reshape((self.Solution[self.n * (self.N + 1) + np.arange(self.d * self.N)]), (self.N, self.d)))).T
    #         self.slack = self.Solution[inputIdx:slackIdx]
    #         self.lambd = self.Solution[slackIdx:lambdIdx]
    #         self.slackTerminal = self.Solution[lambdIdx:]
    #
    #         self.xStoredPredTraj_it.append(self.xPred)
    #         self.uStoredPredTraj_it.append(self.uPred)
    #         self.SSStoredPredTraj_it.append(self.SS_PointSelectedTot.T)
    #
    #     def feasibleStateInput(self):
    #         self.zt = np.dot(self.Succ_SS_PointSelectedTot, self.lambd)
    #         self.zt_u = np.dot(self.Succ_uSS_PointSelectedTot, self.lambd)
    #
    #     # def addTerminalComponents(self, x0):
    #     #     """add terminal constraint and terminal cost
    #     #     Arguments:
    #     #         x: initial condition
    #     #     """
    #     #     # Update zt and xLin is they have crossed the finish line. We want s \in [0, TrackLength]
    #     #     if (self.zt[4] - x0[4] > self.predictiveModel.map.TrackLength / 2):
    #     #         self.zt[4] = np.max([self.zt[4] - self.predictiveModel.map.TrackLength, 0])
    #     #         self.xLin[4, -1] = self.xLin[4, -1] - self.predictiveModel.map.TrackLength
    #     #     sortedLapTime = np.argsort(np.array(self.LapTime))
    #     #
    #     #     # Select Points from historical data. These points will be used to construct the terminal cost function and constraint set
    #     #     SS_PointSelectedTot = np.empty((self.n, 0))
    #     #     Succ_SS_PointSelectedTot = np.empty((self.n, 0))
    #     #     Succ_uSS_PointSelectedTot = np.empty((self.d, 0))
    #     #     Qfun_SelectedTot = np.empty((0))
    #     #     for jj in sortedLapTime[0:self.numSS_it]:
    #     #         SS_PointSelected, uSS_PointSelected, Qfun_Selected = self.selectPoints(jj, self.zt,
    #     #                                                                                self.numSS_Points / self.numSS_it + 1)
    #     #         Succ_SS_PointSelectedTot = np.append(Succ_SS_PointSelectedTot, SS_PointSelected[:, 1:], axis=1)
    #     #         Succ_uSS_PointSelectedTot = np.append(Succ_uSS_PointSelectedTot, uSS_PointSelected[:, 1:], axis=1)
    #     #         SS_PointSelectedTot = np.append(SS_PointSelectedTot, SS_PointSelected[:, 0:-1], axis=1)
    #     #         Qfun_SelectedTot = np.append(Qfun_SelectedTot, Qfun_Selected[0:-1], axis=0)
    #     #
    #     #     self.Succ_SS_PointSelectedTot = Succ_SS_PointSelectedTot
    #     #     self.Succ_uSS_PointSelectedTot = Succ_uSS_PointSelectedTot
    #     #     self.SS_PointSelectedTot = SS_PointSelectedTot
    #     #     self.Qfun_SelectedTot = Qfun_SelectedTot
    #     #
    #     #     # Update terminal set and cost
    #     #     self.addSafeSetEqConstr()
    #     #     self.addSafeSetCost()
    #
    #     def addTrajectory(self, x, u, x_glob):
    #         """update iteration index and construct SS, uSS and Qfun
    #         Arguments:
    #             x: closed-loop trajectory
    #             u: applied inputs
    #             x_gloab: closed-loop trajectory in global frame
    #         """
    #         self.LapTime.append(x.shape[0])
    #         self.SS.append(x)
    #         self.SS_glob.append(x_glob)
    #         self.uSS.append(u)
    #         self.Qfun.append(self.computeCost(x, u))
    #
    #         if self.it == 0:
    #             self.xLin = self.SS[self.it][1:self.N + 2, :]
    #             self.uLin = self.uSS[self.it][1:self.N + 1, :]
    #
    #         self.xStoredPredTraj.append(self.xStoredPredTraj_it)
    #         self.xStoredPredTraj_it = []
    #
    #         self.uStoredPredTraj.append(self.uStoredPredTraj_it)
    #         self.uStoredPredTraj_it = []
    #
    #         self.SSStoredPredTraj.append(self.SSStoredPredTraj_it)
    #         self.SSStoredPredTraj_it = []
    #
    #         self.it = self.it + 1
    #         self.timeStep = 0
    #
    #     def computeCost(self, x, u):
    #         """compute roll-out cost
    #         Arguments:
    #             x: closed-loop trajectory
    #             u: applied inputs
    #         """
    #         Cost = 10000 * np.ones((x.shape[0]))  # The cost has the same elements of the vector x --> time +1
    #         # Now compute the cost moving backwards in a Dynamic Programming (DP) fashion.
    #         # We start from the last element of the vector x and we sum the running cost
    #         for i in range(0, x.shape[0]):
    #             if (i == 0):  # Note that for i = 0 --> pick the latest element of the vector x
    #                 Cost[x.shape[0] - 1 - i] = 0
    #             elif x[x.shape[0] - 1 - i, 4] < self.predictiveModel.map.TrackLength:
    #                 Cost[x.shape[0] - 1 - i] = Cost[x.shape[0] - 1 - i + 1] + 1
    #             else:
    #                 Cost[x.shape[0] - 1 - i] = 0
    #
    #         return Cost
    #
    #     def addPoint(self, x, u):
    #         """at iteration j add the current point to SS, uSS and Qfun of the previous iteration
    #         Arguments:
    #             x: current state
    #             u: current input
    #         """
    #         self.SS[self.it - 1] = np.append(self.SS[self.it - 1], np.array(
    #             [x + np.array([0, 0, 0, 0, self.predictiveModel.map.TrackLength, 0])]), axis=0)
    #         self.uSS[self.it - 1] = np.append(self.uSS[self.it - 1], np.array([u]), axis=0)
    #         self.Qfun[self.it - 1] = np.append(self.Qfun[self.it - 1], self.Qfun[self.it - 1][-1] - 1)
    #         # The above two lines are needed as the once the predicted trajectory has crossed the finish line the goal is
    #         # to reach the end of the lap which is about to start
    #
    #     def selectPoints(self, it, zt, numPoints):
    #         """selecte (numPoints)-nearest neivbor to zt. These states will be used to construct the safe set and the value function approximation
    #         Arguments:
    #             x: current state
    #             u: current input
    #         """
    #         x = self.SS[it]
    #         u = self.uSS[it]
    #         oneVec = np.ones((x.shape[0], 1))
    #         x0Vec = (np.dot(np.array([zt]).T, oneVec.T)).T
    #         diff = x - x0Vec
    #         norm = la.norm(diff, 1, axis=1)
    #         MinNorm = np.argmin(norm)
    #
    #         if (MinNorm - numPoints / 2 >= 0):
    #             indexSSandQfun = range(-int(numPoints / 2) + MinNorm, int(numPoints / 2) + MinNorm + 1)
    #         else:
    #             indexSSandQfun = range(MinNorm, MinNorm + int(numPoints))
    #
    #         SS_Points = x[indexSSandQfun, :].T
    #         SSu_Points = u[indexSSandQfun, :].T
    #         Sel_Qfun = self.Qfun[it][indexSSandQfun]
    #
    #         # Modify the cost if the predicion has crossed the finisch line
    #         if self.xPred == []:
    #             Sel_Qfun = self.Qfun[it][indexSSandQfun]
    #         elif (np.all((self.xPred[:, 4] > self.predictiveModel.map.TrackLength) == False)):
    #             Sel_Qfun = self.Qfun[it][indexSSandQfun]
    #         elif it < self.it - 1:
    #             Sel_Qfun = self.Qfun[it][indexSSandQfun] + self.Qfun[it][0]
    #         else:
    #             sPred = self.xPred[:, 4]
    #             predCurrLap = self.N - sum(sPred > self.predictiveModel.map.TrackLength)
    #             currLapTime = self.timeStep
    #             Sel_Qfun = self.Qfun[it][indexSSandQfun] + currLapTime + predCurrLap
    #
    #         return SS_Points, SSu_Points, Sel_Qfun