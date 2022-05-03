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
import math
import datetime


solvers.options['show_progress'] = False



def Compute_Q_M(SS, uSS, indexSelected, stateFeatures, inputFeatures, usedIt, np, matrix, lamb, K):
    Counter = 0
    it = 1
    X0   = np.empty((0,len(stateFeatures)+len(inputFeatures)))
    Ktot = np.empty((0))

    for it in usedIt:
        X0 = np.append(X0, np.hstack((np.squeeze(SS[np.ix_(indexSelected[Counter], stateFeatures, [it])]), np.squeeze(uSS[np.ix_(indexSelected[Counter], inputFeatures, [it])], axis=2))), axis=0)
        Ktot = np.append(Ktot, K[Counter])
        Counter = Counter + 1
    M = np.hstack((X0, np.ones((X0.shape[0], 1))))
    Q0 = np.dot(np.dot(M.T, np.diag(Ktot)), M)
    Q = matrix(Q0 + lamb * np.eye(Q0.shape[0]))

    return Q, M


def Compute_b(SS, yIndex, usedIt, matrix, M, indexSelected, K, np):
    counter = 0
    y = np.empty((0))
    Ktot = np.empty((0))

    for it in usedIt:
        y = np.append(y, np.squeeze(SS[np.ix_(indexSelected[counter] + 1, [yIndex], [it])]))
        Ktot = np.append(Ktot, K[counter])
        counter = counter + 1

    b = matrix(-np.dot(np.dot(M.T, np.diag(Ktot)), y))

    return b


def LMPC_LocLinReg(Q, b, stateFeatures, inputFeatures, qp):
    startTimer = datetime.datetime.now()
    res_cons = qp(Q, b)
    endTimer = datetime.datetime.now()
    deltaTimer_tv = endTimer - startTimer

    Result = np.squeeze(np.array(res_cons['x']))
    A = Result[0:len(stateFeatures)]
    B = Result[len(stateFeatures):(len(stateFeatures)+len(inputFeatures))]
    C = Result[-1]

    return A, B, C


def ComputeIndex(h, SS, uSS, LapCounter, it, x0, stateFeatures, scaling, MaxNumPoint, ConsiderInput):
    oneVec = np.ones( (SS[0:LapCounter[it], :, it].shape[0]-1, 1) )
    x0Vec = (np.dot(np.array([x0]).T, oneVec.T)).T

    if ConsiderInput == 1:
        DataMatrix = np.hstack((SS[0:LapCounter[it]-1, stateFeatures, it], uSS[0:LapCounter[it]-1, :, it]))
    else:
        DataMatrix = SS[0:LapCounter[it]-1, stateFeatures, it]

    diff = np.dot(( DataMatrix - x0Vec ), scaling)
    norm = la.norm(diff, 1, axis=1)
    indexTot = np.squeeze(np.where(norm < h))
    if (indexTot.shape[0] >= MaxNumPoint):
        index = np.argsort(norm)[0:MaxNumPoint]
    else:
        index = indexTot

    K = (1 - (norm[index] / h)**2) * 3/4

    return index, K


def RegressionAndLinearization(LinPoints, LinInput, usedIt, SS, uSS, LapCounter, MaxNumPoint, qp, n, d, matrix, PointAndTangent, dt, i):
    x0 = LinPoints[i, :]
    Ai = np.zeros((n, n))
    Bi = np.zeros((n, d))
    Ci = np.zeros((n, 1))

    h = 5
    lamb = 0.0
    stateFeatures = [0, 1, 2]
    ConsiderInput = 1

    if ConsiderInput == 1:
        scaling = np.array([[0.1, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])
        xLin = np.hstack((LinPoints[i, stateFeatures], LinInput[i, :]))
    else:
        scaling = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
        xLin = LinPoints[i, stateFeatures]

    indexSelected = []
    K = []
    for ii in usedIt:
        indexSelected_i, K_i = ComputeIndex(h, SS, uSS, LapCounter, ii, xLin, stateFeatures, scaling, MaxNumPoint,
                                            ConsiderInput)
        indexSelected.append(indexSelected_i)
        K.append(K_i)

    inputFeatures = [1]
    Q_vx, M_vx = Compute_Q_M(SS, uSS, indexSelected, stateFeatures, inputFeatures, usedIt, np, matrix, lamb, K)

    yIndex = 0
    b = Compute_b(SS, yIndex, usedIt, matrix, M_vx, indexSelected, K, np)
    Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex] = LMPC_LocLinReg(Q_vx, b, stateFeatures,
                                                                                      inputFeatures, qp)

    inputFeatures = [0]
    Q_lat, M_lat = Compute_Q_M(SS, uSS, indexSelected, stateFeatures, inputFeatures, usedIt, np, matrix, lamb, K)

    yIndex = 1  # vy
    b = Compute_b(SS, yIndex, usedIt, matrix, M_lat, indexSelected, K, np)
    Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex] = LMPC_LocLinReg(Q_lat, b, stateFeatures,
                                                                                      inputFeatures, qp)

    yIndex = 2  # wz
    b = Compute_b(SS, yIndex, usedIt, matrix, M_lat, indexSelected, K, np)
    Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex] = LMPC_LocLinReg(Q_lat, b, stateFeatures,
                                                                                      inputFeatures, qp)
    # ===== Linearization =======
    vx = x0[0]
    vy = x0[1]
    wz = x0[2]
    epsi = x0[3]
    s = x0[4]
    ey = x0[5]

    if s < 0:
        print("s is negative, here the state: \n", LinPoints)

    startTimer = datetime.datetime.now()
    cur = Curvature(s, PointAndTangent)
    den = 1 - cur * ey

    depsi_vx = -dt * np.cos(epsi) / den * cur
    depsi_vy = dt * np.sin(epsi) / den * cur
    depsi_wz = dt
    depsi_epsi = 1 - dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den * cur
    depsi_s = 0
    depsi_ey = dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * cur * (-cur)

    Ai[3, :] = [depsi_vx, depsi_vy, depsi_wz, depsi_epsi, depsi_s, depsi_ey]
    Ci[3] = epsi + dt * (wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur) - np.dot(Ai[3, :], x0)

    ds_vx = dt * (np.cos(epsi) / den)
    ds_vy = -dt * (np.sin(epsi) / den)
    ds_wz = 0
    ds_epsi = dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den
    ds_s = 1  # + Ts * (Vx * cos(epsi) - Vy * sin(epsi)) / (1 - ey * rho) ^ 2 * (-ey * drho);
    ds_ey = -dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den * 2) * (-cur)

    Ai[4, :] = [ds_vx, ds_vy, ds_wz, ds_epsi, ds_s, ds_ey]
    Ci[4] = s + dt * ((vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey)) - np.dot(Ai[4, :], x0)

    dey_vx = dt * np.sin(epsi)
    dey_vy = dt * np.cos(epsi)
    dey_wz = 0
    dey_epsi = dt * (vx * np.cos(epsi) - vy * np.sin(epsi))
    dey_s = 0
    dey_ey = 1

    Ai[5, :] = [dey_vx, dey_vy, dey_wz, dey_epsi, dey_s, dey_ey]
    Ci[5] = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi)) - np.dot(Ai[5, :], x0)

    endTimer = datetime.datetime.now()
    deltaTimer_tv = endTimer - startTimer

    return Ai, Bi, Ci, indexSelected


class LMPCplus():
    def __init__(self, dim_state, dim_input, N, Q, R, Qf, Fx, bx, Fu, bu, dt, dR, numSS_Points, numSS_it, QterminalSlack, map, Laps, TimeLMPC, Solver):
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.N = N
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.Fx = Fx
        self.bx = bx
        self.Fu = Fu
        self.bu = bu
        self.dt = dt
        self.dR = dR
        self.numSS_Points = numSS_Points
        self.numSS_it = numSS_it
        self.QterminalSlack = QterminalSlack
        self.map = map
        self.Solver = Solver

        self.LapTime = 0
        self.OldInput = np.zeros((1, 2))
        self.xPred = []
        self.zt = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0])
        self.itUsedSYsID = 1

        NumPoints = int(TimeLMPC / dt) + 1
        self.LapCounter = 10000 * np.ones(Laps).astype(int)
        self.TimeSS = 10000 * np.ones(Laps).astype(int)
        self.SS = 10000 * np.ones((NumPoints, 6, Laps))       # Sampled Safe SS
        self.uSS = 10000 * np.ones((NumPoints, 2, Laps))      # Input associated with the points in SS
        self.Qfun = 0 * np.ones((NumPoints, Laps))            # Qfun: cost-to-go from each point in SS
        self.SS_glob = 10000 * np.ones((NumPoints, 6, Laps))  # SS in global (X-Y) used for plotting

        # Initialize the controller iteration
        self.it = 0
        self.F, self.b = self.buildIneqConstr()

    def solve(self, x0, uOld = np.zeros([0, 0])):
        n = self.dim_state
        d = self.dim_input
        N = self.N
        F = self.F
        b = self.b
        SS = self.SS
        Qfun = self.Qfun
        uSS = self.uSS
        TimeSS = self.TimeSS
        Q = self.Q
        R = self.R
        dR =self.dR
        OldInput = self.OldInput
        dt = self.dt
        it = self.it
        numSS_Points = self.numSS_Points
        Qslack = self.QterminalSlack

        LinPoints = self.LinPoints
        LinInput  = self.LinInput
        map = self.map

        # Select Points from SS
        if (self.zt[4]-x0[4] > map.TrackLength/2):
            self.zt[4] = np.max([self.zt[4] - map.TrackLength,0])
            self.LinPoints[4,-1] = self.LinPoints[4,-1] - map.TrackLength

        sortedLapTime = np.argsort(self.Qfun[0, 0:it])

        SS_PointSelectedTot = np.empty((n, 0))
        Succ_SS_PointSelectedTot = np.empty((n, 0))
        Succ_uSS_PointSelectedTot = np.empty((d, 0))
        Qfun_SelectedTot = np.empty((0))
        for jj in sortedLapTime[0:self.numSS_it]:
            SS_PointSelected, uSS_PointSelected, Qfun_Selected = self.SelectPoints(jj, self.zt, numSS_Points / self.numSS_it + 1)
            Succ_SS_PointSelectedTot =  np.append(Succ_SS_PointSelectedTot, SS_PointSelected[:,1:], axis=1)
            Succ_uSS_PointSelectedTot =  np.append(Succ_uSS_PointSelectedTot, uSS_PointSelected[:,1:], axis=1)
            SS_PointSelectedTot      = np.append(SS_PointSelectedTot, SS_PointSelected[:,0:-1], axis=1)
            Qfun_SelectedTot         = np.append(Qfun_SelectedTot, Qfun_Selected[0:-1], axis=0)

        self.SS_PointSelectedTot = SS_PointSelectedTot
        self.Qfun_SelectedTot = Qfun_SelectedTot

        startTimer = datetime.datetime.now()
        Atv, Btv, Ctv, indexUsed_list = self.EstimateABC(sortedLapTime)
        endTimer = datetime.datetime.now()
        deltaTimer = endTimer - startTimer
        L, npG, npE = self.buildEqConst(Atv, Btv, Ctv, N, n, d)
        self.linearizationTime = deltaTimer

        # Build Terminal cost and Constraint
        G, E = self.TermConstr(npG, npE, N, n, d, SS_PointSelectedTot)
        M, q = self.buildCost(Qfun_SelectedTot, numSS_Points, N, Qslack, Q, R, dR, OldInput)

        # Solve QP
        startTimer = datetime.datetime.now()

        if self.Solver == "CVX":
            res_cons = qp(M, matrix(q), F, matrix(b), G, E * matrix(x0) + L)
            if res_cons['status'] == 'optimal':
                feasible = 1
            else:
                feasible = 0
                print(res_cons['status'])
            Solution = np.squeeze(res_cons['x'])

        elif self.Solver == "OSQP":
            res_cons, feasible = self.osqp_solve_qp(sparse.csr_matrix(M), q, sparse.csr_matrix(F), b, sparse.csr_matrix(G), np.add(np.dot(E,x0),L[:,0]))
            Solution = res_cons.x

        self.feasible = feasible

        endTimer = datetime.datetime.now()
        deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer

        # Extract solution and set linerizations points
        xPred, uPred, lambd, slack = self.GetPred(Solution, n, d, N, np)

        self.zt = np.dot(Succ_SS_PointSelectedTot, lambd)
        self.uVector = np.dot(Succ_uSS_PointSelectedTot, lambd)

        self.xPred = xPred.T
        if self.N == 1:
            self.uPred    = np.array([[uPred[0], uPred[1]]])
            self.LinInput =  np.array([[uPred[0], uPred[1]]])
        else:
            self.uPred = uPred.T
            self.LinInput = np.vstack((uPred.T[1:, :], self.uVector))

        self.LinPoints = np.vstack((xPred.T[1:,:], self.zt))

        self.OldInput = uPred.T[0,:]

    def addTrajectory(self, ClosedLoopData):

        it = self.it
        self.TimeSS[it] = ClosedLoopData.SimTime
        self.LapCounter[it] = ClosedLoopData.SimTime
        self.SS[0:(self.TimeSS[it] + 1), :, it] = ClosedLoopData.x[0:(self.TimeSS[it] + 1), :]
        self.SS_glob[0:(self.TimeSS[it] + 1), :, it] = ClosedLoopData.x_glob[0:(self.TimeSS[it] + 1), :]
        self.uSS[0:self.TimeSS[it], :, it] = ClosedLoopData.u[0:(self.TimeSS[it]), :]
        self.Qfun[0:(self.TimeSS[it] + 1), it]  = self.ComputeCost(ClosedLoopData.x[0:(self.TimeSS[it] + 1), :],
                                                                   ClosedLoopData.u[0:(self.TimeSS[it]), :], self.map.TrackLength)
        for i in np.arange(0, self.Qfun.shape[0]):
            if self.Qfun[i, it] == 0:
                self.Qfun[i, it] = self.Qfun[i - 1, it] - 1

        if self.it == 0:
            self.LinPoints = self.SS[1:self.N + 2, :, it]
            self.LinInput  = self.uSS[1:self.N + 1, :, it]

        self.it = self.it + 1

    def addPoint(self, x, u):
        """at iteration j add the current point to SS, uSS and Qfun of the previous iteration
        Arguments:
            x: current state
            u: current input
            i: at the j-th iteration i is the time at which (x,u) are recorded
        """
        Counter = self.TimeSS[self.it - 1]
        self.SS[Counter, :, self.it - 1] = x + np.array([0, 0, 0, 0, self.map.TrackLength, 0])
        self.uSS[Counter, :, self.it - 1] = u

        # The above two lines are needed as the once the predicted trajectory has crossed the finish line the goal is
        # to reach the end of the lap which is about to start
        if self.Qfun[Counter, self.it - 1] == 0:
            self.Qfun[Counter, self.it - 1] = self.Qfun[Counter, self.it - 1] - 1
        self.TimeSS[self.it - 1] = self.TimeSS[self.it - 1] + 1

    def update(self, SS, uSS, Qfun, TimeSS, it, LinPoints, LinInput):
        self.SS  = SS
        self.uSS = uSS
        self.Qfun  = Qfun
        self.TimeSS  = TimeSS
        self.it = it
        self.LinPoints = LinPoints
        self.LinInput  = LinInput

    def buildIneqConstr(self):
        rep_a = [self.Fx] * (self.N)
        Mat = linalg.block_diag(*rep_a)
        NoTerminalConstr = np.zeros((np.shape(Mat)[0], self.dim_state))
        Fxtot = np.hstack((Mat, NoTerminalConstr))
        bxtot = np.tile(np.squeeze(self.bx), self.N)

        rep_b = [self.Fu] * (self.N)
        Futot = linalg.block_diag(*rep_b)
        butot = np.tile(np.squeeze(self.bu), self.N)

        rFxtot, cFxtot = np.shape(Fxtot)
        rFutot, cFutot = np.shape(Futot)
        a1 = np.hstack((Fxtot, np.zeros((rFxtot, cFutot))))
        a2 = np.hstack((np.zeros((rFutot, cFxtot)), Futot))

        FDummy = np.vstack((a1, a2))
        I = -np.eye(self.numSS_Points)
        FDummy2 = linalg.block_diag(FDummy, I)
        Fslack = np.zeros((FDummy2.shape[0], self.dim_state))
        F_hard = np.hstack((FDummy2, Fslack))

        LaneSlack = np.zeros((F_hard.shape[0], 2 * self.N))
        colIndexPositive = []
        rowIndexPositive = []
        colIndexNegative = []
        rowIndexNegative = []
        for i in range(0, self.N):
            colIndexPositive.append(i * 2 + 0)
            colIndexNegative.append(i * 2 + 1)
            rowIndexPositive.append(i * self.Fx.shape[0] + 0)
            rowIndexNegative.append(i * self.Fx.shape[0] + 1)

        LaneSlack[rowIndexPositive, colIndexPositive] = -1.0
        LaneSlack[rowIndexNegative, rowIndexNegative] = -1.0

        gam = np.hstack((F_hard, LaneSlack))
        I = - np.eye(2 * self.N)
        Zeros = np.zeros((2 *self.N, F_hard.shape[1]))
        Positivity = np.hstack((Zeros, I))
        F = np.vstack((gam, Positivity))

        b_1 = np.hstack((bxtot, butot, np.zeros(self.numSS_Points)))
        b = np.hstack((b_1, np.zeros(2 * self.N)))

        if self.Solver == "CVX":
            F_sparse = spmatrix(F[np.nonzero(F)], np.nonzero(F)[0].astype(int), np.nonzero(F)[1].astype(int), F.shape)
            F_return = F_sparse
        else:
            F_return = F

        return F_return, b

    def GetPred(self, Solution, n, d, N, np):
        xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(n * (N + 1))]), (N + 1, n))))
        uPred = np.squeeze(np.transpose(np.reshape((Solution[n * (N + 1) + np.arange(d * N)]), (N, d))))
        lambd = Solution[(n * (N + 1) + d * N):(Solution.shape[0] - n - 2 * N)]
        slack = Solution[Solution.shape[0] - n - 2 * N:Solution.shape[0] - 2 * N]
        laneSlack = Solution[Solution.shape[0] - 2 * N:]

        return xPred, uPred, lambd, slack

    def TermConstr(self, G, E, N, n, d, SS_Points):
        TermCons = np.zeros((n, (N + 1) * n + N * d))
        TermCons[:, N * n:(N + 1) * n] = np.eye(n)

        G_enlarged = np.vstack((G, TermCons))

        G_lambda = np.zeros((G_enlarged.shape[0], SS_Points.shape[1] + n))
        G_lambda[G_enlarged.shape[0] - n:G_enlarged.shape[0], :] = np.hstack((-SS_Points, np.eye(n)))

        G_LMPC0 = np.hstack((G_enlarged, G_lambda))
        G_ConHull = np.zeros((1, G_LMPC0.shape[1]))
        G_ConHull[-1, G_ConHull.shape[1] - SS_Points.shape[1] - n:G_ConHull.shape[1] - n] = np.ones(
            (1, SS_Points.shape[1]))

        G_LMPC_hard = np.vstack((G_LMPC0, G_ConHull))

        SlackLane = np.zeros((G_LMPC_hard.shape[0], 2 * N))
        G_LMPC = np.hstack((G_LMPC_hard, SlackLane))

        E_LMPC = np.vstack((E, np.zeros((n + 1, n))))

        if self.Solver == "CVX":
            G_LMPC_sparse = spmatrix(G_LMPC[np.nonzero(G_LMPC)], np.nonzero(G_LMPC)[0].astype(int),
                                     np.nonzero(G_LMPC)[1].astype(int), G_LMPC.shape)
            E_LMPC_sparse = spmatrix(E_LMPC[np.nonzero(E_LMPC)], np.nonzero(E_LMPC)[0].astype(int),
                                     np.nonzero(E_LMPC)[1].astype(int), E_LMPC.shape)
            G_LMPC_return = G_LMPC_sparse
            E_LMPC_return = E_LMPC_sparse
        else:
            G_LMPC_return = G_LMPC
            E_LMPC_return = E_LMPC

        return G_LMPC_return, E_LMPC_return

    def buildCost(self, Sel_Qfun, numSS_Points, N, Qslack, Q, R, dR, uOld):
        n = Q.shape[0]
        P = Q
        vt = 2
        b = [Q] * (N)
        Mx = linalg.block_diag(*b)
        c = [R + 2 * np.diag(dR)] * (N)

        Mu = linalg.block_diag(*c)
        Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] = Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] - dR[1]
        Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] = Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] - dR[0]

        OffDiaf = -np.tile(dR, N - 1)
        np.fill_diagonal(Mu[2:], OffDiaf)
        np.fill_diagonal(Mu[:, 2:], OffDiaf)

        M00 = linalg.block_diag(Mx, P, Mu)
        quadLaneSlack = self.Qf[0] * np.eye(2 * self.N)
        M0 = linalg.block_diag(M00, np.zeros((numSS_Points, numSS_Points)), Qslack, quadLaneSlack)

        xtrack = np.array([vt, 0, 0, 0, 0, 0])
        q0 = - 2 * np.dot(np.append(np.tile(xtrack, N + 1), np.zeros(R.shape[0] * N)), M00)
        # Derivative Input
        q0[n * (N + 1):n * (N + 1) + 2] = -2 * np.dot(uOld, np.diag(dR))

        linLaneSlack = self.Qf[1] * np.ones(2 * self.N)

        q = np.append(np.append(np.append(q0, Sel_Qfun), np.zeros(Q.shape[0])), linLaneSlack)
        M = 2 * M0
        if self.Solver == "CVX":
            M_sparse = spmatrix(M[np.nonzero(M)], np.nonzero(M)[0].astype(int), np.nonzero(M)[1].astype(int), M.shape)
            M_return = M_sparse
        else:
            M_return = M

        return M_return, q

    def EstimateABC(self, sortedLapTime):
        ParallelComputation = 0
        Atv = []
        Btv = []
        Ctv = []
        indexUsed_list = []

        usedIt = sortedLapTime[0: self.itUsedSysID]
        MaxNumPoint = 40

        for i in range(0, self.N):
            Ai, Bi, Ci, indexSelected = RegressionAndLinearization(self.LinPoints, self.LinInput, usedIt, self.SS, self.uSS, self.TimeSS,
                                                                   MaxNumPoint, qp, self.dim_state, self.dim_input, matrix, self.map.PointAndTangent, self.dt, i)
            Atv.append(Ai)
            Btv.append(Bi)
            Ctv.append(Ci)
            indexUsed_list.append(indexSelected)

        return Atv, Btv, Ctv, indexUsed_list

    def SelectPoints(self, it, x0, numSS_Points):
        x = self.SS[:, 0:(self.TimeSS[it] - 1), it]
        u = self.uSS[:, 0:(self.TimeSS[it] - 1), it]
        x_glob = self.SS_glob[:, :, it]
        oneVec = np.ones((x.shape[0], 1))
        x0Vec = (np.dot(np.array([x0]).T, oneVec.T)).T
        diff = x - x0Vec
        norm = la.norm(diff, 1, axis=1)
        MinNorm = np.argmin(norm)

        if (MinNorm - numSS_Points / 2 >= 0):
            indexSSandQfun = range(-int(numSS_Points / 2) + MinNorm, int(numSS_Points / 2) + MinNorm + 1)
        else:
            indexSSandQfun = range(int(MinNorm), MinNorm + int(numSS_Points))

        SS_Points = x[indexSSandQfun, :].T
        SSu_Points = u[indexSSandQfun, :].T
        SS_glob_Points = x_glob[indexSSandQfun, :].T
        Sel_Qfun = self.Qfun[indexSSandQfun, it]

        # Modify the cost if the predicion has crossed the finisch line
        if self.xPred == []:
            Sel_Qfun = self.Qfun[indexSSandQfun, it]
        elif (np.all((self.xPred[:, 4] > self.map.TrackLength) == False)):
            Sel_Qfun = self.Qfun[indexSSandQfun, it]
        elif it < self.it - 1:
            Sel_Qfun =self. Qfun[indexSSandQfun, it] + self.Qfun[0, it + 1]
        else:
            sPred = self.xPred[:, 4]
            predCurrLap = self.N - sum(sPred > self.map.TrackLength)
            currLapTime = self.LapTime
            Sel_Qfun = self.Qfun[indexSSandQfun, it] + currLapTime + predCurrLap

        return SS_Points, SSu_Points, Sel_Qfun

    def buildEqConst(self, A, B, C, N, n, d):
        Gx = np.eye(n * (N + 1))
        Gu = np.zeros((n * (N + 1), d * (N)))
        E = np.zeros((n * (N + 1), n))
        E[np.arange(n)] = np.eye(n)

        L = np.zeros((n * (N + 1) + n + 1, 1))
        L[-1] = 1

        for i in range(0, N):
            ind1 = n + i * n + np.arange(n)
            ind2x = i * n + np.arange(n)
            ind2u = i * d + np.arange(d)

            Gx[np.ix_(ind1, ind2x)] = -A[i]
            Gu[np.ix_(ind1, ind2u)] = -B[i]
            L[ind1, :] = C[i]
        G = np.hstack((Gx, Gu))
        if self.Solver == "CVX":
            L_sparse = spmatrix(L[np.nonzero(L)], np.nonzero(L)[0].astype(int), np.nonzero(L)[1].astype(int), L.shape)
            L_return = L_sparse
        else:
            L_return = L

        return L_return, G, E

    def ComputeCost(self, x, u, TrackLength):
        Cost = 10000 * np.ones((x.shape[0]))  # The cost has the same elements of the vector x --> time +1

        for i in range(0, x.shape[0]):
            if (i == 0):  # Note that for i = 0 --> pick the latest element of the vector x
                Cost[x.shape[0] - 1 - i] = 0
            elif x[x.shape[0] - 1 - i, 4] < TrackLength:
                Cost[x.shape[0] - 1 - i] = Cost[x.shape[0] - 1 - i + 1] + 1
            else:
                Cost[x.shape[0] - 1 - i] = 0

        return Cost

    def osqp_solve_qp(self, P, q, G=None, h=None, A=None, b=None, initvals=None):
        osqp = OSQP()
        if G is not None:
            l = -np.inf * np.ones(len(h))
            if A is not None:
                qp_A = np.vstack([G, A]).tocsc()
                qp_l = np.hstack([l, b])
                qp_u = np.hstack([h, b])
            else:  # no equality constraint
                qp_A = G
                qp_l = l
                qp_u = h
            osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
        else:
            osqp.setup(P=P, q=q, A=None, l=None, u=None, verbose=False)
        if initvals is not None:
            osqp.warm_start(x=initvals)
        res = osqp.solve()
        if res.info.status_val != osqp.constant('OSQP_SOLVED'):
            print("OSQP exited with status '%s'" % res.info.status)
        feasible = 0
        if res.info.status_val == osqp.constant('OSQP_SOLVED') or res.info.status_val == osqp.constant(
                'OSQP_SOLVED_INACCURATE') or res.info.status_val == osqp.constant('OSQP_MAX_ITER_REACHED'):
            feasible = 1
        return res, feasible