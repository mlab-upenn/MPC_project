import numpy as np
import pdb
import dill
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import copy
import datetime
import os
import pickle
import sys

from LMPCplus import LMPCplus
from sysmodel import Map, Carmodel, unityTestChangeOfCoordinates
from OPTP import OCTP
from utils import ClosedLoopData, PID, PredictiveModel, Regression
from plot import plotTrajectory, plotClosedLoopLMPC, animation_xy, animation_states, saveGif_xyResults, Save_statesAnimation


def main():
    if not os.path.exists('storedData'):
        os.makedirs('storedData')

    # Parameter initialization
    dt = 1.0 / 10.0   # Controller discretization time
    Time = 100        # Simulation time for PID
    TimeMPC = 100     # Time for LTI-MPC
    TimeMPC_tv = 100  # Time for LTV-MPC
    TimeLMPC = 400    # Time for LMPC
    vt = 0.8          # Reference velocity for path controllers
    v0 = 0.5          # Initial velocity at lap 0
    N = 12            # Horizon
    dim_state = 6     # State dimension
    dim_input = 2     # Input dimension

    Q = np.diag([1.0, 1.0, 1, 1, 0.0, 100.0])             # vx, vy, wz, epsi, s, ey
    R = np.diag([1.0, 10.0])                              # delta, a
    Q_lmpc = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * 0  # vx, vy, wz, epsi, s, ey
    R_lmpc = np.diag([1.0, 1.0]) * 0                      # delta, a
    Qm = np.eye(dim_state) * 1000
    Qf = np.array([0, 10]) * 1
    Qslack = np.array([0, 50]) * 1
    QterminalSlack = np.diag([10, 1, 1, 1, 10, 1]) * 20
    dR_LMPC = np.array([1.0, 10.0]) * 10
    xRef = np.array([vt, 0, 0, 0, 0, 0])

    LMPC_Solver = "CVX"   # Can pick CVX for cvxopt or OSQP. For OSQP uncomment line 14 in LMPC.py
    numSS_it = 4          # Number of trajectories used at each iteration to build the safe set
    numSS_Points = 40     # Number of points to select from each trajectory to build the safe set

    Laps = 46 + numSS_it  # Total LMPC laps (50 laps)

    map = Map(0.4)                                            # Initialize the map
    model = Carmodel(map)                                     # Initialize the MPC model
    LMPCmodel = Carmodel(map, 1, 1)                           # Initialize the LMPC model

    # State constraints
    Fx = np.array([[0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0., -1.]])
    bx = np.array([[2.],
                   [2.]])

    # Input constraints
    Fu = np.kron(np.eye(2), np.array([1, -1])).T
    bu = np.array([[0.5], [0.5],
                   [10.0], [10.0]])

    ClosedLoopDataPID = ClosedLoopData(dt, Time, v0)
    PIDController = PID(vt)
    model.Sim(ClosedLoopDataPID, PIDController)

    print("Starting PID")
    file_data = open(sys.path[0]+'\data\ClosedLoopDataPID.obj', 'wb')
    pickle.dump(ClosedLoopDataPID, file_data)
    file_data.close()
    print("===== PID terminated")

    print("Starting LTI-MPC")
    lamb = 0.0000001
    A, B, Error = Regression(ClosedLoopDataPID.x, ClosedLoopDataPID.u, lamb)
    ClosedLoopDataLTI_MPC = ClosedLoopData(dt, TimeMPC, v0)
    timevarying = False
    slacks = True
    dR = np.array([1.0, 10.0]) * 10
    LTI_MPC = OCTP(dim_state, dim_input, N, Q, R, Qm, Fx, bx, Fu, bu, Qslack, slacks, A, B, xRef, timevarying, dR)
    model.Sim(ClosedLoopDataLTI_MPC, LTI_MPC)

    file_data = open(sys.path[0] + '\data\ClosedLoopDataLTI_MPC.obj', 'wb')
    pickle.dump(ClosedLoopDataLTI_MPC, file_data)
    file_data.close()
    print("===== LTI-MPC terminated")

    print("Starting LTV-MPC")
    ClosedLoopDataLTV_MPC = ClosedLoopData(dt, TimeMPC_tv, v0)
    timevarying = True
    LTV_MPC = OCTP(dim_state, dim_input, N, Q, R, Qm, Fx, bx, Fu, bu, Qslack, slacks, A, B, xRef, timevarying, dR)
    model.Sim(ClosedLoopDataLTV_MPC, LTV_MPC)

    file_data = open(sys.path[0] + 'data\ClosedLoopDataLTV_MPC.obj', 'wb')
    pickle.dump(ClosedLoopDataLTV_MPC, file_data)
    file_data.close()
    print("===== LTV-MPC terminated")

    print("Starting LMPC")
    ClosedLoopLMPC =  ClosedLoopData(dt, TimeLMPC, v0)
    LMPCOpenLoopData = PredictiveModel(N, dim_state, dim_input, TimeLMPC, numSS_Points, Laps)

    LMPC = LMPCplus(dim_state, dim_input, N, Q_lmpc, R_lmpc, Qf, Fx, bx, Fu, bu, dt, dR_LMPC, numSS_Points, numSS_it, QterminalSlack, map, Laps, TimeLMPC, LMPC_Solver)
    LMPC.addTrajectory(ClosedLoopDataPID)
    LMPC.addTrajectory(ClosedLoopDataLTV_MPC)
    LMPC.addTrajectory(ClosedLoopDataPID)
    LMPC.addTrajectory(ClosedLoopDataLTI_MPC)

    x0 = np.zeros((1, dim_state))
    x0_glob = np.zeros((1, dim_state))
    x0[0, :] = ClosedLoopLMPC.x[0, :]
    x0_glob[0, :] = ClosedLoopLMPC.x_glob[0, :]

    for it in range(numSS_it, Laps):
        ClosedLoopLMPC.updateInitialConditions(x0, x0_glob)
        LMPCmodel.Sim(ClosedLoopLMPC, LMPC, LMPCOpenLoopData)
        LMPC.addTrajectory(ClosedLoopLMPC)

        if LMPC.feasible == 0:
            break
        else:
            # Reset Initial Conditions
            x0[0, :] = ClosedLoopLMPC.x[ClosedLoopLMPC.SimTime, :] - np.array([0, 0, 0, 0, map.TrackLength, 0])
            x0_glob[0, :] = ClosedLoopLMPC.x_glob[ClosedLoopLMPC.SimTime, :]

    file_data = open(sys.path[0] + '\data\LMPController.obj', 'wb')
    pickle.dump(ClosedLoopLMPC, file_data)
    pickle.dump(LMPC, file_data)
    pickle.dump(LMPCOpenLoopData, file_data)
    file_data.close()
    print("===== LMPC terminated")

    laptimes = np.zeros((50, 2))

    # Laptime Plot
    for i in range(0, LMPC.it):
        print("Lap time at iteration ", i, " is ", LMPC.Qfun[0, i] * dt, "s")
        laptimes[i, 0] = LMPC.Qfun[0, i] * dt
        laptimes[i, 1] = i
    plt.figure(3)
    plt.plot(laptimes[:, 1], laptimes[:, 0], '-o')
    plt.ylabel('Lap Time (sec)')
    plt.xlabel('Lap Number')

    print("===== Start Plotting")

    plotTrajectory(map, ClosedLoopDataPID.x, ClosedLoopDataPID.x_glob, ClosedLoopDataPID.u)

    plotTrajectory(map, ClosedLoopDataLTI_MPC.x, ClosedLoopDataLTI_MPC.x_glob, ClosedLoopDataLTI_MPC.u)

    plotTrajectory(map, ClosedLoopDataLTV_MPC.x, ClosedLoopDataLTV_MPC.x_glob, ClosedLoopDataLTV_MPC.u)

    plotClosedLoopLMPC(LMPC, map)

    animation_xy(map, LMPCOpenLoopData, LMPC, Laps - 2)

    animation_states(map, LMPCOpenLoopData, LMPC, 10)

    unityTestChangeOfCoordinates(map, ClosedLoopDataPID)
    unityTestChangeOfCoordinates(map, ClosedLoopDataLTI_MPC)
    unityTestChangeOfCoordinates(map, ClosedLoopLMPC)

    saveGif_xyResults(map, LMPCOpenLoopData, LMPC, Laps-1)
    Save_statesAnimation(map, LMPCOpenLoopData, LMPC, 5)

    plt.show()





    # roadHalfWidth = 2.0
    # outfile = TemporaryFile()
    # ftocp = FTOCP(N=N, roadHalfWidth=roadHalfWidth)  # ftocp used by LMPC
    # itMax = 10  # max number of itertions
    # itCounter = 1  # iteration counter
    # x0 = [0, 0, 0]  # initial condition
    #
    # # Compute feasible trajectory
    # xclFeasible, uclFeasible = feasTraj(ftocp, 101, x0)
    # print(np.round(xclFeasible, decimals=2))
    # np.savetxt('storedData/closedLoopFeasible.txt', xclFeasible.T, fmt='%f')
    # np.savetxt('storedData/inputFeasible.txt', uclFeasible.T, fmt='%f')
    #
    # # Initialize LMPC object
    # # lmpc = LMPC(ftocp, l=10, P = 200, verbose = False)
    # # lmpc = LMPC(ftocp, l=4, P = 50, verbose = False)
    # # lmpc = LMPC(ftocp, l=3, P = 20, verbose = False)
    # lmpc = LMPC(ftocp, l=1, P=15, verbose=False)
    #
    # # Add feasible trajectory to the safe set
    # lmpc.addTrajectory(xclFeasible, uclFeasible)
    #
    # # Pick terminal state or terminal set
    # terminalSet = True
    # if terminalSet == False:
    #     lmpc.ftocp.set_xf(np.array([xclFeasible[:, -1]]).T)
    # else:
    #     sFinishLine = xclFeasible[0, -1]
    #     delta_s = 0.5  # 2.0
    #     Xf = np.array([[sFinishLine, -roadHalfWidth, 0.0], [sFinishLine + delta_s, roadHalfWidth, 0.0]]).T
    #     print("Box Xf")
    #     print(Xf)
    #     lmpc.ftocp.set_xf(Xf)
    #
    #     Xf_vertices = np.concatenate((Xf, Xf), axis=1)
    #     Xf_vertices[1, 2] = roadHalfWidth
    #     Xf_vertices[1, 3] = -roadHalfWidth
    #     lmpc.Xf_vertices = Xf_vertices
    #     print("Verices Xf")
    #     print(Xf_vertices)
    #
    # # Iteration loop
    # meanTimeCostLMPC = []
    # while itCounter <= itMax:
    #     time = 0
    #     itFlag = 0
    #     xcl = [x0]
    #     ucl = []
    #     timeLMPC = []
    #     # Time loop
    #     while (itFlag == 0):
    #         xt = xcl[time]  # read measurement
    #
    #         startTimer = datetime.datetime.now()
    #         lmpc.solve(xt, verbose=1)  # solve LMPC
    #         deltaTimer = datetime.datetime.now() - startTimer
    #         timeLMPC.append(deltaTimer.total_seconds())
    #
    #         # Apply input and store closed-loop data
    #         ut = lmpc.ut
    #         ucl.append(copy.copy(ut))
    #         xcl.append(copy.copy(ftocp.f(xcl[time], ut)))
    #
    #         # Print results
    #         if lmpc.verbose == True:
    #             print("State trajectory at time ", time)
    #             print(np.round(np.array(xcl).T, decimals=2))
    #             print(np.round(np.array(ucl).T, decimals=2))
    #             print("===============================================")
    #
    #         # Check if goal state has been reached
    #         if lmpc.ftocp.checkTaskCompletion(xcl[-1]):
    #             # if np.linalg.norm([xcl[-1]-xclFeasible[:,-1]]) <= 1e-4:
    #             if lmpc.verbose == True:
    #                 print("Distance from terminal point:", xcl[-1] - xclFeasible[:, -1])
    #             break
    #
    #         # increment time counter
    #         time += 1
    #
    #     # iteration completed. Add trajectory to the safe set
    #     xcl = np.array(xcl).T
    #     ucl = np.array(ucl).T
    #     lmpc.addTrajectory(xcl, ucl)
    #
    #     # Store time and cost
    #     meanTimeCostLMPC.append(np.array([np.sum(timeLMPC) / lmpc.cost, lmpc.cost]))
    #
    #     # Print and store results
    #     print("++++++===============================================++++++")
    #     print("Completed Iteration: ", itCounter)
    #     print("++++++===============================================++++++")
    #     np.savetxt('storedData/closedLoopIteration' + str(itCounter) + '_P_' + str(lmpc.P) + '.txt',
    #                np.round(np.array(xcl), decimals=5).T, fmt='%f')
    #     np.savetxt('storedData/inputIteration' + str(itCounter) + '_P_' + str(lmpc.P) + '.txt',
    #                np.round(np.array(ucl), decimals=5).T, fmt='%f')
    #     np.savetxt('storedData/meanTimeLMPC_P_' + str(lmpc.P) + '.txt', np.array(meanTimeCostLMPC), fmt='%f')
    #     np.save(outfile, xcl)
    #
    #     itCounter += 1  # increment iteration counter


# def feasTraj(ftocp, timeSteps, x0):
#     # Compute first feasible trajectory
#
#     # Intial condition
#     xcl = [np.array(x0)]
#     ucl = []
#     u = [0, 0]
#     radius = ftocp.radius
#
#     # Simple brute force hard coded if logic
#     for i in range(0, timeSteps):
#         if i == 0:
#             u[1] = 0.25
#         elif i == 1:
#             u[1] = 0.25
#         elif i == 2:
#             u[1] = 0.25
#         elif i == (timeSteps - 4):
#             u[1] = -0.25
#         elif i == (timeSteps - 3):
#             u[1] = -0.25
#         elif i == (timeSteps - 2):
#             u[1] = -0.25
#         else:
#             u[1] = 0
#
#         u[0] = xcl[-1][0] / radius;
#
#         xcl.append(ftocp.f(xcl[-1], u))
#         ucl.append(np.array(u))
#
#     return np.array(xcl).T[:, :-1], np.array(ucl).T[:, :-1]


if __name__ == "__main__":
    main()