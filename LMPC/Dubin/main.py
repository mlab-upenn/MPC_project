import numpy as np
from FTOCP import FTOCP
from LMPC import LMPC
import pdb
import dill
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import copy
import datetime
import os

def main():
	
	if not os.path.exists('storedData'):
		os.makedirs('storedData')

	# parameter initialization
	N = 4
	roadHalfWidth = 2.0
	outfile = TemporaryFile()
	ftocp = FTOCP(N=N, roadHalfWidth=roadHalfWidth) 
	itMax = 10 
	itCounter = 1
	x0 = [0, 0, 0] 


	# Compute feasible trajectory
	xclFeasible, uclFeasible = feasTraj(ftocp, 101, x0)
	print(np.round(xclFeasible, decimals=2))
	np.savetxt('storedData/closedLoopFeasible.txt',xclFeasible.T, fmt='%f' )
	np.savetxt('storedData/inputFeasible.txt',uclFeasible.T, fmt='%f' )

	
	# lmpc = LMPC(ftocp, l=10, P = 200, verbose = False)
	# lmpc = LMPC(ftocp, l=4, P = 50, verbose = False)
	# lmpc = LMPC(ftocp, l=3, P = 20, verbose = False)
	lmpc = LMPC(ftocp, l=1, P = 15, verbose = False)


	lmpc.addTrajectory(xclFeasible, uclFeasible)

	
	terminalSet = True
	if terminalSet == False:
		lmpc.ftocp.set_xf(np.array([xclFeasible[:,-1]]).T)
	else:	
		sFinishLine = xclFeasible[0,-1]
		delta_s = 0.5 #2.0
		Xf = np.array([[sFinishLine, -roadHalfWidth, 0.0],[sFinishLine+delta_s, roadHalfWidth, 0.0]]).T
		print("Box Xf")
		print(Xf)
		lmpc.ftocp.set_xf(Xf)

		Xf_vertices = np.concatenate((Xf,Xf), axis = 1 )
		Xf_vertices[1,2] = roadHalfWidth
		Xf_vertices[1,3] = -roadHalfWidth
		lmpc.Xf_vertices = Xf_vertices
		print("Verices Xf")
		print(Xf_vertices)
	
	
	meanTimeCostLMPC = []
	while itCounter <= itMax:
		time = 0
		itFlag = 0
		xcl = [x0]
		ucl = []
		timeLMPC = []
	
		while (itFlag == 0):
			xt = xcl[time] 

			startTimer = datetime.datetime.now()
			lmpc.solve(xt, verbose = 1) 
			deltaTimer = datetime.datetime.now() - startTimer
			timeLMPC.append(deltaTimer.total_seconds())

		
			ut = lmpc.ut
			ucl.append(copy.copy(ut))
			xcl.append(copy.copy(ftocp.f(xcl[time], ut)))

			# Print results
			if lmpc.verbose == True:
				print("State trajectory at time ", time)
				print(np.round(np.array(xcl).T, decimals=2))
				print(np.round(np.array(ucl).T, decimals=2))
				print("===============================================")

		
			if lmpc.ftocp.checkTaskCompletion(xcl[-1]):
			# if np.linalg.norm([xcl[-1]-xclFeasible[:,-1]]) <= 1e-4:
				if lmpc.verbose == True:
					print("Distance from terminal point:", xcl[-1]-xclFeasible[:,-1])
				break

			# increment time counter
			time += 1

	
		xcl = np.array(xcl).T
		ucl = np.array(ucl).T
		lmpc.addTrajectory(xcl, ucl) 

		
		meanTimeCostLMPC.append(np.array([np.sum(timeLMPC)/lmpc.cost, lmpc.cost]))

	
		print("++++++===============================================++++++")
		print("Completed Iteration: ", itCounter)
		print("++++++===============================================++++++")
		np.savetxt('storedData/closedLoopIteration'+str(itCounter)+'_P_'+str(lmpc.P)+'.txt', np.round(np.array(xcl), decimals=5).T, fmt='%f' )
		np.savetxt('storedData/inputIteration'+str(itCounter)+'_P_'+str(lmpc.P)+'.txt', np.round(np.array(ucl), decimals=5).T, fmt='%f' )
		np.savetxt('storedData/meanTimeLMPC_P_'+str(lmpc.P)+'.txt', np.array(meanTimeCostLMPC), fmt='%f' )
		np.save(outfile, xcl)

		itCounter += 1
	


def feasTraj(ftocp, timeSteps,x0):
	
	xcl = [np.array(x0)]
	ucl =[]
	u = [0,0]
	radius = ftocp.radius

	for i in range(0, timeSteps):
		if i ==0:
			u[1] =  0.25
		elif i== 1:
			u[1] =  0.25
		elif i== 2:
			u[1] =  0.25
		elif i==(timeSteps-4):
			u[1] = -0.25
		elif i==(timeSteps-3):
			u[1] =  -0.25
		elif i==(timeSteps-2):
			u[1] = -0.25
		else:
			u[1] = 0   
		
		u[0] = xcl[-1][0] / radius

		xcl.append(ftocp.f(xcl[-1], u))
		ucl.append(np.array(u))


	return np.array(xcl).T[:,:-1], np.array(ucl).T[:,:-1]



if __name__== "__main__":
	main()
