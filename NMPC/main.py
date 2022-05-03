import numpy as np
from utils import systemdy
import pdb
import matplotlib.pyplot as plt
from ndyn_ftocp import NFTOCPNLP


## Parameters initialization
N = 30  # 20
n_dy = 6
d = 2
x0_dy = np.zeros(6)
dt = 0.1
sys_dy = systemdy(x0_dy, dt)
maxTime = 30
xRef = np.array([10, 10, 0, np.pi/2])
xRef_dy = np.array([10, 10, np.pi/2, 0, 0, 0])

R = 1*np.eye(d)
Q_dy = 1*np.eye(n_dy)
# Qf_dy = 1000*np.eye(n_dy)
Qf_dy = np.diag([11.8, 2.0, 50.0, 280.0, 100.0, 1000.0])

bx = np.array([15, 15, 15, 15])
bx_dy = np.array([15, 15, 15, 15, 15, 15])
bu = np.array([10, 0.5])

## Solving the problem
nlp_dy = NFTOCPNLP(N, Q_dy, R, Qf_dy, xRef_dy, dt, bx_dy, bu)
ut_dy = nlp_dy.solve(x0_dy)

sys_dy.reset_IC()
xPredNLP_dy = []
uPredNLP_dy = []
CostSolved_dy = []
for t in range(0, maxTime):
	xt_dy = sys_dy.x[-1]
	ut_dy = nlp_dy.solve(xt_dy)
	xPredNLP_dy.append(nlp_dy.xPred)
	uPredNLP_dy.append(nlp_dy.uPred)
	CostSolved_dy.append(nlp_dy.qcost)
	sys_dy.applyInput(ut_dy)

x_cl_nlp_dy = np.array(sys_dy.x)
u_cl_nlp_dy = np.array(sys_dy.u)
# print("Close-loop output:")
# print(x_cl_nlp_dy)

cost_a = 0
cost_act = []
for i in range(0, N):
	cost_a = (x_cl_nlp_dy[i] - xRef_dy).T @ Q_dy @ (x_cl_nlp_dy[i] - xRef_dy)
	cost_a += u_cl_nlp_dy[i].T @ R @ u_cl_nlp_dy[i]
	cost_a += (x_cl_nlp_dy[i] - xRef_dy).T @ Qf_dy @ (x_cl_nlp_dy[i] - xRef_dy) + u_cl_nlp_dy[i].T @ R @ u_cl_nlp_dy[i]
	# print("Actual cost:", cost_a)
	cost_act.append(cost_a)

print("Actual cost:", sum(cost_act))


lost_x = 0
lost_xct = []
lost_y = 0
lost_yct = []
for i in range(0, N):
	lost_x = (x_cl_nlp_dy[i, 0] - xPredNLP_dy[0][i, 0])**2
	lost_xct.append(lost_x)
	lost_y = (x_cl_nlp_dy[i, 1] - xPredNLP_dy[0][i, 1]) ** 2
	lost_yct.append(lost_y)

SolveTime_dy = sum(nlp_dy.solverTime) / len(nlp_dy.solverTime)
print("Solving time for dynamic model:", SolveTime_dy)


for timeToPlot in [0, 10]:
	plt.figure()
	plt.plot(xPredNLP_dy[timeToPlot][:,0], xPredNLP_dy[timeToPlot][:,1], '--.b', label="Simulated trajectory using NLP-aided MPC at time $t = $"+str(timeToPlot))
	plt.plot(xPredNLP_dy[timeToPlot][0,0], xPredNLP_dy[timeToPlot][0,1], 'ok', label="$x_t$ at time $t = $"+str(timeToPlot))
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.xlim(-1,15)
	plt.ylim(-1,15)
	plt.legend()
	plt.show()

plt.figure()
for t in range(0, maxTime):
	if t == 0:
		plt.plot(xPredNLP_dy[t][:,0], xPredNLP_dy[t][:,1], '--.b', label='Simulated trajectory using NLP-aided MPC')
	else:
		plt.plot(xPredNLP_dy[t][:,0], xPredNLP_dy[t][:,1], '--.b')
plt.plot(x_cl_nlp_dy[:,0], x_cl_nlp_dy[:,1], '-*r', label="Closed-loop trajectory")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim(-1,15)
plt.ylim(-1,15)
plt.legend()
plt.show()

plt.figure()
plt.plot(xPredNLP_dy[0][:,0], xPredNLP_dy[0][:,1], '--.b', label='Simulated trajectory using NLP-aided MPC')
plt.plot(x_cl_nlp_dy[:,0], x_cl_nlp_dy[:,1], '-*r', label="Closed-loop trajectory")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim(-1,15)
plt.ylim(-1,15)
plt.legend()
plt.show()

plt.figure()
plt.plot(u_cl_nlp_dy[:,0], '-*r', label="Closed-loop input: Acceleration")
plt.plot(uPredNLP_dy[0][:,0], '-ob', label="Simulated input: Acceleration")
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend()
plt.show()

plt.figure()
plt.plot(u_cl_nlp_dy[:,1], '-*r', label="Closed-loop input: Steering")
plt.plot(uPredNLP_dy[0][:,1], '-ob', label="Simulated input: Steering")
plt.xlabel('Time')
plt.ylabel('Steering')
plt.legend()
plt.show()

plt.figure()
plt.plot(xPredNLP_dy[0][:,0], xPredNLP_dy[0][:,1], '-*r', label='Solution from the NLP')
plt.title('Simulated trajectory')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-1,15)
plt.ylim(-1,15)
plt.legend()
plt.show()

plt.figure()
plt.plot(xPredNLP_dy[0][:,3], '-*r', label='NLP performance')
plt.plot(x_cl_nlp_dy[:,3], 'ok', label='Closed-loop performance')
plt.xlabel('Time')
plt.ylabel('Velocity of the x-axis')
plt.legend()
plt.show()


plt.figure()
plt.plot(CostSolved_dy, '-ob')
plt.xlabel('Time')
plt.ylabel('Iteration cost')
plt.legend()
plt.show()

plt.figure()
plt.plot(lost_xct, 'ob', label='X-Position')
plt.plot(lost_yct, '*r', label='Y-Position')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.show()


for t in range(0, maxTime):
	plt.figure()
	plt.plot(xPredNLP_dy[t][:,0], xPredNLP_dy[t][:,1], '-*r', label='Simulated trajectory using NLP-aided MPC at time $t = $'+str(t))
	plt.plot(xPredNLP_dy[t][0,0], xPredNLP_dy[t][0,1], 'ok', label="$x_t$ at time $t = $"+str(t))
	plt.xlabel('$x_1$')
	plt.ylabel('$x_2$')
	plt.xlim(-1,15)
	plt.ylim(-1,15)
	plt.legend()
	plt.show()
