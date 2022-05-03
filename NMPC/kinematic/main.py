import numpy as np
from utils import system
import pdb
import matplotlib.pyplot as plt
from dyn_ftocp import FTOCP
from ftocp_nlp import FTOCPNLP


## Parameters initialization
N = 60  # 20
n = 4
d = 2
x0 = np.zeros(4)
dt = 0.1
sys = system(x0, dt)
maxTime = 60
xRef = np.array([10, 10, 0, np.pi/2])

Q = 1*np.eye(n)
R = 1*np.eye(d)
Qf = 1000*np.eye(n)

bx = np.array([15, 15, 15, 15])
bu = np.array([10, 0.5])

## Solving the problem
nlp = FTOCPNLP(N, Q, R, Qf, xRef, dt, bx, bu)
ut = nlp.solve(x0)

sys.reset_IC()
xPredNLP = []
uPredNLP = []
CostSolved = []
for t in range(0, maxTime):
	xt = sys.x[-1]
	# print("xt:")
	# print(xt)
	ut = nlp.solve(xt)
	xPredNLP.append(nlp.xPred)
	uPredNLP.append(nlp.uPred)
	CostSolved.append(nlp.qcost)
	sys.applyInput(ut)

x_cl_nlp = np.array(sys.x)
u_cl_nlp = np.array(sys.u)
# print("close-loop state:")
# print(x_cl_nlp)
# print("close-loop input:")
# print(u_cl_nlp)

SolveTime = sum(nlp.solverTime) / len(nlp.solverTime)
print("Solving time:", SolveTime)

for timeToPlot in [0, 30]:
	plt.figure()
	plt.plot(xPredNLP[timeToPlot][:,0], xPredNLP[timeToPlot][:,1], '--.b', label="Predicted trajectory at time $t = $"+str(timeToPlot))
	plt.plot(xPredNLP[timeToPlot][0,0], xPredNLP[timeToPlot][0,1], 'ok', label="$x_t$ at time $t = $"+str(timeToPlot))
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.xlim(-1,12)
	plt.ylim(-1,15)
	plt.legend()
	plt.show()

plt.figure()
for t in range(0, maxTime):
	if t == 0:
		plt.plot(xPredNLP[t][:,0], xPredNLP[t][:,1], '--.b', label='Predicted trajectory using NLP-aided MPC')
	else:
		plt.plot(xPredNLP[t][:,0], xPredNLP[t][:,1], '--.b')
plt.plot(x_cl_nlp[:,0], x_cl_nlp[:,1], '-*r', label="Closed-loop trajectory")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim(-1,12)
plt.ylim(-1,15)
plt.legend()
plt.show()


plt.figure()
plt.plot(u_cl_nlp[:,0], '-*r', label="Closed-loop input: Acceleration")
plt.plot(uPredNLP[0][:,0], '-ob', label="Predicted input: Acceleration")
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.legend()
plt.show()

plt.figure()
plt.plot(u_cl_nlp[:,1], '-*r', label="Closed-loop input: Steering")
plt.plot(uPredNLP[0][:,1], '-ob', label="Predicted input: Steering")
plt.xlabel('Time')
plt.ylabel('Steering')
plt.legend()
plt.show()

plt.figure()
plt.plot(xPredNLP[0][:,0], xPredNLP[0][:,1], '-*r', label='Solution from the NLP')
plt.title('Predicted trajectory')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-1,12)
plt.ylim(-1,15)
plt.legend()
plt.show()

plt.figure()
plt.plot(xPredNLP[0][:,3]* 180 / np.pi, '-*r', label='NLP performance')
plt.plot(x_cl_nlp[:,3]* 180 / np.pi, 'ok', label='Closed-loop performance')
plt.xlabel('Time')
plt.ylabel('Heading angle (degree)')
plt.xlim(-1,62)
plt.ylim(-1,100)
plt.legend()
plt.show()

plt.figure()
plt.plot(CostSolved, '-ob')
plt.xlabel('Time')
plt.ylabel('Cost')
plt.legend()
plt.show()

for t in range(0, 6):
	plt.figure()
	plt.plot(xPredNLP[t][:,0], xPredNLP[t][:,1], '-*r', label='Predicted trajectory using NLP at time $t = $'+str(t))
	plt.plot(xPredNLP[t][0,0], xPredNLP[t][0,1], 'ok', label="$x_t$ at time $t = $"+str(t))
	plt.xlabel('$x_1$')
	plt.ylabel('$x_2$')
	plt.xlim(-1,12)
	plt.ylim(-1,15)
	plt.legend()
	plt.show()