####################################################################
####################################################################
#Refs.
# 1 : A robust method to estimate the maximal Lyapunov exponent of a time series - https://doi.org/10.1016/0375-9601(94)90991-1
# 2 : Nonlinear time series analysis - H. Kantz & T. Schreiber- Pg. 70
####################################################################
####################################################################

import time
import  concurrent.futures 
import numpy as np
import functools
start = time.perf_counter()


def henon(xy, a = 1.4, b = 0.3):
	x, y = xy
	xit = 1 - a * x**2 + y
	yit = b * x
	return xit, yit


def makelist(lxy,tau):
	eps = 0.008
	xyt = lxy[:,0]
	lista = [[] for i in range(len(xyt))]
	for K in range(len(xyt)-tau):
		n = 0
		soma = 0
		for i in range(len(xyt)-tau):
			if(K!=i):
				soma = np.linalg.norm(lxy[i,:] - lxy[K,:])
				if((soma) <= eps):
					lista[K].append(np.linalg.norm(lxy[i + tau,:] - lxy[K + tau,:]))
	lista = [ll for ll in lista if ll != []] #delete null neighborhoods
#	lista = [ll for ll in lista if len(ll) <= 5] #select min/max size of neighborhoods
	return lista



trs= 6000
t = int(trs/2)

at = t 
xp = np.zeros((trs+1 , 2))
xp[0] = 0.353*np.ones(2)

for i in range(trs):
	xp[i+1] = henon(xp[i])
	

xyn = np.zeros((at+1,2))
xyn[0] = xp[-1]


for i in range(at):
	xyn[i+1] = henon(xyn[i])


#####Delay reconstruction ###If tau from reconstruction if != then the tau from the neighborhoods measures, uncomment this block
#tau = 1
#xy_tau = np.zeros((len(xyn)-tau,2))
#xy_tau[:,0] = xyn[tau:,0]
#xy_tau[:,1] = xyn[:len(xy_tau),0]
#####Delay reconstruction

def lyaS(entries):
	S = 0
	tau = entries[0]
#### Uncomment if the tau's are the same
#	xy_tau = np.zeros((len(xyn)-tau,2))
#	xy_tau[:,0] = xyn[tau:,0]
#	xy_tau[:,1] = xyn[:len(xy_tau),0]


	diffs = makelist(xy_tau,tau)[:500]# taking of 500 neighborhoods, you can take into account all of them if you want
	l_diff = []


	for i in range(len(diffs)):
		soma = 0
		Us = 0
		l_diff = diffs[i]
		for k in l_diff:
			soma += (k)
			Us += 1
		S += np.log(soma/Us)


	S /= len(diffs)
	return S



rss = []
with concurrent.futures.ProcessPoolExecutor() as executor:
	yns = [[1],[10]]#here you put the list of taus that you want to calculate the Lyapunov exponent. S(tau)
	results = executor.map(lyaS, yns)
	for result in results:
		rss.append(result)

np.savetxt("lyapunov_kant.dat",np.array(rss))

finish = time.perf_counter()

print(f'Finish in {round(finish-start,2)} second(s)')



