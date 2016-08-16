from pylab import *
from heatsolver import HeatSolver
import solver as sl

Tinf = 100.0
n = 101
y = linspace(0.0, 1.0, n)
q = np.ones_like(y)*Tinf
beta = np.ones_like(y)

for i in range(100):
    solver = HeatSolver(y, beta, q, verbose=True, mode=sl.dns, Tinf = Tinf)
    solver.solve(dt=1e3)
    u = solver.get_solution()
    plot(y, u, "b--")


show()
