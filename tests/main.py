import os, sys
import subprocess as sp
import unittest
import numpy as np
import numpy.testing as npt
from matplotlib.pyplot import *
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

class TestHeat(unittest.TestCase):
    def setUp(self):
        from heatsolver import HeatSolver
        import solver as sl
        Tinf = 50.0
        n = 21
        y = np.linspace(0.0, 1.0, n)
        q = np.ones_like(y)*Tinf
        beta = np.ones_like(y)
        solver = HeatSolver(y, beta, q, verbose=True, mode=sl.dns, Tinf = Tinf)
        solver.solve(dt=1e3)
        udns = solver.get_solution()
        
        solver = HeatSolver(y, beta, q, verbose=True, Tinf = Tinf)
        solver.solve(dt=1e3)
        u = solver.get_solution()
        utarget = udns.copy()
        
        solver.set_target(utarget, beta)
        solver.solve()
        solver.set_sigmas(1e-1, 0.8)
        uprior = solver.get_solution()
        self.solver = solver
        self.beta = beta
        self.n = n
        self.y = y
        
    def test_gradient(self):
        J = self.solver.get_objective()
        dJdbeta = self.solver.get_sensitivity()
        dJdbeta_fd = np.zeros_like(self.beta)
        dbeta = 1e-7
        for i in range(self.n):
            print i
            self.beta[i] += dbeta
            self.solver.reset()
            self.solver.set_beta(self.beta)
            self.solver.solve()
            dJdbeta_fd[i] = (self.solver.get_objective() - J)/dbeta
            self.beta[i] -= dbeta
        l2norm = np.linalg.norm(dJdbeta_fd - dJdbeta)/np.linalg.norm(dJdbeta)
        figure()
        plot(self.y, dJdbeta, 'r-', label="Adjoint")
        plot(self.y, dJdbeta_fd, 'b.', label="FD")
        legend(loc="best")
        xlabel("y")
        ylabel("dJdbeta")
        savefig("figures/test_laminar_grad.pdf")
        npt.assert_almost_equal(l2norm, 0, 0)
        npt.assert_allclose(dJdbeta_fd[1:], dJdbeta[1:], rtol=1e-2, atol=1e13)
        show()
if __name__ == "__main__":
    unittest.main()
