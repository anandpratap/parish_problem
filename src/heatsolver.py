import numpy as np
import solver as sl
import time

class HeatSolver(object):
    def __init__(self, y, beta=1.0, q=None, filename=None, verbose=False, solver=sl.HeatSolver, mode=sl.rans, Tinf = 50.0):
        self.y = y
        self.n = len(q)
        self.mode = mode
        self.Tinf = Tinf
        if hasattr(beta, "__len__"):
            self.beta = beta.copy()
        else:
            self.beta = np.ones(self.n)*beta
            
        self.verbose = verbose
        if q is not None:
            assert self.n == len(q), "solution dimension not correct!"
            self.q = q.copy()
        else:
            if filename is not None:
                self.q = self.initialize_from_file(filename)
            else:
                self.q = self.initialize_from_data()


        self.solver = solver(mode, Tinf, verbose)
        self.solver.initialize(self.y, self.q, self.beta)

    def solve(self, dt=1e3, iteration_max=100, rtol_cutoff=1e-10, atol_cutoff=1e-10, iteration_ramp=100):
        self.t1 = time.time()
        error = self.solver.solve(dt, iteration_max, rtol_cutoff, atol_cutoff, iteration_ramp)
        self.t2 = time.time()
        if self.verbose:
            print "Time taken for solve: %0.4f seconds."%(self.t2 - self.t1)
        return error

    def reset(self):
        self.solver.reset()

    def get_solution(self):
        return self.solver.get_solution(self.n)
    
    def get_objective(self):
        return self.solver.calc_objective()
    
    def get_sensitivity(self):
        return self.solver.get_sensitivity(self.n)
    
    def set_beta(self, beta):
        return self.solver.set_beta(beta)
    
    def set_target(self, qtarget, beta_prior):
        return self.solver.set_target(qtarget, beta_prior)
    
    def set_sigmas(self, sigma_obs, sigma_prior):
        return self.solver.set_sigmas(sigma_obs, sigma_prior)

    def get_true_beta(self, q):
        h = 0.5
        eps = 5e-4
        beta = 1/eps*(1.0 + 5.0*np.sin(3.0*np.pi/200.0*q) + np.exp(0.02*q))*1e-4 + h/eps*(self.Tinf - q)/(self.Tinf**4 - q**4)
        return beta
