import numpy as np
import matplotlib.pyplot as plt
from heatsolver import HeatSolver
import solver as sl
class InverseSolver(object):
    def __init__(self, solver, beta, solver_options={}):
        self.solver = solver
        self.beta = beta.copy()
        self.n = len(beta)
        self.dostats = False
        self.solver_options = solver_options
        
    def sample_prior(self):
        pass
    
    def sample_posterior(self):
        pass

    def calc_stats(self):
        pass

    def get_sensitivity(self):
        pass

    def get_stepsize(self):
        return 1.0

    def step_sd(self):
        if self.i == 0:
            self.solver.reset()
            self.solver.set_beta(self.beta)
            self.solver.solve(**self.solver_options)
            self.neval += 1
        dJdbeta = self.solver.get_sensitivity()
        dJdbeta_norm = dJdbeta/np.linalg.norm(dJdbeta)
        pk = -dJdbeta_norm
        stepsize = self.linesearch(self.stepsize, pk)
        self.beta += stepsize*pk
        self.dJdbeta_norm = dJdbeta_norm
        self.dJdbeta_l2norm = np.linalg.norm(dJdbeta)
        
    def step_bfgs(self):
        if self.i == 0:
            self.solver.reset()
            self.solver.set_beta(self.beta)
            self.solver.solve(**self.solver_options)
            self.neval += 1
        dJdbeta = self.solver.get_sensitivity()
        dJdbeta_norm= dJdbeta/np.linalg.norm(dJdbeta)
        if self.i == 0:
            self.B = np.eye(np.size(self.beta))
        else:
            yk = (dJdbeta_norm - self.dJdbeta_norm)[np.newaxis].T
            sk = self.sk[np.newaxis].T
            term_1_num = yk.dot(yk.transpose())
            term_1_den = yk.transpose().dot(sk)
            term_2_num = self.B.dot(sk.dot(sk.transpose().dot(self.B)))
            term_2_den = sk.transpose().dot(self.B.dot(sk))
            self.B = self.B + term_1_num/term_1_den - term_2_num/term_2_den

        pk = np.linalg.solve(self.B, -dJdbeta_norm)
        pk = pk/np.linalg.norm(pk)
        stepsize = self.linesearch(self.stepsize, pk)
        sk = stepsize*pk
        self.beta += sk
        self.sk = sk
        self.dJdbeta_norm = dJdbeta_norm
        self.dJdbeta_l2norm = np.linalg.norm(dJdbeta)

    def linesearch(self, stepsize, pk):
        beta_ = self.beta.copy()
        J_ = self.solver.get_objective()
        for i in range(10):
            self.beta = beta_ + pk*stepsize
            self.solver.reset()
            self.solver.set_beta(self.beta)
            self.solver.solve(**self.solver_options)
            self.neval += 1
            J = self.solver.get_objective()
            if J < J_:
                self.beta[:] = beta_[:]
                break
            else:
                stepsize /= 2.0
        self.beta[:] = beta_[:]
        self.stepsize = 0.7*self.stepsize + 0.3*stepsize
        return stepsize


    def calculate_hessian(self):
        pass

    def calculate_cholesky(self):
        H = self.calculate_hessian()
        Cov = np.linalg.inv(H)
        R = np.chol(Cov)
        return R
        
    def solve(self, maxiter=100, stepsize=0.01, algo='sd'):
        self.neval = 0
        self.algo = algo
        self.maxiter = maxiter
        self.stepsize = stepsize
        if self.dostats:
            self.sample_prior()

        for i in range(self.maxiter):
            self.i = i
            if self.algo == "sd":
                self.step_sd()
            else:
                self.step_bfgs()
            J = self.solver.get_objective()
            print 30*"#", "Inverse iter: %i N Eval: %i"%(i, self.neval), "J: %.4e norm dJdbeta: %.4e"%(J,self.dJdbeta_l2norm)
            np.savetxt("inverse_solution/beta.%i"%i, self.beta)            
        if self.dostats:
            R = self.calculate_cholesky()
            self.sample_posterior()

        return self.beta
            
        
if __name__ == "__main__":
    Tinf = 50.0
    n = 33
    y = np.linspace(0.0, 1.0, n)
    q = np.ones_like(y)*Tinf
    beta = np.ones_like(y)
    solver = HeatSolver(y, beta, q, verbose=True, mode=sl.dns, Tinf = Tinf)
    solver.solve(dt=1e3)
    udns = solver.get_solution()
    
    solver = HeatSolver(y, beta, q, verbose=True, Tinf = Tinf)
    solver.solve(dt=1e3)
    u = solver.get_solution()
    dJdbeta_rans = solver.get_sensitivity()
    
    utarget = udns.copy()
    
    solver.set_target(utarget, beta)
    solver.solve()
    solver.set_sigmas(1e-1, 0.8)
    uprior = solver.get_solution()
    inverse_solver = InverseSolver(solver, beta)
    beta = inverse_solver.solve(maxiter=2000)
    upost = solver.get_solution()
    print beta
    from matplotlib.pyplot import *
    figure()
    plot(y, beta, label="True")
    plot(y, solver.get_true_beta(upost), label="Inverse")
    legend(loc="best")
    ylim(-0.3, 3.0)
    figure()
    plot(y, uprior, label='Prior')
    plot(y, upost, label="Posterior")
    plot(y, utarget, "o", label="Target")
    legend(loc="best")
    show()
    
