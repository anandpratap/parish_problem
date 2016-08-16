#ifndef __LAMINARSOLVER_H
#define __LAMINARSOLVER_H
#define EPS 1e-14
#define AUTO_DIFF 2
#define M_PI 3.141592653589793238462643383279502884L /* pi */
#include <random>

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include <complex>
#include <armadillo>
#if AUTO_DIFF == 1
// use adept
#include "adept.h"
#elif AUTO_DIFF == 2
#include "adolc/adolc.h"
// adolc
#endif

enum Mode {rans, dns};
#if AUTO_DIFF == -1
typedef double atype;
#elif AUTO_DIFF == 1
using adept::adouble;
typedef adouble atype;
#elif AUTO_DIFF == 2
typedef adouble atype;
#endif


class HeatSolver{
protected:
	// system related variables
	int n;
	double *y, *q, *R;
	double *beta;
	double sigma_obs, sigma_prior;
	double Tinf;
	double *randarray;
	int objective_type, objective_index;
	bool verbose;
	Mode mode;
	// solver related variables
	atype *a_q_target, *a_beta_prior;
	atype *a_q, *a_R, *a_qy, *a_qyy;
	atype *a_beta;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution;
	
	atype *a_R_1, *a_R_2, *a_R_tmp;
#if AUTO_DIFF == 2
	double **dRdq_2d;
	double **dRdbeta_2d;
#endif
	double *dRdq, *dJdq;
	double *dRdbeta, *delJdbeta;
	int jac_freq;
	// matrix storage for linear algebra
	arma::mat dRdq_arma, R_arma, dJdq_arma;
	arma::mat dq_arma, psi_arma, dJdbeta_arma;
	arma::mat dRdbeta_arma, delJdbeta_arma;
	arma::mat psi_arma_t, dRdq_arma_t, dJdq_arma_t;
#if AUTO_DIFF == -1
#elif AUTO_DIFF == 1
	adept::Stack s;
#endif

	// input parameters
	double rtol_cutoff, atol_cutoff;
	int iteration, iteration_max;
	int iteration_ramp, iteration_print;
	double rtol, atol, atol_0;
	double dt;
public:
	HeatSolver(Mode modei=rans, double Tinfi = 50.0, bool verbosei = false);
	~HeatSolver(void);
	virtual void initialize(int nyi, double *yi, int nqi, double *qi, int nbetai, double *betai);
	int solve(double dti = 1e8, int iteration_maxi = 10, double rtol_cutoffi = 1e-10, double atol_cutoffi = 1e-10, int iteration_rampi=10);
    int step(void);
	void get_solution(int nqo, double *qo);
	void get_sensitivity(int nyo, double *dJdbetao);
	void set_target(int nqi, double *qi, int nbetai, double *betai);
	atype calc_objective(void);
	void set_beta(int nbetai, double *betai);
	void reset(void);
	void finalize(void);
	void set_sigmas(double sigma_obsi, double sigma_priori);
	void set_objective_type(int objective_typei, int objective_indexi);
	void set_print_frequency(int iteration_printi){
		iteration_print = iteration_printi;
	};
protected: 
	void setup_arma_solver(void);
	void setup_arma_adjoints(void);
	virtual void calc_residual(void);
	virtual void boundary_condition_forced(void);
	virtual void copy_to_atype(void);
	virtual void copy_from_atype(void);
	int calc_adjoint(void);
	int calc_sensitivity(void);
	void calc_jacobian_fd(double *jac);
	void calc_dRdbeta_fd(double *dRdbetai);
	void calc_dJdq_fd(double *dJdqi);
	void calc_delJdbeta_fd(double *delJdbetai);
	
};
HeatSolver::HeatSolver(Mode modei, double Tinfi, bool verbosei){
	distribution = std::normal_distribution<double>(0.0,0.1);
	mode = modei;
	Tinf = Tinfi;
	iteration = 0;
	iteration_ramp = 10;
	iteration_print = 1;
	rtol = 1.0;
	verbose = verbosei;
	jac_freq = 1;
	objective_type = 1;
	objective_index = -100;
}

HeatSolver::~HeatSolver(void){
	delete[] y;
	delete[] q;
	delete[] R;
	delete[] dRdq;
	delete[] beta;
	delete[] dJdq;
	delete[] dRdbeta;
	delete[] delJdbeta;
	delete[] randarray;
	delete[] a_q;
	delete[] a_qy;
	delete[] a_qyy;
	delete[] a_R;
	delete[] a_R_1;
	delete[] a_R_2;
	delete[] a_R_tmp;
	delete[] a_beta;
	delete[] a_beta_prior;
	delete[] a_q_target;
}

void HeatSolver::initialize(int nyi, double *yi, int nqi, double *qi, int nbetai, double *betai){
	n = nyi;
	y = new double[n]();
	q = new double[n]();
	R = new double[n]();
	randarray = new double[n]();
		
	beta = new double[n]();
	sigma_obs = 1.0;
	sigma_prior = 1.0;
	
	for(int i=0; i<n; i++){
		y[i] = yi[i];
		beta[i] = betai[i];
		randarray[i] = distribution(generator)*0;
	}
	for(int i=0; i<n; i++){
		q[i] = qi[i];
	}

	a_beta = new atype[n]();
	a_beta_prior = new atype[n]();
	a_q = new atype[n]();
	a_q_target = new atype[n]();
	a_qy = new atype[n]();

	a_qyy = new atype[n]();
	a_R = new atype[n]();

	a_R_1 = new atype[n]();
	a_R_2 = new atype[n]();
	a_R_tmp = new atype[n]();

	dRdq  = new double[n*n]();
	dJdq = new double[n]();
	dRdq_arma.zeros(n, n);
	R_arma.zeros(n, 1);
	dJdq_arma.zeros(1, n);
	copy_to_atype();
	dRdbeta = new double[n*n]();
	delJdbeta = new double[n]();
	dRdbeta_arma.zeros(n, n);
	delJdbeta_arma.zeros(1, n);

#if AUTO_DIFF == 2
	dRdq_2d = allocate_2d_array<double>(n, n);
	dRdbeta_2d = allocate_2d_array<double>(n, n);
#endif
}


int HeatSolver::solve(double dti, int iteration_maxi, double rtol_cutoffi, double atol_cutoffi, int iteration_rampi){
	dt = dti;
	iteration_max = iteration_maxi;
	iteration_ramp = iteration_rampi;
	rtol_cutoff = rtol_cutoffi;
	atol_cutoff = atol_cutoffi;
	
	while (1){
		int error = step();
		if(error == -1){
			printf("Solver failed!\n");
			return -1;
		}
			
		if(iteration > iteration_max || atol < atol_cutoff || rtol < rtol_cutoff)
			break;
		if(iteration > iteration_ramp)
			dt = dt/pow(rtol, 0.3);
	}
	copy_from_atype();
	return 0;
}

void HeatSolver::calc_jacobian_fd(double *jac){
	double dq = 1e-6;
	
	for(int i=0; i<n; i++){
		a_R_tmp[i] = a_R[i];
	}
	
	for(int i=0; i<n; i++){
		a_q[i] += dq;
		calc_residual();
		for(int j=0; j<n; j++){
			a_R_1[j] = a_R[j];
		}
		a_q[i] -= 2*dq;
		calc_residual();
		for(int j=0; j<n; j++){
			a_R_2[j] = a_R[j];
		}
		a_q[i] += dq;
		
		for(int j=0; j<n; j++){
#if AUTO_DIFF == 2
			jac[i*n + j] = ((a_R_1[j] - a_R_2[j])/(2.0*dq)).value();
#else
			jac[i*n + j] = (a_R_1[j] - a_R_2[j])/(2.0*dq);
#endif
		}
	}
	for(int i=0; i<n; i++){
		a_R[i] = a_R_tmp[i];
	}
}


void HeatSolver::calc_dRdbeta_fd(double *dRdbetai){
	atype dbeta = 1e-10;
	
	for(int i=0; i<n; i++){
		a_R_tmp[i] = a_R[i];
	}
	
	for(int i=0; i<n; i++){
		a_beta[i] += dbeta;
		calc_residual();
		for(int j=0; j<n; j++){
			a_R_1[j] = a_R[j];
		}
		a_beta[i] -= 2*dbeta;
		calc_residual();
		for(int j=0; j<n; j++){
			a_R_2[j] = a_R[j];
		}
		a_beta[i] += dbeta;
		
		for(int j=0; j<n; j++){
#if AUTO_DIFF == 2
			dRdbetai[i*n + j] = ((a_R_1[j] - a_R_2[j])/(2.0*dbeta)).value();
#else
			dRdbetai[i*n + j] = (a_R_1[j] - a_R_2[j])/(2.0*dbeta);
#endif
		}
	}
	for(int i=0; i<n; i++){
		a_R[i] = a_R_tmp[i];
	}
}

void HeatSolver::calc_dJdq_fd(double *dJdqi){
	double dq = 1e-6;
	
	for(int i=0; i<n; i++){
		a_q[i] += dq;
		atype Jp = calc_objective();
		a_q[i] -= 2*dq;
		atype Jm = calc_objective();
		a_q[i] += dq;
#if AUTO_DIFF == 2
		dJdqi[i] = ((Jp - Jm)/(2.0*dq)).value();
#else
		dJdqi[i] = (Jp - Jm)/(2.0*dq);
#endif
	}
}

void HeatSolver::calc_delJdbeta_fd(double *delJdbetai){
	double dbeta = 1e-10;
	for(int i=0; i<n; i++){
		a_beta[i] += dbeta;
		atype Jp = calc_objective();
		a_beta[i] -= 2*dbeta;
		atype Jm = calc_objective();
		a_beta[i] += dbeta;
#if AUTO_DIFF == 2
		delJdbetai[i] = ((Jp - Jm)/(2.0*dbeta)).value();
#else
		delJdbetai[i] = (Jp - Jm)/(2.0*dbeta);
#endif
	}
}

int HeatSolver::step(void){
#if AUTO_DIFF == -1
	calc_residual();
	if(iteration % jac_freq == 0)
		calc_jacobian_fd(dRdq);
#elif AUTO_DIFF == 1		
	s.new_recording();
	calc_residual();
	s.independent(a_q, n);
	s.dependent(a_R, n);
	if(iteration % jac_freq == 0)
		s.jacobian(dRdq);
#elif AUTO_DIFF == 2
	copy_from_atype();
	trace_on(1);
	for(int i=0; i<n; i++)
		a_q[i] <<= q[i];
	calc_residual();
	for(int i=0; i<n; i++)
		a_R[i] >>= R[i];
	trace_off();
	jacobian(1, n, n, q, dRdq_2d);
#endif
	setup_arma_solver();
	try{
		dq_arma = arma::solve(dRdq_arma, R_arma, arma::solve_opts::no_approx);
	}
	catch(...){
		return -1;
	}
	for(int i=0; i<n; i++){
		a_q[i] += dq_arma[i];
	}
	boundary_condition_forced();
	atol = norm(R_arma, 2)/(n);
	if(iteration == 0)
		atol_0 = atol;
	rtol = atol/atol_0;
	if(verbose && (iteration%iteration_print == 0))
		printf("iteration: %d absolute tol: %.2e relative tol: %.2e\n", iteration, atol, rtol);
	iteration += 1;
	return 0;
}

void HeatSolver::get_solution(int nqo, double *qo){
	for(int i=0; i<nqo; i++){
		qo[i] = q[i];
	}
}

void HeatSolver::get_sensitivity(int nyo, double *dJdbetao){
	calc_sensitivity();
	for(int i=0; i<nyo; i++){
		dJdbetao[i] = dJdbeta_arma[i];
	}
}

void HeatSolver::set_target(int nqi, double *qi, int nbetai, double *betai){
	for(int i=0; i<n; i++){
		a_beta_prior[i] = betai[i];
	}
	
	for(int i=0; i<n; i++){
		a_q_target[i] = qi[i];
	}
	copy_to_atype();
}


atype HeatSolver::calc_objective(void){
	atype objective = 0.0;
	for(int i=0; i<n; i++){
		if(objective_type == 1){
			objective += pow(a_q[i]-a_q_target[i],2)/pow(sigma_obs, 2) + pow(a_beta[i]-a_beta_prior[i], 2)/pow(sigma_prior, 2);
		}
		else if(objective_type == 2){
			if(objective_index == i)
				objective += a_q[i]-a_q_target[i];
		}
		else{
			printf("Wrong objective!\n");
		}
	}
	return objective;
}


void HeatSolver::set_beta(int nbetai, double *betai){
	for(int i=0; i<nbetai; i++){
		beta[i] = betai[i];
	}
	copy_to_atype();
}

void HeatSolver::reset(void){
	iteration = 0;
}

void HeatSolver::finalize(void){
#if AUTO_DIFF == -1
#elif AUTO_DIFF == 1
	s.deactivate();
#elif AUTO_DIFF == 2
#endif
}

void HeatSolver::set_sigmas(double sigma_obsi, double sigma_priori){
	sigma_obs = sigma_obsi;
	sigma_prior = sigma_priori;
}

void HeatSolver::set_objective_type(int objective_typei, int objective_indexi){
	objective_type = objective_typei;
	objective_index = objective_indexi;
};

void HeatSolver::boundary_condition_forced(void){
	a_q[0] = 0.0;
	a_q[n-1] = 0.0;
}

void HeatSolver::copy_to_atype(void){
	for(int i=0; i<n; i++){
		a_q[i] = q[i];
	}
	for(int i=0; i<n; i++){
		a_beta[i] = beta[i];
	}
	
}

void HeatSolver::copy_from_atype(void){
	for(int i=0; i<n; i++){
#if AUTO_DIFF == -1
		q[i] = a_q[i];
#elif AUTO_DIFF == 1
		q[i] = a_q[i].value();
#elif AUTO_DIFF == 2
		q[i] = a_q[i].value();
#endif
	}
	for(int i=0; i<n; i++){
#if AUTO_DIFF == -1
		beta[i] = a_beta[i];
#elif AUTO_DIFF == 1
		beta[i] = a_beta[i].value();
#elif AUTO_DIFF == 2
		beta[i] = a_beta[i].value();
#endif
	}
}


void HeatSolver::calc_residual(void){

	calc_diff(n, y, a_q, a_qy, 1);
	calc_diff_2(n, y, a_q, a_qyy, 1);
	double h = 0.0;
	atype eps;
	for(int j=1; j<n-1; j++){
		if(mode == rans){
			eps = 5e-4;
			a_R[j] = -a_qyy[j] - a_beta[j]*eps*(pow(Tinf,4) - pow(a_q[j],4));
		}
		else if(mode ==  dns){
			h = 0.5;
			eps = (1.0 + 5.0*sin(3.0*M_PI/200.0*a_q[j]) + exp(0.02*a_q[j]) + randarray[j]) * 1e-4;
			a_R[j] = -a_qyy[j] - eps*(pow(Tinf,4) - pow(a_q[j],4)) - h*(Tinf - a_q[j]);
		}
		else{
			std::cout<<"WRONG MODE OF OPERATION"<<std::endl;
		}
	}

	a_R[0] = a_q[0];
	a_R[n-1] = a_q[n-1];
};

void HeatSolver::setup_arma_solver(void){
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
#if AUTO_DIFF == 2
			dRdq_arma(i, j) = -dRdq_2d[i][j];
#else
			dRdq_arma(i, j) = -dRdq[i + j*(n)];
#endif
			if(i == j){
				dRdq_arma(i,j) += 1.0/dt;
			}
		}
#if AUTO_DIFF == -1
		R_arma[i] = a_R[i];
#elif AUTO_DIFF == 1
		R_arma[i] = a_R[i].value();
#elif AUTO_DIFF == 2
		R_arma[i] = a_R[i].value();
#endif
	}
	//	std::cout<<norm(dRdq_arma)<<std::endl;
}

int HeatSolver::calc_adjoint(void){
#if AUTO_DIFF == -1
	calc_jacobian_fd(dRdq);
	calc_dJdq_fd(dJdq);
#elif AUTO_DIFF == 1
	s.new_recording();
	calc_residual();
	s.independent(a_q, n);
	s.dependent(a_R, n);
	s.jacobian(dRdq);
	s.new_recording();
	atype objective = calc_objective();
	s.independent(a_q, n);
	s.dependent(&objective, 1);
	s.jacobian(dJdq);
#elif AUTO_DIFF == 2
	copy_from_atype();
	trace_on(1);
	for(int i=0; i<n; i++)
		a_q[i] <<= q[i];
	calc_residual();
	for(int i=0; i<n; i++)
		a_R[i] >>= R[i];
	trace_off();
	jacobian(1, n, n, q, dRdq_2d);

	double obj;
	trace_on(3);
	for(int i=0; i<n; i++)
		a_q[i] <<= q[i];
	atype objective = calc_objective();
	objective >>= obj;
	trace_off();
	gradient(3, n, q, dJdq);
#endif
	
	setup_arma_adjoints();

	dRdq_arma_t = dRdq_arma.t();
	dJdq_arma_t = dJdq_arma.t();
	
	try{
		psi_arma = arma::solve(dRdq_arma_t, -dJdq_arma_t, arma::solve_opts::no_approx);
	}
	catch(...){
		psi_arma.zeros(n, 1);
		return -1;
	}

	return 0;
}

void HeatSolver::setup_arma_adjoints(void){
 	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
#if AUTO_DIFF == 2
			dRdq_arma(i, j) = dRdq_2d[i][j];
#else
			dRdq_arma(i, j) = dRdq[i + j*(n)];
#endif
		}
#if AUTO_DIFF == -1
		R_arma[i] = a_R[i];
#elif AUTO_DIFF == 1
		R_arma[i] = a_R[i].value();
#elif AUTO_DIFF == 2
		R_arma[i] = a_R[i].value();
#endif
		dJdq_arma[i] = dJdq[i];
	}
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
#if AUTO_DIFF == 2
			dRdbeta_arma(i,j) = dRdbeta_2d[i][j];
#else
			dRdbeta_arma(i,j) = dRdbeta[i+j*(n)];
#endif
		}
	}
	for(int j=0; j<n; j++){
		delJdbeta_arma(0,j) = delJdbeta[j];
	}

}

int HeatSolver::calc_sensitivity(void){
	calc_adjoint();
#if AUTO_DIFF == -1
	calc_dRdbeta_fd(dRdbeta);
	calc_delJdbeta_fd(delJdbeta);
#elif AUTO_DIFF == 1
	s.new_recording();
	calc_residual();
	s.independent(a_beta, n);
	s.dependent(a_R, n);
	s.jacobian(dRdbeta);
	
	s.new_recording();
	atype objective = calc_objective();
	s.independent(a_beta, n);
	s.dependent(&objective, 1);
	s.jacobian(delJdbeta);
#elif AUTO_DIFF == 2
	copy_from_atype();
	trace_on(2);
	for(int i=0; i<n; i++)
		a_beta[i] <<= beta[i];
	calc_residual();
	for(int i=0; i<n; i++)
		a_R[i] >>= R[i];
	trace_off();
	jacobian(2, n, n, beta, dRdbeta_2d);

	double obj;
	trace_on(3);
	for(int i=0; i<n; i++)
		a_beta[i] <<= beta[i];
	atype objective = calc_objective();
	objective >>= obj;
	trace_off();
	gradient(3, n, beta, delJdbeta);
#endif
	setup_arma_adjoints();
	//	std::cout<<norm(psi_arma)<<std::endl;
	//std::cout<<"asss"<<norm(dRdbeta_arma)<<std::endl;
	psi_arma_t = psi_arma.t();
	dJdbeta_arma = delJdbeta_arma + psi_arma_t*dRdbeta_arma;
	return 0;
}


#endif
