%module solver
%{
#define SWIG_FILE_WITH_INIT
#include "solver.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (int DIM1, double* ARGOUT_ARRAY1) {(int n1o, double *var1o)}
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int n2o, double *var2o)}
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int n3o, double *var3o)}
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int n4o, double *var4o)}
%apply (int DIM1, double* IN_ARRAY1) {(int nyi, double *yi)}
%apply (int DIM1, double* IN_ARRAY1) {(int nqi, double *qi)}
%apply (int DIM1, double* IN_ARRAY1) {(int nbetai, double *betai)}
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nqo, double *qo)}
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nyo, double *dJdbetao)}
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nxo, double *bary_xo)}
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nyo, double *bary_yo)}
    
%apply double { long double }
%typemap(out) adouble{
	$result = PyFloat_FromDouble(($1).value());
}
%include "solver.h"
