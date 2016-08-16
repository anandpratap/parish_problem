CFLAGS=-std=c++14  -O3
INC=-I/caslab_environment/local/anaconda2/include/python2.7/
INC+=-I/caslab_environment/local/anaconda2/lib/python2.7/site-packages/numpy/core/include/
INC+=-I/caslab_environment/local/adolc_2_6_1_intel/include
LIB+=-L/caslab_environment/local/adolc_2_6_1_intel/lib64 -ladolc -llapack -lblas
LIB+=-L/usr/local/lib -larmadillo

all:
	swig -c++ -python -o solver_wrap.cpp solver.i
	g++ -fPIC -c solver_wrap.cpp -o solver_wrap.o  $(CFLAGS) $(INC)
	g++ -shared solver_wrap.o -o _solver.so $(CFLAGS) $(LIB)
clean:
	rm *.o *.so solver_wrap* ../bin/_solver.so ../bin/solver.py ../bin/channelsolver.py
