#ifndef __UTILS_H
#define __UTILS_H
#include <iostream>
template <class T>
T*** allocate_3d_array(int nx, int ny, int nz){
    T*** A = new T**[nx];
    for(int i(0); i < nx; ++i){
			A[i] = new T*[ny];
			for(int j(0); j < ny; ++j){
				A[i][j] = new T[nz];					
				for(int k(0); k < nz; ++k){
					A[i][j][k]= 0.;
				}
			}
	}
    return A;
}
template <class T>
void release_3d_array(T*** A, int nx, int ny, int nz){
    for (int i = 0; i < nx; ++i){
			for (int j = 0; j < ny; ++j){
				delete[] A[i][j];
			}
			delete[] A[i];
	}
    delete[] A;
}

template <class T>
T** allocate_2d_array(int nx, int ny){
    T** A = new T*[nx];
    for(int i(0); i < nx; ++i){
		A[i] = new T[ny];
	}
    return A;
}

template <class T>
void release_2d_array(T** A, int nx, int ny){
    for (int i = 0; i < nx; ++i){
		delete[] A[i];
	}
    delete[] A;
}

template <class T, class Ty>
void calc_diff(int n, Ty *y, T *q, T *qy, int nvar=1){
    float deta = 1.0f;
	for(int i=0; i<nvar; i++){
		for(int j=1; j<n-1; j++){
			qy[i*n + j] = (q[i*n + j+1] - q[i*n + j-1])/(y[j+1] - y[j-1]);
			qy[i*n + 0] = (q[i*n + 1] - q[i*n + 0])/(y[1] - y[0]);
			qy[i*n + n-1] = (q[i*n + n-1] - q[i*n + n-2])/(y[n-1] - y[n-2]);
		}
	}
}
template <class T, class Ty>
void calc_diff_2(int n, Ty *y, T *q, T *qyy, int nvar=1){
	T h, r;
	for(int i=0; i<nvar; i++){
		for(int j=1; j<n-1; j++){
			h = y[j] - y[j-1];
			r = (y[j+1] - y[j])/h;
			qyy[i*n + j] = (2.0f*q[i*n + j-1]*r + q[i*n + j]*(-2.0f*r - 2.0f) + 2.0f*q[i*n + j+1])/(h*h*(r*r + r));
		}
	}
}

template<class T>
void array_print(int size, T array){
	for(int i=0; i<size; i++){
		std::cout << i<<" :: "<<array[i] << std::endl;
	}
}

template<class T>
void array_set_values(int size, T *array, T val){
	for(int i=0; i<size; i++){
		array[i] = val;
	}
}

template<class T>
void array_linspace(int size, T *array, T start, T end){
	T darray = (end - start)/(size - 1);
	array[0] = start;
	for(int i=1; i<size; i++){
		array[i] = array[i-1] + darray;
	}
}

template<class T>
T add(T a, T b){
	return a + b;
}

template<class T>
T sub(T a, T b){
	return a - b;
}

template<class T>
T mul(T a, T b){
	return a * b;
}

template<class T>
T div(T a, T b){
	if(fabs(b) > 1e-13){
		return a/b;
	}
	else{
		if(fabs(a) > 1e-13){
			return 1e13;
		}
		else{
			return 0.0;
		}
	}
	return 1e10;
}

#endif
