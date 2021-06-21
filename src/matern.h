//#ifndef MATERN_H
//#define MATERN_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

#include <RcppParallel.h>
// [[Rcpp::depends(RcppParallel)]]
using namespace RcppParallel;

#include <Rcpp.h>
using namespace Rcpp;
#include <math.h> 

//#ifdef __cplusplus
extern "C" {
//#endif

// SUBROUTINE MATERN
// Obtain parameters defining crustal motion model
// void rkmat_(std::vector *theta, std::vector *x, std::vector *y, int *n);
void rkmat_(double *theta, double *x, double *y, int *n);


//#ifdef __cplusplus
}
//#endif

//#endif // MATERN_H