#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

#include <RcppParallel.h>
// [[Rcpp::depends(RcppParallel)]]
using namespace RcppParallel;

#include <Rcpp.h>
using namespace Rcpp;

#include <math.h> 
#include "matern.h"
#include "covariance_fn.h"

// [[Rcpp::export]]
arma::mat matern_cov(arma::mat distance, double kappa, double phi) {
  
  // compute parameters needed for Fortran
  int n0 = distance.n_rows;
  int n = n0 * n0;
  arma::vec distance_vec = arma::vectorise(distance);
  arma::vec covariance_vec = distance_vec;
  
  
  // fill out parameters
  arma::vec parameters(2, fill::zeros);
  parameters(0) = phi;
  parameters(1) = kappa;
  
  // call Matern
  rkmat_(parameters.memptr(), distance_vec.memptr(), covariance_vec.memptr(), &n);
  
  // unvectorise the covariance matrix
  arma::mat covariance_prior = reshape(covariance_vec, n0, n0);
  
  return covariance_prior;
  
}

// [[Rcpp::export]]
double conditional_covariance(arma::mat& C, arma::colvec vec_C, double C_i) {
  
  const arma::mat& solve_C = inv(C);
  
  double det = C_i - (vec_C.t() * solve_C * vec_C).eval()(0,0);
  
  return det;
}

