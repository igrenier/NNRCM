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
  
  const arma::mat& solve_C = inv_sympd(C);
  
  double det = C_i - (vec_C.t() * solve_C * vec_C).eval()(0,0);
  
  return det;
}

// [[Rcpp::export]]
const arma::rowvec mvrnormArma(const arma::rowvec mean, const arma::mat variance) {
  int ncols = variance.n_cols;
  const arma::colvec vY = arma::randn(ncols);
  return mean + vY.t() * arma::chol(variance);
}

// [[Rcpp::export]]
const double rnormArma(const double mean, const double sd) {
  const arma::colvec vz = arma::randn(1);
  return mean + vz(0) * sd;
}

// [[Rcpp::export]]
const double mvnpdf(const arma::colvec x, const arma::colvec mean, const arma::mat sigma,
                    const int n) 
{
  double sqrt2pi = std::sqrt(2 * M_PI);
  arma::colvec quadformv  = (x - mean).t() * arma::inv_sympd(sigma) * (x - mean);
  double quadform = quadformv(0);
  double norm = std::pow(sqrt2pi, - n) *
    std::pow(arma::det(sigma), - 0.5);
  
  return norm * exp(-0.5 * quadform);
}

