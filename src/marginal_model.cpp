#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

#include <RcppParallel.h>
// [[Rcpp::depends(RcppParallel)]]
using namespace RcppParallel;

#include <Rcpp.h>
using namespace Rcpp;


// [[Rcpp::export]]
double determinant_rcpp_arm(arma::mat& C, arma::colvec vec_C, double C_i) {

  const arma::mat& solve_C = inv(C);

  double det = C_i - (vec_C.t() * solve_C * vec_C).eval()(0,0);

  return det;
}

// [[Rcpp::export]]
arma::mat matern_cov(arma::mat distance, double kappa, double sigma double phi,
                     double alpha, double tau) {
  
  arma::mat distance_scaled = (2 * kappa)^(1/2) * distance / phi;
  arma::mat covariance = 2^(-(kappa - 1)) / gamma(kappa) * 
    (distance_scaled^kappa) * besselK(x = distance_scaled, nu = kappa);
  
  return covariance;
  
}

// [[Rcpp::export]]
double marginal_rcpp_arm(int n_neighbors, double n_obs, 
                         Rcpp::NumericVector& C, Rcpp::NumericVector& C_y, double a,
                         double phi, double sigma, double tau, double small) {

  // Create cubes
  arma::cube cube_C_i_neighbors = Rcpp::as<arma::cube>(C);
  arma::cube cube_C_i_neighbors_y = Rcpp::as<arma::cube>(C_y);
  arma::mat C_neighbors;
  arma::mat C_neighbors_posterior;
  arma::colvec C_neighbors_c;
  arma::colvec C_neighbors_posterior_c;

  double marginal = 0;

  // compute log prior
  double prior = log(phi) - phi * small / 2 - 4 * log(sigma) - 2 / sigma - 
    4 * log(tau) - 2 / tau - 2 * a;

  // compute normalizing constant
  double constant = n_obs * (1 / 2 * log((a - n_obs + n_neighbors + 2) / 2 - 1) - 1 / 2 * log(M_PI));

  for(int i = n_neighbors; i < n_obs; i++) {

    arma::mat C_i_neighbors = cube_C_i_neighbors.slice(i);
    arma::mat C_i_neighbors_y = cube_C_i_neighbors_y.slice(i);

    C_neighbors = C_i_neighbors.submat(1, 1, n_neighbors, n_neighbors);
    C_neighbors_posterior = C_i_neighbors_y.submat(1, 1, n_neighbors, n_neighbors);

    C_neighbors_c = C_i_neighbors.submat(1, 0, n_neighbors, 0);
    C_neighbors_posterior_c = C_i_neighbors_y.submat(1, 0, n_neighbors, 0);

    // compute determinant of C
    double det_cond = determinant_rcpp_arm(C_neighbors,
                                           C_neighbors_c,
                                           C_i_neighbors(0, 0));

    // compute determinant of C-posterior
    double det_posterior_cond = determinant_rcpp_arm(C_neighbors_posterior,
                                                     C_neighbors_posterior_c,
                                                     C_i_neighbors_y(0, 0));

    // compute the first expression determinant
    double det_sqrt = 1 / 2 * (log(arma::det(C_neighbors)) - log(arma::det(C_neighbors_posterior)));

    // copmute marginal terms
    marginal = marginal + (a - n_obs + n_neighbors) / 2 * log(det_cond) - (a - n_obs + n_neighbors + 1) / 2 * log(det_posterior_cond) + det_sqrt;

  }

  return marginal + prior + constant;
}
