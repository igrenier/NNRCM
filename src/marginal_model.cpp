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

struct marginal_predictive : public Worker {
  
  arma::cube& tmp_covariances;
  int& tmp_n_neighbors, tmp_mcmc_samples;
  double& tmp_a, tmp_n_obs;
  arma::mat& tmp_y;
  arma::mat& tmp_y_star;       // output cube to write to
  
  marginal_predictive(int& n_neighbors, arma::cube& covariances,
                       arma::mat& y, arma::mat& y_star, int& mcmc_samples,
                       double& a, double& n_obs)
    : tmp_n_neighbors(n_neighbors),
      tmp_covariances(covariances),
      tmp_y(y),
      tmp_y_star(y_star),
      tmp_mcmc_samples(mcmc_samples),
      tmp_a(a), tmp_n_obs(n_obs){}
  
  void operator()(std::size_t begin, std::size_t end) {
    
    for (std::size_t x=begin; x < end; x++) {
      
      int m  = tmp_n_neighbors;
      
      // extract the (m+1) X (m+1) covariance matrix
      arma::mat C_i_neighbors = tmp_covariances.slice(x);
      
      // extract the m X m neighbors covariance matrix
      arma::mat C_neighbors = C_i_neighbors.submat(1, 1, m, m);
      arma::mat C_neighbors_inv = inv_sympd(C_neighbors);
      
      // extract the m X 1 observations to neighbors covarianc vector
      arma::colvec C_neighbors_c = C_i_neighbors.submat(1, 0, m, 0);
      
      // compute conditional of C
      double det_cond = conditional_covariance(C_neighbors,
                                               C_neighbors_c,
                                               C_i_neighbors(0, 0));
      
      // sample phi
      std::default_random_engine generator;
      arma::colvec phi_array(tmp_mcmc_samples, fill::zeros); 
      std::gamma_distribution<double> phi_distribution((tmp_a - tmp_n_obs + m + 1) / 2, 2 / det_cond);
      for (int i=0; i<tmp_mcmc_samples; ++i) {
        phi_array[i] = 1 / phi_distribution(generator);
      }
     
      // sample gamma
      arma::mat gamma_array(tmp_mcmc_samples, m, fill::zeros);
      for (int i=0; i<tmp_mcmc_samples; i++) {
        arma::colvec mu = C_neighbors_inv * C_neighbors_c;
        arma::mat cov = phi_array[i] * C_neighbors_inv;
        gamma_array.row(i) = mvrnormArma(mu.t(), cov);

      }

      // sample y*
      for (int i=0; i<tmp_mcmc_samples; i++) {
        double y_star_mean = (gamma_array.row(i) * tmp_y.row(x).t()).eval()(0,0);
        double y_star_sd = std::sqrt(phi_array(i));
        tmp_y_star(x, i) =  R::rnorm(y_star_mean, y_star_sd);
      }
    }
  }
};

struct mv_marginal_predictive : public Worker {
  
  arma::cube& tmp_covariances;
  int& tmp_n_neighbors, tmp_mcmc_samples;
  double& tmp_a, tmp_n_obs;
  arma::mat& tmp_y;
  arma::mat& tmp_y_star;       // output cube to write to
  
  mv_marginal_predictive(int& n_neighbors, arma::cube& covariances,
                      arma::mat& y, arma::mat& y_star, int& mcmc_samples,
                      double& a, double& n_obs)
    : tmp_n_neighbors(n_neighbors),
      tmp_covariances(covariances),
      tmp_y(y),
      tmp_y_star(y_star),
      tmp_mcmc_samples(mcmc_samples),
      tmp_a(a), tmp_n_obs(n_obs){}
  
  void operator()(std::size_t begin, std::size_t end) {
    
    for (std::size_t x=begin; x < end; x++) {
      
      int m  = 2 * tmp_n_neighbors;
      
      // extract the (m+1) X (m+1) covariance matrix
      arma::mat C_i_neighbors = tmp_covariances.slice(x);
      
      // extract the m X m neighbors covariance matrix
      arma::mat C_neighbors = C_i_neighbors.submat(1, 1, m, m);
      arma::mat C_neighbors_inv = inv_sympd(C_neighbors);
      
      // extract the m X 1 observations to neighbors covarianc vector
      arma::colvec C_neighbors_c = C_i_neighbors.submat(1, 0, m, 0);
      
      // compute conditional of C
      double det_cond = conditional_covariance(C_neighbors,
                                               C_neighbors_c,
                                               C_i_neighbors(0, 0));
      
      // sample phi
      std::default_random_engine generator;
      arma::colvec phi_array(tmp_mcmc_samples, fill::zeros); 
      std::gamma_distribution<double> phi_distribution((tmp_a - tmp_n_obs + m + 1) / 2, 2 / det_cond);
      for (int i=0; i<tmp_mcmc_samples; ++i) {
        phi_array[i] = 1 / phi_distribution(generator);
      }
      
      // sample gamma
      arma::mat gamma_array(tmp_mcmc_samples, m, fill::zeros);
      for (int i=0; i<tmp_mcmc_samples; i++) {
        arma::colvec mu = C_neighbors_inv * C_neighbors_c;
        arma::mat cov = phi_array[i] * C_neighbors_inv;
        gamma_array.row(i) = mvrnormArma(mu.t(), cov);
        
      }
      
      // sample y*
      for (int i=0; i<tmp_mcmc_samples; i++) {
        double y_star_mean = (gamma_array.row(i) * tmp_y.row(x).t()).eval()(0,0);
        double y_star_sd = std::sqrt(phi_array(i));
        tmp_y_star(x, i) =  R::rnorm(y_star_mean, y_star_sd);
      }
    }
  }
};  
      
// [[Rcpp::export]]
double posterior_marginal(int n_neighbors, double n_obs, Rcpp::NumericVector& D,
                          Rcpp::NumericVector& Y_post, double a, double kappa, 
                         arma::colvec y, arma::Mat<int> W,
                         double phi, double sigma, double tau, double small, 
                         double a_sig, double b_sig, double a_tau, double b_tau, double a_nu) {

  // Create cubes
  arma::cube cube_D = Rcpp::as<arma::cube>(D);
  arma::cube cube_Y = Rcpp::as<arma::cube>(Y_post);
  arma::cube cube_C_i_neighbors(n_neighbors + 1, n_neighbors +1, n_obs, fill::zeros);
  arma::cube cube_C_i_neighbors_y(n_neighbors + 1, n_neighbors +1, n_obs, fill::zeros);

  // create the workers
  bool pred_ic = false;
  covariance covariance(a, n_obs, cube_D, n_neighbors,
                        tau, phi, kappa,
                        sigma, cube_C_i_neighbors, pred_ic);
  
  covariance_posterior covariance_posterior(n_neighbors, 
                                            cube_C_i_neighbors, W,
                                            cube_Y, cube_C_i_neighbors_y,
                                            y);

  // call the loop with parellelFor
  parallelFor(2, n_obs, covariance, n_obs / 10);
  parallelFor(2, n_obs, covariance_posterior, n_obs / 10);

  // compute the marginal likelihood
  marginal_likelihood marginal_likelihood(cube_C_i_neighbors,
                                          cube_C_i_neighbors_y,
                                          a,
                                          n_neighbors,
                                          n_obs);
  
  parallelReduce(2, n_obs, marginal_likelihood, n_obs / 10);
  double marginal = marginal_likelihood.tmp_marginal;
  
  // compute log prior
  double prior = (a_nu - 1) * log(phi) - a_nu * phi / small - (a_sig + 1) * log(sigma) - b_sig / sigma - 
      (a_tau + 1) * log(tau) - b_tau / tau - 2 * log(a);
      
  // compute normalizing constant
  // double constant = n_obs * (1 / 2 * log((a - n_obs + n_neighbors + 2) / 2 - 1) - 1 / 2 * log(M_PI));
  double constant = n_obs * (std::lgamma((a - n_obs + n_neighbors + 2) / 2) -
                             std::lgamma((a - n_obs + n_neighbors + 1) / 2) -
                             1 / 2 * log(M_PI));

  return  marginal + prior + constant;
}

// [[Rcpp::export]]
arma::mat posterior_marginal_prediction(int n_neighbors, double n_pred, Rcpp::NumericVector& D,
                                        Rcpp::NumericVector& Y_post, double a, double kappa, 
                                        double nu, double sigma, double tau,
                                        int mcmc_samples, double n_obs) {
  
  
  // Create cubes and matrices
  arma::cube cube_D = Rcpp::as<arma::cube>(D);
  arma::mat mat_Y = Rcpp::as<arma::mat>(Y_post);
  arma::cube cube_C(n_neighbors + 1, n_neighbors + 1, n_pred, fill::zeros);
  arma::mat mat_Y_star(n_pred, mcmc_samples, fill::zeros);
  
  // compute the matern for m+1 X m+1 matrix
  bool pred_ic = true;
  covariance covariance(a, n_obs, cube_D, n_neighbors,
                        tau, nu, kappa,
                        sigma, cube_C, pred_ic);

  parallelFor(0, n_pred, covariance, n_pred / 10);

  // obtain the predictive samples
  marginal_predictive marginal_predictive(n_neighbors, cube_C, mat_Y, mat_Y_star,
                                          mcmc_samples, a, n_obs);
  
  parallelFor(0, n_pred, marginal_predictive, n_pred / 10);

  return mat_Y_star;
  
}


// [[Rcpp::export]]
arma::mat mv_posterior_marginal_prediction(int n_neighbors, double n_pred, Rcpp::NumericVector& D,
                                        Rcpp::NumericVector& Y_post, arma::colvec kappa, 
                                        arma::colvec parameters,
                                        int mcmc_samples, double n_obs, 
                                        arma::Mat<int> Wnb) {
  
  // extract parameters
  int m2 = 2 * n_neighbors;
  double a = parameters(0);
  arma::colvec sigma = parameters.subvec(1, 3);
  arma::colvec nu(3, fill::zeros);
  nu(0) = parameters(4);
  nu(1) = parameters(5);
  nu(2) = std::sqrt(0.5 * (parameters(4) * parameters(4)+ parameters(5) * parameters(5)));
  arma::colvec tau = parameters.subvec(6,7);
  
  
  // Create cubes and matrices
  arma::cube cube_D = Rcpp::as<arma::cube>(D);
  arma::mat mat_Y = Rcpp::as<arma::mat>(Y_post);
  arma::cube cube_C(m2 + 1, m2 + 1, n_pred, fill::zeros);
  arma::mat mat_Y_star(n_pred, mcmc_samples, fill::zeros);
  
  // compute the matern for m+1 X m+1 matrix
  bool pred_ic = true;
  mv_covariance mv_covariance(a, n_obs, Wnb, cube_D, n_neighbors,
                        tau, nu, kappa,
                        sigma, cube_C, pred_ic);

  parallelFor(0, n_pred, mv_covariance, n_pred / 10);

  // obtain the predictive samples
  mv_marginal_predictive mv_marginal_predictive(n_neighbors, cube_C, mat_Y, mat_Y_star,
                                          mcmc_samples, a, n_obs);

  parallelFor(0, n_pred, mv_marginal_predictive, n_pred / 10);

  return mat_Y_star;
  
}

// [[Rcpp::export]]
arma::mat mv_posterior_marginal_prediction_coregionalization(
    int n_neighbors, double n_pred, Rcpp::NumericVector& D,
    Rcpp::NumericVector& Y_post, arma::colvec kappa, 
    arma::colvec parameters,
    int mcmc_samples, double n_obs, 
    arma::Mat<int> Wnb) {
  
  // extract parameters
  int m2 = 2 * n_neighbors;
  double a = parameters(0);
  arma::colvec A = parameters.subvec(1, 3);
  arma::colvec nu(3, fill::zeros);
  nu(0) = parameters(4);
  nu(1) = parameters(5);
  nu(2) = std::sqrt(0.5 * (parameters(4) * parameters(4)+ parameters(5) * parameters(5)));
  arma::colvec tau = parameters.subvec(6,7);
  
  
  // Create cubes and matrices
  arma::cube cube_D = Rcpp::as<arma::cube>(D);
  arma::mat mat_Y = Rcpp::as<arma::mat>(Y_post);
  arma::cube cube_C(m2 + 1, m2 + 1, n_pred, fill::zeros);
  arma::mat mat_Y_star(n_pred, mcmc_samples, fill::zeros);
  
  // compute the matern for m+1 X m+1 matrix
  bool pred_ic = true;
  mv_coregionalization mv_coregionalization(a, n_obs, Wnb, cube_D, n_neighbors,
                              tau, nu, kappa,
                              cube_C, pred_ic, A);
  
  parallelFor(0, n_pred, mv_coregionalization, n_pred / 10);
  
  // obtain the predictive samples
  mv_marginal_predictive mv_marginal_predictive(n_neighbors, cube_C, mat_Y, mat_Y_star,
                                                mcmc_samples, a, n_obs);
  
  parallelFor(0, n_pred, mv_marginal_predictive, n_pred / 10);
  
  return mat_Y_star;
  
}


// [[Rcpp::export]]
double mv_posterior_marginal(int n_neighbors, double n_obs, Rcpp::NumericVector& D,
                          Rcpp::NumericVector& Y_post, double a, arma::colvec kappa, 
                          arma::colvec y, arma::Mat<int> W, arma::Mat<int> Wnb,
                          arma::colvec phi, arma::colvec sigma, arma::colvec tau, 
                          double small, double a_sig, double b_sig, double a_tau, double b_tau, double a_nu) {
  // compute qm
  int m2 = 2 * n_neighbors;
  
  // Create cubes
  arma::cube cube_D = Rcpp::as<arma::cube>(D);
  arma::cube cube_Y = Rcpp::as<arma::cube>(Y_post);
  arma::cube cube_C_i_neighbors(m2 + 1, m2 + 1, n_obs, fill::zeros);
  arma::cube cube_C_i_neighbors_y(m2 + 1, m2 + 1, n_obs, fill::zeros);
  
  // create the workers
  bool pred_ic = false;
  mv_covariance mv_covariance(a, n_obs, Wnb, cube_D, n_neighbors,
                              tau, phi, kappa,
                              sigma, cube_C_i_neighbors, pred_ic);

  // i think this works still:
  covariance_posterior covariance_posterior(m2,
                                            cube_C_i_neighbors, W,
                                            cube_Y, cube_C_i_neighbors_y,
                                            y);

  // call the loop with parellelFor
  parallelFor(2, n_obs, mv_covariance, n_obs / 10);
  parallelFor(2, n_obs, covariance_posterior, n_obs / 10);
  
  // compute the marginal likelihood
  marginal_likelihood marginal_likelihood(cube_C_i_neighbors,
                                          cube_C_i_neighbors_y,
                                          a,
                                          m2,
                                          n_obs);

  parallelReduce(2, n_obs, marginal_likelihood, n_obs / 10);
  double marginal = marginal_likelihood.tmp_marginal;
  
  // compute log prior
  // double prior = 0;
  double prior = (a_nu - 1) * log(phi(0)) - a_nu * phi(0) / small + 
    (a_nu - 1) * log(phi(1)) - a_nu * phi(1) / small +
    (a_nu - 1) * log(phi(2)) - a_nu * phi(2) / small - 
    (a_sig + 1) * log(sigma(0)) - b_sig / sigma(0) - 
    (a_sig + 1) * log(sigma(1)) - b_sig / sigma(1) -
    (a_tau + 1) * log(tau(0)) - b_tau / tau(0) -
    (a_tau + 1) * log(tau(1)) - b_tau / tau(1) - 
    2 * log(a);
  
  // compute normalizing constant
  // double constant = n_obs * (1 / 2 * log((a - n_obs + n_neighbors + 2) / 2 - 1) - 1 / 2 * log(M_PI));
  double constant = n_obs * (std::lgamma((a - n_obs + 2 * n_neighbors + 2) / 2) -
                             std::lgamma((a - n_obs + 2 * n_neighbors + 1) / 2) -
                             1 / 2 * log(M_PI));
  
  return  marginal + prior + constant;
}

// [[Rcpp::export]]
double mv_posterior_marginal_coregionalization(int n_neighbors, double n_obs, Rcpp::NumericVector& D,
                             Rcpp::NumericVector& Y_post, double a, arma::colvec kappa, 
                             arma::colvec y, arma::Mat<int> W, arma::Mat<int> Wnb,
                             arma::colvec phi, arma::colvec A, arma::colvec tau, 
                             double small, double a_tau, double b_tau, double a_nu) {
  // compute qm
  int m2 = 2 * n_neighbors;
  
  // Create cubes
  arma::cube cube_D = Rcpp::as<arma::cube>(D);
  arma::cube cube_Y = Rcpp::as<arma::cube>(Y_post);
  arma::cube cube_C_i_neighbors(m2 + 1, m2 + 1, n_obs, fill::zeros);
  arma::cube cube_C_i_neighbors_y(m2 + 1, m2 + 1, n_obs, fill::zeros);
  
  // create the workers
  bool pred_ic = false;
  mv_coregionalization mv_coregionalization(a, n_obs, Wnb, cube_D, n_neighbors,
                              tau, phi, kappa,
                              cube_C_i_neighbors, pred_ic, A);
  
  // i think this works still:
  covariance_posterior covariance_posterior(m2,
                                            cube_C_i_neighbors, W,
                                            cube_Y, cube_C_i_neighbors_y,
                                            y);
  
  // call the loop with parellelFor
  parallelFor(2, n_obs, mv_coregionalization, n_obs / 10);
  parallelFor(2, n_obs, covariance_posterior, n_obs / 10);
  
  // compute the marginal likelihood
  marginal_likelihood marginal_likelihood(cube_C_i_neighbors,
                                          cube_C_i_neighbors_y,
                                          a,
                                          m2,
                                          n_obs);
  
  parallelReduce(2, n_obs, marginal_likelihood, n_obs / 10);
  double marginal = marginal_likelihood.tmp_marginal;
  
  // compute log prior
  // double prior = 0;
  double prior = (a_nu - 1) * log(phi(0)) - a_nu * phi(0) / small + 
    (a_nu - 1) * log(phi(1)) - a_nu * phi(1) / small -
    (a_tau + 1) * log(tau(0)) - b_tau / tau(0) -
    (a_tau + 1) * log(tau(1)) - b_tau / tau(1) - 
    2 * log(a);
  
  // compute normalizing constant
  // double constant = n_obs * (1 / 2 * log((a - n_obs + n_neighbors + 2) / 2 - 1) - 1 / 2 * log(M_PI));
  double constant = n_obs * (std::lgamma((a - n_obs + 2 * n_neighbors + 2) / 2) -
                             std::lgamma((a - n_obs + 2 * n_neighbors + 1) / 2) -
                             1 / 2 * log(M_PI));
  
  return  marginal + prior + constant;
}
