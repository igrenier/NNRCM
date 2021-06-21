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
double phi_rcpp_arm(double a, int x_m, double w, arma::colvec& w_neighbors,
                    const arma::mat& solve_C, const arma::mat& cov_C, double cond_C, const arma::colvec& vec_C, int n_obs,
                    const arma::rowvec& gammar, double tau, double y, double sig) {

  arma::colvec gamma = gammar.t();

  double shape = a - n_obs + 1 + x_m + 1 / 2 * (x_m + 1);
  double q = (w - dot(gamma, w_neighbors)) * (w - dot(gamma, w_neighbors));
  double r = ((gamma - solve_C * vec_C).t() * cov_C * (gamma - solve_C * vec_C)).eval()(0,0);
  double rate = (a - n_obs - 1) * cond_C + q / (2 * sig)  + r * (a - n_obs - 1) / 2;
  double phi =  1 / R::rgamma(shape, 1 / rate); // R::rgamma(shape, scale)

  return phi;
}

// [[Rcpp::export]]
double w_rcpp_arm(arma::rowvec gammar, double phi, arma::colvec& w_neighbors, double y, double tau,
                  double beta, double sig) {

  arma::colvec gamma = gammar.t();

  double sigma = sig * phi * tau / (tau + sig * phi);
  double mu = sigma * (dot(gamma, w_neighbors) / (sig * phi) + (y - beta) / tau);
  double w_new = R::rnorm(mu, sqrt(sigma));

  return w_new;
}

// [[Rcpp::export]]
const arma::rowvec mvrnormArma(const arma::rowvec mean, const arma::mat variance) {
  int ncols = variance.n_cols;
  const arma::colvec Y = arma::randu(ncols);
  return mean + Y.t() * arma::chol(variance);
}

// [[Rcpp::export]]
arma::rowvec gamma_rcpp_arm(double a, double n_obs, const arma::colvec& vec_C, const arma::mat& solve_C,
                            const arma::mat& cov_C, double phi, arma::colvec& w_neighbors,
                            double w, double sig) {

  const arma::mat& cov = phi * arma::inv_sympd(w_neighbors * w_neighbors.t() / sig + (a - n_obs - 1) * cov_C);

  const arma::colvec& mu = (1 / phi) * cov * (w * w_neighbors / sig + (a - n_obs - 1) * vec_C);
  const arma::rowvec& gamma = mvrnormArma(mu.t(), cov);
  return gamma;
}

struct MCMC_update : public Worker {

  double& tmp_a, tmp_n_obs, tmp_tau, tmp_beta, tmp_sig;
  arma::cube& tmp_covariances;
  arma::colvec& tmp_w;
  int& tmp_n_neighbors;
  arma::Mat<int>& tmp_W;
  arma::colvec& tmp_phi;
  arma::colvec& tmp_y;
  arma::mat& tmp_samples;       // output matrix to write to

  MCMC_update(double& a, double& n_obs, arma::cube& covariances,
              arma::colvec& w, int& n_neighbors,
              arma::Mat<int>& W, double& tau, arma::colvec& phi,
              arma::colvec& y, double& beta, double& sig, arma::mat& samples)
    : tmp_a(a), tmp_n_obs(n_obs), tmp_covariances(covariances),
      tmp_w(w), tmp_n_neighbors(n_neighbors),
      tmp_W(W), tmp_tau(tau), tmp_phi(phi), tmp_y(y), tmp_beta(beta), tmp_sig(sig),
      tmp_samples(samples){}

  void operator()(std::size_t begin, std::size_t end) {

    for (std::size_t i=begin; i < end; i++) {
      
      int m_x = std::min<unsigned int>(tmp_n_neighbors, i);
      
      // extract w
      arma::colvec w_neighbors_c = tmp_w;
      arma::uvec idx = arma::conv_to<arma::uvec>::from(tmp_W.submat(i, 1, i, m_x));
      w_neighbors_c = tmp_w.elem(idx);

      // extract the covariances matrices and vectors
      // extract the (m+1) X (m+1) covariance matrix
      arma::mat E = tmp_covariances.slice(i);
      // extract the m X m neighbors covariance matrix
      arma::mat C = E.submat(1, 1, m_x, m_x);
      // extrqct the m X m neighbors precision matrix
      arma::mat B = inv_sympd(C);
      // extract the m X 1 observations to neighbors covarianc vector
      arma::colvec A = E.submat(1, 0, m_x, 0);
      // extract the 1 X 1 conditional value
      double F = conditional_covariance(E, A, E(0, 0));

      // sample gamma
      tmp_samples.submat(i, 2, i, m_x + 1) =
        gamma_rcpp_arm(tmp_a,
                       tmp_n_obs,
                       A,
                       B,
                       C,
                       tmp_phi(i),
                       w_neighbors_c,
                       tmp_w(i),
                       tmp_sig);
      
      // sample phi
      tmp_samples(i, 1) = phi_rcpp_arm(tmp_a,
                  m_x,
                  tmp_w(i),
                  w_neighbors_c,
                  B,
                  C,
                  F,
                  A,
                  tmp_n_obs,
                  tmp_samples.submat(i, 2, i, m_x + 1),
                  tmp_tau,
                  tmp_y(i),
                  tmp_sig);

      // sample random effects w
      tmp_samples(i, 0) = w_rcpp_arm(tmp_samples.submat(i, 2, i, m_x + 1),
                  tmp_samples(i, 1),
                  w_neighbors_c,
                  tmp_y(i),
                  tmp_tau,
                  tmp_beta,
                  tmp_sig);

    }

  }

};

// [[Rcpp::export]]
arma::mat data_loop_rcpp_arm(double a, double n_obs, Rcpp::NumericVector& D,
                             arma::colvec w, int n_neighbors,
                             arma::colvec m_location,
                             arma::Mat<int> W, double tau,  arma::colvec phi,
                             arma::colvec y, double beta, double sig, double kappa,
                             double nu) {


  // Gibbs sampling
  // create the output matrix
  arma::mat samples;
  samples.zeros(n_obs, n_neighbors + 2);

  // Create cubes
  arma::cube cube_D = Rcpp::as<arma::cube>(D);
  arma::cube cube_C_i(n_neighbors + 1, n_neighbors +1, n_obs, fill::zeros);
  
  // create the workers
  covariance covariance(a, n_obs, cube_D, n_neighbors,
                        tau, nu, kappa,
                        sig, cube_C_i);
  
  // call the loop with parellelFor
  parallelFor(2, n_obs, covariance, n_obs / 10);

  // create the worker
  MCMC_update MCMC_update(a, n_obs, cube_C_i,
                          w, n_neighbors, W, tau, phi, y, beta, sig,
                          samples);

  // call the loop with parellelFor
  parallelFor(n_neighbors, n_obs, MCMC_update, n_obs / 10);

  return samples;
}

// [[Rcpp::export]]
arma::mat data_loop_rcpp_no_nugget(double a, double n_obs, const Rcpp::List vec_C, Rcpp::List solve_C,
                                   Rcpp::List cov_C, arma::colvec& cond_C, int n_neighbors,
                                   const arma::colvec& m_location,
                                   const arma::Mat<int> W, double tau, const arma::colvec& phi,
                                   const arma::colvec y, double beta, double sig) {

  arma::mat samples;
  arma::colvec y_neighbors_c = y;
  samples.zeros(n_obs, n_neighbors + 2);

  for(int i = n_neighbors; i < n_obs; i++) {
    if (i < n_neighbors) {
      y_neighbors_c = y.subvec(0, i - 1);
    } else {
      arma::uvec idx = arma::conv_to<arma::uvec>::from(W.submat(i, 0, i, n_neighbors - 1));
      y_neighbors_c = y.elem(idx);
    }

    samples.submat(i, 2, i, m_location(i) + 1) = gamma_rcpp_arm(a,
                   n_obs,
                   Rcpp::as<arma::colvec>(vec_C[i]),
                   Rcpp::as<arma::mat>(solve_C[i]),
                   Rcpp::as<arma::mat>(cov_C[i]),
                   phi(i),
                   y_neighbors_c,
                   y(i),
                   sig);

    samples(i, 1) = phi_rcpp_arm(a,
            m_location(i),
            y(i),
            y_neighbors_c,
            Rcpp::as<arma::mat>(solve_C[i]),
            Rcpp::as<arma::mat>(cov_C[i]),
            cond_C(i),
            Rcpp::as<arma::colvec>(vec_C[i]),
            n_obs,
            samples.submat(i, 2, i, m_location(i) + 1),
            tau,
            y(i),
            sig);
  }

  return samples;
}

