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

struct marginal_likelihood : public Worker {
  
  arma::cube& tmp_covariances;
  arma::cube& tmp_covariances_posterior;
  double& tmp_a, tmp_n_obs;
  int& tmp_n_neighbors;
  double tmp_marginal;       // output of the loop
    
  // constructors  
    marginal_likelihood(arma::cube& covariances, arma::cube& covariances_posterior,
                        double& a, int& n_neighbors, double& n_obs) 
      : tmp_covariances(covariances),
        tmp_covariances_posterior(covariances_posterior),
        tmp_a(a),
        tmp_n_neighbors(n_neighbors),
        tmp_n_obs(n_obs),
        tmp_marginal(0){}
    
    marginal_likelihood(marginal_likelihood& marg, Split) : 
      tmp_covariances(marg.tmp_covariances),
      tmp_covariances_posterior(marg.tmp_covariances_posterior),
      tmp_a(marg.tmp_a),
      tmp_n_neighbors(marg.tmp_n_neighbors),
      tmp_n_obs(marg.tmp_n_obs),
      tmp_marginal(0) {}
    
    void operator()(std::size_t begin, std::size_t end) { 
      
      for (std::size_t i=begin; i < end; i++) {
      
        int m_x = std::min<unsigned int>(tmp_n_neighbors, i);
      
        // extract the (m+1) X (m+1) covariance matrix
        arma::mat C_i_neighbors = tmp_covariances.slice(i);
        arma::mat C_i_neighbors_y = tmp_covariances_posterior.slice(i);
      
        // extract the m X m neighbors covariance matrix
        arma::mat C_neighbors = C_i_neighbors.submat(1, 1, m_x, m_x);
        arma::mat C_neighbors_posterior = C_i_neighbors_y.submat(1, 1, m_x, m_x);
      
        // extract the m X 1 observations to neighbors covarianc vector
        arma::colvec C_neighbors_c = C_i_neighbors.submat(1, 0, m_x, 0);
        arma::colvec C_neighbors_posterior_c = C_i_neighbors_y.submat(1, 0, m_x, 0);
      
        // compute conditional of C
        double det_cond = conditional_covariance(C_neighbors,
                                                 C_neighbors_c,
                                                 C_i_neighbors(0, 0));
      
        // compute conditional of C-posterior
        double det_posterior_cond = conditional_covariance(C_neighbors_posterior,
                                                           C_neighbors_posterior_c,
                                                           C_i_neighbors_y(0, 0));
      
      // compute the first expression determinant
      // double det_sqrt = 1 / 2 * (log(arma::det(C_i_neighbors(0, 0))) - log(arma::det(C_i_neighbors_y(0, 0))));
      double det_sqrt = 1 / 2 * (log(C_i_neighbors(0, 0)) - log(C_i_neighbors_y(0, 0)));
      
      // copmute marginal terms
      tmp_marginal += (tmp_a - tmp_n_obs + m_x) / 2 * log(det_cond) - 
        (tmp_a - tmp_n_obs + m_x + 1) / 2 * log(det_posterior_cond) + 
        det_sqrt;
      
    }
  } 
    
    // join my value with that of another Sum
    void join(const marginal_likelihood& rhs) {
      tmp_marginal += rhs.tmp_marginal;
    }
}; 
      

// [[Rcpp::export]]
double posterior_marginal(int n_neighbors, double n_obs, Rcpp::NumericVector& D,
                         Rcpp::NumericVector& Y_post, double a, double kappa, 
                         arma::colvec y, arma::Mat<int> W,
                         double phi, double sigma, double tau, double small) {

  // Create cubes
  arma::cube cube_D = Rcpp::as<arma::cube>(D);
  arma::cube cube_Y = Rcpp::as<arma::cube>(Y_post);
  arma::cube cube_C_i_neighbors(n_neighbors + 1, n_neighbors +1, n_obs, fill::zeros);
  arma::cube cube_C_i_neighbors_y(n_neighbors + 1, n_neighbors +1, n_obs, fill::zeros);

  // create the workers
  covariance covariance(a, n_obs, cube_D, n_neighbors,
                        tau, phi, kappa,
                        sigma, cube_C_i_neighbors);
  
  covariance_posterior covariance_posterior(n_neighbors, 
                                                          cube_C_i_neighbors, W,
                                                          cube_Y, cube_C_i_neighbors_y);

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
  double prior = log(phi) - phi * small / 2 - 4 * log(sigma) - 2 / sigma - 
    4 * log(tau) - 2 / tau - 2 * a;

  // compute normalizing constant
  double constant = n_obs * (1 / 2 * log((a - n_obs + n_neighbors + 2) / 2 - 1) - 1 / 2 * log(M_PI));

  return  marginal + prior + constant;
}
