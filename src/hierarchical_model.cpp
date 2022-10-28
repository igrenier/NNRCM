#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

#include <RcppParallel.h>
// [[Rcpp::depends(RcppParallel)]]
using namespace RcppParallel;
// [[Rcpp::depends(BH)]]
#include <Rcpp.h>
using namespace Rcpp;

#include <math.h> 
#include "matern.h"
#include "covariance_fn.h"
#include <boost/math/special_functions/gamma.hpp>

struct hierarchical_predictive : public Worker {
  
  arma::cube& tmp_covariances;
  arma::colvec& tmp_sigma_array;
  arma::colvec& tmp_tau_array;
  int& tmp_n_neighbors, tmp_mcmc_samples;
  double& tmp_a, tmp_n_obs;
  arma::cube& tmp_w;
  arma::mat& tmp_X;
  arma::mat& tmp_beta;
  arma::mat& tmp_y_star;       // output mat to write to
  
  hierarchical_predictive(int& n_neighbors, arma::cube& covariances,
                      arma::cube& w, arma::mat& y_star, int& mcmc_samples,
                      double& a, double& n_obs, arma::colvec& sigma_array,
                      arma::colvec& tau_array, arma::mat& X, arma::mat& beta)
    : tmp_n_neighbors(n_neighbors),
      tmp_covariances(covariances),
      tmp_w(w),
      tmp_y_star(y_star),
      tmp_mcmc_samples(mcmc_samples),
      tmp_a(a), tmp_n_obs(n_obs),
      tmp_sigma_array(sigma_array),
      tmp_tau_array(tau_array),
      tmp_X(X),
      tmp_beta(beta){}
  
  void operator()(std::size_t begin, std::size_t end) {
    
    for (std::size_t x=begin; x < end; x++) {
      
      double m  = tmp_n_neighbors;
      
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
      arma::colvec new_phi(tmp_mcmc_samples, fill::zeros); 
      std::gamma_distribution<double> phi_distribution((tmp_a - tmp_n_obs + m + 1) / 2, 2 / det_cond);
      for (int i=0; i<tmp_mcmc_samples; ++i) {
        new_phi(i) = 1 / phi_distribution(generator);
      }
      
      // sample gamma
      arma::mat new_gamma(tmp_mcmc_samples, m, fill::zeros);
      for (int i=0; i<tmp_mcmc_samples; i++) {
        arma::colvec mu = C_neighbors_inv * C_neighbors_c;
        arma::mat cov = new_phi(i) * C_neighbors_inv;
        new_gamma.row(i) = mvrnormArma(mu.t(), cov);

      }
 
      // sample w
      arma::colvec new_w(tmp_mcmc_samples, fill::zeros);
      for (int i=0; i<tmp_mcmc_samples; i++) {
        arma::mat tmp_w_sub = tmp_w.slice(x);
        double w_mean = dot(new_gamma.row(i), tmp_w_sub.row(i));
        double w_sd = std::sqrt(new_phi(i) * tmp_sigma_array(i));
        new_w(i) =  rnormArma(w_mean, w_sd);
      }
      
      // sample y
      for (int i=0; i<tmp_mcmc_samples; i++) {
        double y_star_mean = new_w(i) + dot(tmp_X.row(x), tmp_beta.col(i));
        double y_star_sd = std::sqrt(tmp_tau_array(i));
        // tmp_y_star(x, i) = rnormArma(y_star_mean, y_star_sd);
        tmp_y_star(x, i) = y_star_mean;
      }
      
    }
  }
};

struct mv_hierarchical_predictive : public Worker {
  
  arma::cube& tmp_covariances;
  arma::mat& tmp_tau_array;
  int& tmp_n_neighbors, tmp_mcmc_samples;
  double& tmp_a, tmp_n_obs;
  arma::cube& tmp_w;
  arma::mat& tmp_X;
  arma::mat& tmp_Aw;
  arma::Mat<int>& tmp_Wnb;
  arma::mat& tmp_beta;
  arma::mat& tmp_w_star;       // output mat to write to
  
  mv_hierarchical_predictive(int& n_neighbors, arma::cube& covariances,
                          arma::cube& w, arma::mat& w_star, int& mcmc_samples,
                          double& a, double& n_obs, 
                          arma::mat& tau_array, arma::mat& X, arma::mat& beta,
                          arma::Mat<int>& Wnb, arma::mat& Aw)
    : tmp_n_neighbors(n_neighbors),
      tmp_covariances(covariances),
      tmp_w(w),
      tmp_w_star(w_star),
      tmp_mcmc_samples(mcmc_samples),
      tmp_a(a), tmp_n_obs(n_obs),
      tmp_tau_array(tau_array),
      tmp_X(X),
      tmp_beta(beta),
      tmp_Wnb(Wnb),
      tmp_Aw(Aw){}
  
  void operator()(std::size_t begin, std::size_t end) {
    
    for (std::size_t x=begin; x < end; x++) {
      
      double m  = tmp_n_neighbors;
      double source = tmp_Wnb(x, 0) - 1;
      
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
      arma::colvec new_phi(tmp_mcmc_samples, fill::zeros); 
      std::gamma_distribution<double> phi_distribution((tmp_a - tmp_n_obs + m + 1) / 2, 2 / det_cond);
      for (int i=0; i<tmp_mcmc_samples; ++i) {
        new_phi(i) = 1 / phi_distribution(generator);
      }
      
      // sample gamma
      arma::mat new_gamma(tmp_mcmc_samples, m, fill::zeros);
      for (int i=0; i<tmp_mcmc_samples; i++) {
        arma::colvec mu = C_neighbors_inv * C_neighbors_c;
        arma::mat cov = new_phi(i) * C_neighbors_inv;
        new_gamma.row(i) = mvrnormArma(mu.t(), cov);
        
      }
      
      // sample w
      arma::colvec new_w(tmp_mcmc_samples, fill::zeros);
      for (int i=0; i<tmp_mcmc_samples; i++) {
        arma::mat tmp_w_sub = tmp_w.slice(x);
        double w_mean = dot(new_gamma.row(i), tmp_w_sub.row(i));
        double w_sd = std::sqrt(new_phi(i));
        tmp_w_star(x, i) =  rnormArma(w_mean, w_sd);
      }
      
    }
  }
};

struct mv_hierarchical_predictive_final_step : public Worker {
  
  arma::mat& tmp_tau_array;
  int& tmp_mcmc_samples;
  arma::mat& tmp_new_w;
  arma::mat& tmp_X;
  arma::mat& tmp_Aw;
  arma::Mat<int>& tmp_Wnb;
  arma::mat& tmp_beta;
  arma::mat& tmp_y_star;       // output mat to write to
  
  mv_hierarchical_predictive_final_step(arma::mat& new_w, arma::mat& y_star, int& mcmc_samples,
                             arma::mat& tau_array, arma::mat& X, arma::mat& beta,
                             arma::Mat<int>& Wnb, arma::mat& Aw)
    : tmp_new_w(new_w),
      tmp_y_star(y_star),
      tmp_mcmc_samples(mcmc_samples),
      tmp_tau_array(tau_array),
      tmp_X(X),
      tmp_beta(beta),
      tmp_Wnb(Wnb),
      tmp_Aw(Aw){}
  
  void operator()(std::size_t begin, std::size_t end) {
    
    for (std::size_t x=begin; x < end; x++) {

      double source = tmp_Wnb(x, 0) - 1;
      double other = 0;
      if(source == 0){
        other = 1;
      }
      
      // extract the spatial random effects vector
      arma::colvec w_s(2, fill::zeros);
      for (int i=0; i<tmp_mcmc_samples; i++) {
        w_s(0) = tmp_new_w(2 * x, i);
        w_s(1) = tmp_new_w(2 * x + 1, i);
        
        // sample y_1(s)
        double y_star_mean = dot(tmp_Aw.row(0), w_s); //+ dot(tmp_X.row(x), tmp_beta.col(i));
        double y_star_sd = std::sqrt(tmp_tau_array(source, i));
        tmp_y_star(2 * x, i) = rnormArma(y_star_mean, y_star_sd);
        
        // sample y_2(s)
        y_star_mean = dot(tmp_Aw.row(1), w_s); //+ dot(tmp_X.row(x), tmp_beta.col(i));
        y_star_sd = std::sqrt(tmp_tau_array(source, i));
        tmp_y_star(2 * x + 1, i) = rnormArma(y_star_mean, y_star_sd);
      }

    }
  }
};

struct w_neighbors_update : public Worker {
  
  arma::Mat<int>& tmp_W;
  arma::colvec& tmp_w;
  arma::mat& tmp_gamma;
  double& tmp_n_neighbors;
  arma::colvec& tmp_w_neighbors;
  
  w_neighbors_update(arma::Mat<int>& W, arma::colvec& w, arma::mat& gamma,
                     double& n_neighbors, arma::colvec& w_neighbors) :
    tmp_W(W), tmp_w(w), tmp_gamma(gamma), 
    tmp_n_neighbors(n_neighbors), tmp_w_neighbors(w_neighbors){}
  
  void operator()(std::size_t begin, std::size_t end) {
    
    for (std::size_t i=begin; i < end; i++) {
      
      int m_x = std::min<unsigned int>(tmp_n_neighbors, i);
      
      // extract w
      arma::colvec w_neighbors_c(m_x, fill::zeros);
      arma::uvec idx = arma::conv_to<arma::uvec>::from(tmp_W.submat(i, 1, i, m_x));
      w_neighbors_c = tmp_w.elem(idx);
      
      // compute the gamma * w_neighbors
      tmp_w_neighbors(i) = dot(w_neighbors_c, tmp_gamma.submat(i, 0, i, m_x - 1));
      
    }
  }
  };

// [[Rcpp::export]]
double alpha_sample(double a_cur, double a_prior, double a_tuning, double n_obs,
                    arma::cube C_inv, arma::colvec C_cond, int n_neighbors,
                    arma::mat g, arma::mat C_vec, arma::colvec phi, arma::cube C_all,
                    arma::Mat<int> W, arma::colvec w_cur) {
  
  // create the output
  double alpha = 0;
  
  // propose a new alpha
  double a_prop = std::max(rnormArma(a_cur, a_tuning), n_obs + 2);
  
  // compute the prior under both current and proposed value
  double prior_prop = - (a_prior + 1) * log(a_prop);
  double prior_cur = -(a_prior + 1) * log(a_cur);
  
  // compute the constant at the front of the reduced marginal
  double constant_prop = n_obs * (std::lgamma((a_prop - n_obs + n_neighbors + 2) / 2) -
                             std::lgamma((a_prop - n_obs + n_neighbors + 1) / 2) -
                             1 / 2 * log(M_PI));
  double constant_cur = n_obs * (std::lgamma((a_cur - n_obs + n_neighbors + 2) / 2) -
                                  std::lgamma((a_cur - n_obs + n_neighbors + 1) / 2) -
                                  1 / 2 * log(M_PI));

  // compute the new covariance 
  arma::cube C_all_prop = (a_prop - n_obs - 1) / (a_cur - n_obs - 1) * C_all;
  
  // compute the marginal
  arma::cube cube_C_i_neighbors_w(n_neighbors + 1, n_neighbors +1, n_obs, fill::zeros);
  arma::cube cube_C_i_neighbors_w_prop(n_neighbors + 1, n_neighbors +1, n_obs, fill::zeros);
  covariance_posterior covariance_posterior_cur(n_neighbors,
                                            C_all, W,
                                            C_all, cube_C_i_neighbors_w,
                                            w_cur);
  covariance_posterior covariance_posterior_prop(n_neighbors,
                                                C_all_prop, W,
                                                C_all_prop, cube_C_i_neighbors_w_prop,
                                                w_cur);

  // call the loop with parellelFor
  parallelFor(2, n_obs, covariance_posterior_prop, n_obs / 10);
  parallelFor(2, n_obs, covariance_posterior_cur, n_obs / 10);
  
  // compute the marginal likelihood
  marginal_likelihood marginal_likelihood(C_all_prop,
                                          cube_C_i_neighbors_w_prop,
                                          a_prop,
                                          n_neighbors,
                                          n_obs);

  parallelReduce(2, n_obs, marginal_likelihood, n_obs / 10);
  double marginal_prop = marginal_likelihood.tmp_marginal;
  
  // compute the marginal likelihood
  marginal_likelihood2 marginal_likelihood2(C_all,
                                          cube_C_i_neighbors_w,
                                          a_cur,
                                          n_neighbors,
                                          n_obs);
  
  parallelReduce(2, n_obs, marginal_likelihood2, n_obs / 10);
  double marginal_cur = marginal_likelihood2.tmp_marginal;

    
  // // compute the likelihood under both current and proposed value
  // a_likelihood a_likelihood(a_prop, a_cur, n_neighbors, n_obs,
  //                           C_inv, g, C_vec, C_cond, phi);
  // 
  // parallelReduce(2, n_obs, a_likelihood, n_obs / 10);
  // double likelihood = a_likelihood.tmp_likelihood;

  // accept/reject the value
  double posterior_ratio = prior_prop - prior_cur + constant_prop - constant_cur
    + marginal_prop - marginal_cur;
  
  arma::colvec u = randu(1);
  if(log(u(0)) < posterior_ratio) {
    alpha = a_prop;
  } else {
    alpha = a_cur;
  }
  
  return alpha;
  
}

// [[Rcpp::export]]
arma::cube range_sample(double nu_cur, double nu_tuning, arma::colvec nu_prior, double n_obs, 
                    arma::cube C_inv, arma::colvec C_cond, 
                    int n_neighbors, arma::mat g, arma::mat C_vec, arma::colvec phi,
                    arma::cube C_all, double a,
                    arma::cube D, double nugget, double kappa, 
                    arma::Mat<int> W, arma::colvec w_cur) {
  
  // create the output
  double nu = 0;
  arma::cube C_new(n_neighbors + 1, n_neighbors + 1, n_obs, fill::zeros);
  
  // propose a new nu
  double nu_prop = std::abs(rnormArma(nu_cur, nu_tuning));
  
  // compute the prior under both current and proposed value
  double prior_prop = boost::math::gamma_p_derivative(nu_prior(0), nu_prop / nu_prior(1)) / nu_prior(1);
  double prior_cur = boost::math::gamma_p_derivative(nu_prior(0), nu_cur / nu_prior(1)) / nu_prior(1);
  
  // compute the covariance for the new range parameter
  arma::cube cube_C_i_prop(n_neighbors + 1, n_neighbors + 1, n_obs, fill::zeros); 
  double sig_unit = 1;
  bool pred_ic = false;
  
  covariance covariance(a, n_obs, D, n_neighbors,
             nugget, nu_prop, kappa,
             sig_unit, cube_C_i_prop, pred_ic);

  // call the loop with parellelFor
  parallelFor(2, n_obs, covariance, n_obs / 10);
  
  
  // compute the posterior covariance for both ranges
  arma::cube cube_C_post_cur(n_neighbors + 1, n_neighbors +1, n_obs, fill::zeros);
  arma::cube cube_C_post_prop(n_neighbors + 1, n_neighbors +1, n_obs, fill::zeros);
  covariance_posterior covariance_posterior_cur(n_neighbors,
                                            C_all, W,
                                            C_all, cube_C_post_cur,
                                            w_cur);
  // call the loop with parellelFor
  parallelFor(2, n_obs, covariance_posterior_cur, n_obs / 10);
  
  covariance_posterior covariance_posterior_prop(n_neighbors,
                                            cube_C_i_prop, W,
                                            cube_C_i_prop, cube_C_post_prop,
                                            w_cur);
  
  // call the loop with parellelFor
  parallelFor(2, n_obs, covariance_posterior_prop, n_obs / 10);
  
  
  // compute the marginal likelihood
  marginal_likelihood marginal_likelihood(C_all,
                                          cube_C_post_cur,
                                          a,
                                          n_neighbors,
                                          n_obs);
  
  parallelReduce(2, n_obs, marginal_likelihood, n_obs / 10);
  double marginal_cur = marginal_likelihood.tmp_marginal;
  
  // compute the marginal likelihood
  marginal_likelihood2 marginal_likelihood2(cube_C_i_prop,
                                            cube_C_post_prop,
                                            a,
                                            n_neighbors,
                                            n_obs);
  
  parallelReduce(2, n_obs, marginal_likelihood2, n_obs / 10);
  double marginal_prop = marginal_likelihood2.tmp_marginal;
  
  // // compute the likelihood under both current and proposed value
  // nu_likelihood nu_likelihood(a, n_neighbors, n_obs,
  //                             C_inv, g, C_vec, C_cond, phi, cube_C_i_prop);
  // 
  // parallelReduce(2, n_obs, nu_likelihood, n_obs / 10);
  // double likelihood = nu_likelihood.tmp_likelihood;
  // 
  // accept/reject the value
  double posterior_ratio = log(prior_prop) - log(prior_cur) + marginal_prop - marginal_cur;
  arma::colvec u = randu(1);
  if(log(u(0)) < posterior_ratio) {
    nu = nu_prop;
    C_new = cube_C_i_prop;
  } else {
    nu = nu_cur;
    C_new = C_all;
  }
  
  C_new(0,0,0) = nu;
  
  return C_new;
  
}

  
// [[Rcpp::export]]
double sig_sample(arma::colvec sigma_prior, arma::Mat<int> W, arma::mat samples,
                  double n_neighbors, double n_obs) {
  
    // extract current values
    arma::colvec phi = samples.col(1);
    arma::colvec w = samples.col(0);
    arma::mat gamma = samples.cols(2, n_neighbors + 1);
  
    // compute the posterior shape
    double shape = sigma_prior(0) + n_obs / 2;
  
    // compute the posterior scale
      // create the vector gamma * w_neighbors
      arma::colvec mean_neighbors(n_obs, fill::zeros);
        // fill out the vector
        w_neighbors_update w_neighbors_update(W, w, gamma,
                                              n_neighbors, mean_neighbors);
    
        parallelFor(2, n_obs, w_neighbors_update, n_obs / 10);
    
      // compute the difference between w and gamma * w_neighbors
    arma::colvec norm_rate = (w.subvec(2, n_obs - 1) - mean_neighbors.subvec(2, n_obs - 1));
    double scale = sigma_prior(1) + dot(norm_rate / phi.subvec(2, n_obs - 1), norm_rate) / 2;
    
    std::default_random_engine generator;
    std::gamma_distribution<double> gam_distribution(shape, 1 / scale);
    double sigma_new = 1 / gam_distribution(generator);
    
    return sigma_new;
    
}

// [[Rcpp::export]]
double tau_sample(arma::colvec tau_prior, arma::colvec Y,
                  arma::mat samples, arma::colvec beta,
                  double n_obs) {

  arma::colvec w = samples.col(0);
  
  double shape = tau_prior(0) + n_obs / 2;
  arma::colvec mean = Y - w - beta;
  double rate = tau_prior(1) + dot(mean, mean) / 2;
  //double tau = 1 / R::rgamma(shape, 1 / rate);
  std::default_random_engine generator;
  std::gamma_distribution<double> tau_distribution(shape, 1 / rate);
  double tau = 1 / tau_distribution(generator);

  return tau;

}

// [[Rcpp::export]]
arma::rowvec beta_sample(double beta_prior, arma::colvec Y, arma::colvec w, arma::mat X,
                         double tau, int n_covariates) {
  
  arma::mat cov = arma::inv_sympd((1 / tau) * X.t() * X + 1 / beta_prior * arma::eye(n_covariates, n_covariates));
  arma::colvec mu = (1 / tau) * cov * X.t() * (Y - w);
  
  arma::rowvec beta = mvrnormArma(mu.t(), cov);
  return beta;
}

// [[Rcpp::export]]
double phi_rcpp_arm(double a, int x_m, double w, arma::colvec& w_neighbors,
                    const arma::mat& solve_C, const arma::mat& cov_C, double cond_C, const arma::colvec& vec_C, int n_obs,
                    const arma::rowvec& gammar, double sig) {

  arma::colvec gamma = gammar.t();

  double shape = (a - n_obs + 1 + x_m) / 2 + 1 / 2 * (x_m + 1);
  double q = (w - dot(gamma, w_neighbors)) * (w - dot(gamma, w_neighbors));
  double r = ((gamma - solve_C * vec_C).t() * cov_C * (gamma - solve_C * vec_C)).eval()(0,0);
  // double rate = 1000 * cond_C / 2 + q / (2 * sig)  + 1000 * r / 2;
  double rate = cond_C / 2 + q / (2 * sig) + r / 2;
  
  std::default_random_engine generator;
  std::gamma_distribution<double> ph_distribution(shape, 1 / rate);
  double phi = 1 / ph_distribution(generator);

  return phi;
}

// [[Rcpp::export]]
double w_rcpp_arm(arma::rowvec gammar, double phi, arma::colvec& w_neighbors, double y, double tau,
                  double beta, double sig, arma::colvec gamma_old, arma::colvec b_old,
                  arma::colvec phi_old) {

  arma::colvec gamma = gammar.t();

  double sigma = 1 / (1 / tau + 1 / (sig * phi) + dot(gamma_old / phi_old, gamma_old) / sig);
  double mu = sigma * (dot(gamma, w_neighbors) / (sig * phi) + (y - beta) / tau + 
                       dot(gamma_old / phi_old, b_old) / sig);
  double w_new = rnormArma(mu, sqrt(sigma));

  return w_new;
}

// [[Rcpp::export]]
double mv_w_rcpp_arm(arma::rowvec gammar, double phi, arma::colvec& w_neighbors, arma::colvec y, 
                     arma::colvec tau,
                  double beta, arma::colvec gamma_old, arma::colvec b_old,
                  arma::colvec phi_old, arma::Row<int> Yt, arma::colvec A_w, arma::colvec A_y,
                  double w_old, double source, double other, double ix)  {
  
  arma::colvec gamma = gammar.t();
  
  arma::colvec y_star(2, fill::zeros);
  arma::uvec iduy = arma::conv_to<arma::uvec>::from(Yt.subvec(1, 2));
  // arma::uvec iduy0 = arma::conv_to<arma::uvec>::from(Yt(1));
  // arma::uvec iduy1 = arma::conv_to<arma::uvec>::from(Yt(2));
  double AYb_t = 0;
  double AA_t = 0;
  if(Yt(0) == 0) {
    y_star = y.elem(iduy) - A_y * w_old;
    AYb_t = dot(A_w, (y_star - beta) / tau);
    AA_t = dot(A_w, A_w / tau);
  }
  if(Yt(0) == 2) {
    y_star(0) = y(iduy(0)) - A_y(0) * w_old;
    AYb_t = A_w(0) * (y_star(0) - beta) / tau(0);
    AA_t = A_w(0) * A_w(0) / tau(0);
  }
  if(Yt(0) == 1) {
    y_star(0) = y(iduy(1)) - A_y(1) * w_old;
    AYb_t = A_w(1) * (y_star(0) - beta) / tau(1);
    AA_t = A_w(1) * A_w(1) / tau(1);
  }
  
  double sigma = 1 / (AA_t + 1 / phi + dot(gamma_old / phi_old, gamma_old));
  double mu = sigma * (dot(gamma, w_neighbors) / phi + AYb_t +
                       dot(gamma_old / phi_old, b_old));
  // double sigma = 1 / (1 / tau(source) + 1 / phi + dot(gamma_old / phi_old, gamma_old));
  // double mu = sigma * (dot(gamma, w_neighbors) / phi + (y(ix) - beta) / tau(source) +
  //                      dot(gamma_old / phi_old, b_old));
  // double sigma = 1 / (1 / tau(source) + 1 / phi);
  // double mu = sigma * (dot(gamma, w_neighbors) / phi + (y(ix) - beta) / tau(source));
  
  double w_new = rnormArma(mu, sqrt(sigma));
  //double w_new = rnormArma(mu, sqrt(sigma));

  return w_new;
  // return 0;
}

// [[Rcpp::export]]
arma::rowvec gamma_rcpp_arm(double a, double n_obs, const arma::colvec& vec_C, const arma::mat& solve_C,
                            const arma::mat& cov_C, double phi, arma::colvec& w_neighbors,
                            double w, double sig) {

  // const arma::mat& cov = phi * arma::inv_sympd(w_neighbors * w_neighbors.t() / sig + 1000 * cov_C);
  
  // const arma::colvec& mu = (1 / phi) * cov * (w * w_neighbors / sig +  1000 * vec_C);
  const arma::mat& cov = phi * arma::inv_sympd(w_neighbors * w_neighbors.t() / sig + cov_C);
  
  const arma::colvec& mu = (1 / phi) * cov * (w * w_neighbors / sig + vec_C);
  
  const arma::rowvec& gamma = mvrnormArma(mu.t(), cov);
  return gamma;
}

struct MCMC_update : public Worker {

  double& tmp_a, tmp_n_obs, tmp_tau, tmp_sig;
  arma::cube& tmp_covariances;
  arma::colvec& tmp_w;
  int& tmp_n_neighbors;
  arma::Mat<int>& tmp_W;
  arma::colvec& tmp_phi, tmp_beta;
  arma::mat& tmp_gamma;
  arma::colvec& tmp_y;
  arma::mat& tmp_samples;
  arma::cube& tmp_cov_inverse;
  arma::mat& tmp_cov_vector;
  arma::colvec& tmp_neigh_count;
  arma::Mat<int>& tmp_Wt; 
  arma::Mat<int>& tmp_Wt_ix;
  arma::colvec& tmp_cov_cond; // output matrix to write to

  MCMC_update(double& a, double& n_obs, arma::cube& covariances,
              arma::colvec& w, int& n_neighbors,
              arma::Mat<int>& W, double& tau, arma::colvec& phi,
              arma::mat& gamma,
              arma::colvec& y, arma::colvec& beta, double& sig, 
              arma::mat& samples, arma::cube& cov_inverse,
              arma::mat& cov_vector, arma::colvec& neigh_count,
              arma::Mat<int>& Wt, arma::Mat<int>& Wt_ix, arma::colvec& cov_cond)
    : tmp_a(a), tmp_n_obs(n_obs), tmp_covariances(covariances),
      tmp_w(w), tmp_n_neighbors(n_neighbors),
      tmp_W(W), tmp_tau(tau), tmp_phi(phi), tmp_gamma(gamma), tmp_y(y), tmp_beta(beta), tmp_sig(sig),
      tmp_samples(samples), tmp_cov_inverse(cov_inverse), tmp_cov_vector(cov_vector),
      tmp_neigh_count(neigh_count), tmp_Wt(Wt), tmp_Wt_ix(Wt_ix),
      tmp_cov_cond(cov_cond){}

  void operator()(std::size_t begin, std::size_t end) {

    for (std::size_t i=begin; i < end; i++) {
      
      int m_x = std::min<unsigned int>(tmp_n_neighbors, i);
      int m_u = tmp_neigh_count(i);
      
      // extract w
      arma::colvec w_neighbors_c(m_x, fill::zeros);
      arma::uvec idx = arma::conv_to<arma::uvec>::from(tmp_W.submat(i, 1, i, m_x));
      w_neighbors_c = tmp_w.elem(idx);
      
      arma::colvec gamma_u_si;
      arma::colvec b_u_si;
      arma::colvec phi_u_si;
      // // extract the information about future locations that use i as a neighbor
      if(m_u == 0){
        gamma_u_si.set_size(1);
        gamma_u_si(0) = 0;
        b_u_si.set_size(1);
        b_u_si(0) = 0; 
        phi_u_si.set_size(1);
        phi_u_si(0) = 1;
      } else {
        arma::uvec idu = arma::conv_to<arma::uvec>::from(tmp_Wt.submat(i, 0, i, m_u - 1));
        arma::uvec iduc = arma::conv_to<arma::uvec>::from(tmp_Wt_ix.submat(i, 0, i, m_u - 1));
        gamma_u_si.set_size(m_u);
        b_u_si.set_size(m_u);
        for (int u = 0; u < m_u; u++) {
          int ux = idu(u);
          int m_ux = std::min<unsigned int>(tmp_n_neighbors, ux);
          arma::colvec w_neighbors_uxc(m_ux, fill::zeros);
          arma::uvec idux = arma::conv_to<arma::uvec>::from(tmp_W.submat(ux, 1, ux, m_ux));
          w_neighbors_uxc = tmp_w.elem(idux);
          gamma_u_si(u) = tmp_gamma(ux, iduc(u));
          b_u_si(u) = tmp_w(ux) - dot(tmp_gamma.submat(ux, 0, ux, m_ux - 1), w_neighbors_uxc) + gamma_u_si(u) * tmp_w(i);
        }
        phi_u_si = tmp_phi.elem(idu);
      }
      
      
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
      double F = conditional_covariance(C, A, E(0, 0));

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
                  tmp_sig);

      // sample random effects w
      tmp_samples(i, 0) = w_rcpp_arm(tmp_samples.submat(i, 2, i, m_x + 1),
                  tmp_samples(i, 1),
                  w_neighbors_c,
                  tmp_y(i),
                  tmp_tau,
                  tmp_beta(i),
                  tmp_sig,
                  gamma_u_si,
                  b_u_si,
                  phi_u_si);
      // tmp_samples(i, 0) = tmp_y(i)-0.2;
      
      // write the covariance elements into the outputs
      tmp_cov_inverse.subcube(0, 0, i, m_x - 1, m_x - 1, i) = B;
      tmp_cov_vector.submat(0, i, m_x - 1, i) = A;
      tmp_cov_cond(i) = F;
      
    }

  }

};

// [[Rcpp::export]]
arma::cube mcmc_loop(double a, double n_obs, Rcpp::NumericVector& D,
                     arma::colvec w, int n_neighbors,
                     arma::colvec m_location,
                     arma::Mat<int> W, double tau,  arma::colvec phi, arma::mat gamma,
                     arma::colvec y, arma::colvec beta, double sig, double kappa,
                     double nu, arma::colvec tau_prior, arma::colvec sigma_prior,
                     double beta_prior, int n_covariates, arma::mat X,
                     int mcmc_samples, double nugget, 
                     double a_prior, double a_tuning, double nu_tuning, arma::colvec nu_prior,
                     double nugget_tuning, arma::colvec nugget_prior,
                     double a_nu_interval, arma::colvec neigh_count,
                     arma::Mat<int> Wt, arma::Mat<int> Wt_ix) {


  // Create cubes
  arma::cube samples_full(n_obs, n_covariates + 4, mcmc_samples, fill::zeros);
  arma::cube cube_D = Rcpp::as<arma::cube>(D);
  arma::cube cube_C_i(n_neighbors + 1, n_neighbors + 1, n_obs, fill::zeros);
  
  // Create the sliced samples
  arma::mat samples(n_obs, n_neighbors + n_covariates + 4, fill::zeros);
  arma::mat samples_light(n_obs, n_covariates + 4, fill::zeros);
  arma::cube cov_inv(n_neighbors, n_neighbors, n_obs, fill::zeros);
  arma::mat cov_vec(n_neighbors, n_obs, fill::zeros);
  arma::colvec cov_con(n_obs, fill::zeros);
  
  // create the workers
  bool pred_ic = false;
  double sig_unit = 1;
  covariance covariance(a, n_obs, cube_D, n_neighbors,
                        nugget, nu, kappa,
                        sig_unit, cube_C_i, pred_ic);
  
  // call the loop with parellelFor
  parallelFor(2, n_obs, covariance, n_obs / 10);
  
  // Create the sequential MCMC loop
  for (int p=0; p < mcmc_samples; p++) {
  
  // create the worker
  MCMC_update MCMC_update(a, n_obs, cube_C_i,
                          w, n_neighbors, W, tau, phi, gamma, y, beta, sig,
                          samples, cov_inv, cov_vec, neigh_count, Wt, Wt_ix,
                          cov_con);

  // call the loop with parellelFor
  parallelFor(2, n_obs, MCMC_update, n_obs / 10);

  // sample tau^2 the observational error
  samples(0, n_neighbors + 2) = tau_sample(tau_prior, y, samples, beta, n_obs);
  
  // sample sigma^2 the partial sill
  samples(1, n_neighbors + 2) = sig_sample(sigma_prior, W, samples, n_neighbors, n_obs);
  
  // sample fixed effects
  samples.submat(0, n_neighbors + 4, 0, n_neighbors + 3 + n_covariates) = 
    beta_sample(beta_prior, y, samples.col(0), X, samples(0, n_neighbors + 2), n_covariates);
  
  // compute XB
  samples.col(n_neighbors + 3) = 
      X * samples.submat(0, n_neighbors + 4, 0, n_neighbors + 3 + n_covariates).t();
    samples(0, n_neighbors + 3) = y(0);
    samples(1, n_neighbors + 3) = y(1);
    
    if(std::remainder(p + 2, a_nu_interval) == 0) {
      
      // sample alpha and range
      samples(2, n_neighbors + 2) = alpha_sample(a, a_prior, a_tuning, n_obs,
              cov_inv, cov_con, n_neighbors, samples.submat(0, 2, n_obs - 1, n_neighbors + 1),
              cov_vec, samples.col(1), cube_C_i, W, samples.col(0));

      cube_C_i = (samples(2, n_neighbors + 2) - n_obs - 1) / (a - n_obs - 1) * cube_C_i;
      a = samples(2, n_neighbors + 2);
       
      
      cube_C_i = range_sample(nu, nu_tuning, nu_prior, n_obs,
                              cov_inv, cov_con, n_neighbors, samples.submat(0, 2, n_obs - 1, n_neighbors + 1),
                              cov_vec, samples.col(1), cube_C_i, a, cube_D, nugget, kappa,
                              W, samples.col(0));

      samples(3, n_neighbors + 2) = cube_C_i(0,0,0);
        
      // cube_C_i = nugget_sample(nugget, nugget_prior, nugget_tuning,
      //         n_obs, a,
      //         cov_inv, cube_C_i, cov_con, n_neighbors,
      //         samples.submat(0, 2, n_obs - 1, n_neighbors + 1), cov_vec, samples.col(1),
      //         samples(1, n_neighbors + 2));
      // 
      // samples(4, n_neighbors + 2) = cube_C_i(0,0,0);
      
      
      nu = samples(3, n_neighbors + 2);
      // nugget = samples(4, n_neighbors + 2);
      
    } else {
      samples(2, n_neighbors + 2) = a;
      samples(3, n_neighbors + 2) = nu;
    }
       
  // Output to the posterior samples array
  samples_light.cols(0, 1) = samples.cols(0, 1);
  samples_light.cols(2, n_covariates + 3) = samples.cols(n_neighbors + 2, n_neighbors + n_covariates + 3);
  samples_full.slice(p) = samples_light;
  
  // Update the starting values;
  w = samples.col(0);
  tau = samples(0, n_neighbors + 2);
  phi = samples.col(1);
  sig = samples(1, n_neighbors + 2);
  beta = samples.col(n_neighbors + 3);
  gamma = samples.cols(2, n_neighbors + 1);
  
  }
  
  return samples_full;
}

// [[Rcpp::export]]
arma::mat posterior_hierarchical_prediction(int n_neighbors, double n_pred, Rcpp::NumericVector& D,
                                        Rcpp::NumericVector& w_post, double a, double kappa, 
                                        double nu, double nugget, arma::colvec sigma, arma::colvec tau_v,
                                        int mcmc_samples, double n_obs, arma::mat X, arma::mat beta) {
  
  
  // Create cubes and matrices
  arma::cube cube_D = Rcpp::as<arma::cube>(D);
  arma::cube mat_w = Rcpp::as<arma::cube>(w_post);
  arma::cube cube_C(n_neighbors + 1, n_neighbors + 1, n_pred, fill::zeros);
  arma::mat mat_Y_star(n_pred, mcmc_samples, fill::zeros);
  double sigma_d = 1;
  bool pred_ic = true;
  
  // compute the matern for m+1 X m+1 matrix
  covariance covariance(a, n_obs, cube_D, n_neighbors,
                        nugget, nu, kappa,
                        sigma_d, cube_C, pred_ic);

  parallelFor(0, n_pred, covariance, n_pred / 10);
  
  // obtain the predictive samples
  hierarchical_predictive hierarchical_predictive(n_neighbors, cube_C, mat_w, mat_Y_star,
                                          mcmc_samples, a, n_obs, sigma, tau_v, X, beta);
  
  parallelFor(0, n_pred, hierarchical_predictive, n_pred / 10);

  return mat_Y_star;
  
}


// [[Rcpp::export]]
arma::cube mv_range_sample(double nu_cur, double nu_tuning, arma::colvec nu_prior, double n_obs, 
                        arma::cube C_inv, arma::colvec C_cond, 
                        int n_neighbors, arma::mat g, arma::mat C_vec, arma::colvec phi,
                        arma::cube C_all, double a,
                        arma::cube D, arma::colvec nugget, arma::colvec kappa, 
                        arma::Mat<int> W, arma::colvec w_cur, arma::colvec sigma,
                        arma::Mat<int> Wnb) {
  
  // create the output
  double nu = 0;
  arma::cube C_new(n_neighbors + 1, n_neighbors + 1, n_obs, fill::zeros);
  
  // propose a new nu
  double nu_prop = std::abs(rnormArma(nu_cur, nu_tuning));
  arma::colvec range_prop(3, fill::zeros);
  range_prop(0) = nu_prop;
  range_prop(1) = nu_prop;
  range_prop(2) = nu_prop;
  
  // compute the prior under both current and proposed value
  double prior_prop = boost::math::gamma_p_derivative(nu_prior(0), nu_prop / nu_prior(1)) / nu_prior(1);
  double prior_cur = boost::math::gamma_p_derivative(nu_prior(0), nu_cur / nu_prior(1)) / nu_prior(1);
  
  // compute the covariance for the new range parameter
  arma::cube cube_C_i_prop(n_neighbors + 1, n_neighbors + 1, n_obs, fill::zeros); 
  bool pred_ic = false;
  
  mv_covariance mv_covariance(a, n_obs, Wnb, D, n_neighbors,
                        nugget, range_prop, kappa,
                        sigma, cube_C_i_prop, pred_ic);
  
  // call the loop with parellelFor
  parallelFor(2, n_obs, mv_covariance, n_obs / 10);
  
  
  // compute the posterior covariance for both ranges
  arma::cube cube_C_post_cur(n_neighbors + 1, n_neighbors +1, n_obs, fill::zeros);
  arma::cube cube_C_post_prop(n_neighbors + 1, n_neighbors +1, n_obs, fill::zeros);
  covariance_posterior covariance_posterior_cur(n_neighbors,
                                                C_all, W,
                                                C_all, cube_C_post_cur,
                                                w_cur);
  // call the loop with parellelFor
  parallelFor(2, n_obs, covariance_posterior_cur, n_obs / 10);
  
  covariance_posterior covariance_posterior_prop(n_neighbors,
                                                 cube_C_i_prop, W,
                                                 cube_C_i_prop, cube_C_post_prop,
                                                 w_cur);
  
  // call the loop with parellelFor
  parallelFor(2, n_obs, covariance_posterior_prop, n_obs / 10);
  
  
  // compute the marginal likelihood
  marginal_likelihood marginal_likelihood(C_all,
                                          cube_C_post_cur,
                                          a,
                                          n_neighbors,
                                          n_obs);
  
  parallelReduce(2, n_obs, marginal_likelihood, n_obs / 10);
  double marginal_cur = marginal_likelihood.tmp_marginal;
  
  // compute the marginal likelihood
  marginal_likelihood2 marginal_likelihood2(cube_C_i_prop,
                                            cube_C_post_prop,
                                            a,
                                            n_neighbors,
                                            n_obs);
  
  parallelReduce(2, n_obs, marginal_likelihood2, n_obs / 10);
  double marginal_prop = marginal_likelihood2.tmp_marginal;
  
  // // compute the likelihood under both current and proposed value
  // nu_likelihood nu_likelihood(a, n_neighbors, n_obs,
  //                             C_inv, g, C_vec, C_cond, phi, cube_C_i_prop);
  // 
  // parallelReduce(2, n_obs, nu_likelihood, n_obs / 10);
  // double likelihood = nu_likelihood.tmp_likelihood;
  // 
  // accept/reject the value
  double posterior_ratio = log(prior_prop) - log(prior_cur) + marginal_prop - marginal_cur;
  arma::colvec u = randu(1);
  if(log(u(0)) < posterior_ratio) {
    nu = nu_prop;
    C_new = cube_C_i_prop;
  } else {
    nu = nu_cur;
    C_new = C_all;
  }
  
  C_new(0,0,0) = nu;
  
  return C_new;
  
}

// [[Rcpp::export]]
arma::colvec mv_tau_sample(arma::colvec tau_prior, arma::colvec Y,
                  arma::mat samples, arma::colvec beta,
                  arma::colvec y_1_ix, arma::colvec y_2_ix, arma::mat Aw) {
  
  arma::colvec w = samples.col(0);
  arma::colvec tau(2, fill::zeros);
  
  // sample tau_1
  arma::uvec y1w1dx = arma::conv_to<arma::uvec>::from(y_1_ix);
  arma::uvec y1w2dx = y1w1dx + 1;
  double n_obs_1 = y1w1dx.n_elem;
  double shape_1 = tau_prior(0) + n_obs_1 / 2;
  arma::colvec Y_1 = Y.elem(y1w1dx);
  arma::colvec w_1 = w.elem(y1w1dx);
  arma::colvec w_2 = w.elem(y1w2dx);
  arma::colvec beta_1 = beta.elem(y1w1dx);                              
  arma::colvec mean_1 = Y_1 - Aw(0,0) * w_1 - Aw(0, 1) * w_2  - beta_1;
  double rate_1 = tau_prior(1) + dot(mean_1, mean_1) / 2;
  std::default_random_engine generator;
  std::gamma_distribution<double> tau1_distribution(shape_1, 1 / rate_1);
  tau(0) = 1 / tau1_distribution(generator);
  
  // sample tau_2
  arma::uvec y2w2dx = arma::conv_to<arma::uvec>::from(y_2_ix);
  arma::uvec y2w1dx = y2w2dx - 1;
  double n_obs_2 = y2w2dx.n_elem;
  double shape_2 = tau_prior(0) + n_obs_2 / 2;
  arma::colvec Y_2 = Y.elem(y2w2dx);
  w_2 = w.elem(y2w2dx);
  w_1 = w.elem(y2w1dx);
  arma::colvec beta_2 = beta.elem(y2w2dx);      
  arma::colvec mean_2 = Y_2 - Aw(1,0) * w_1 - Aw(1, 1) * w_2 - beta_2;
  double rate_2 = tau_prior(1) + dot(mean_2, mean_2) / 2;
  std::gamma_distribution<double> tau2_distribution(shape_2, 1 / rate_2);
  tau(1) = 1 / tau2_distribution(generator);
  
  return tau;
  
}

struct mv_MCMC_update : public Worker {
  
  double& tmp_a, tmp_n_obs, tmp_sig;
  arma::colvec tmp_tau;
  arma::cube& tmp_covariances;
  arma::colvec& tmp_w;
  int& tmp_n_neighbors;
  arma::Mat<int>& tmp_W;
  arma::mat& tmp_Aw; 
  arma::colvec& tmp_phi, tmp_beta;
  arma::colvec& tmp_y;
  arma::mat& tmp_gamma;
  arma::mat& tmp_samples;
  arma::cube& tmp_cov_inverse;
  arma::mat& tmp_cov_vector;
  arma::Mat<int>& tmp_Wnb;
  arma::colvec& tmp_neigh_count;
  arma::Mat<int>& tmp_Wt; 
  arma::Mat<int>& tmp_Yt; 
  arma::Mat<int>& tmp_Wt_ix;
  arma::colvec& tmp_cov_cond; // output matrix to write to
  
  mv_MCMC_update(double& a, double& n_obs, arma::cube& covariances,
              arma::colvec& w, int& n_neighbors, arma::mat& gamma,
              arma::Mat<int>& W, arma::colvec& tau, arma::colvec& phi,
              arma::colvec& y, arma::colvec& beta, double& sig, 
              arma::mat& samples, arma::cube& cov_inverse,
              arma::mat& cov_vector, arma::colvec& cov_cond, arma::Mat<int>& Wnb,
              arma::colvec& neigh_count, arma::Mat<int>& Wt, 
              arma::Mat<int>& Wt_ix, arma::Mat<int>& Yt, arma::mat& Aw)
    : tmp_a(a), tmp_n_obs(n_obs), tmp_covariances(covariances),
      tmp_w(w), tmp_n_neighbors(n_neighbors), tmp_gamma(gamma),
      tmp_W(W), tmp_tau(tau), tmp_phi(phi), tmp_y(y), tmp_beta(beta), tmp_sig(sig),
      tmp_samples(samples), tmp_cov_inverse(cov_inverse), tmp_cov_vector(cov_vector),
      tmp_cov_cond(cov_cond), tmp_Wnb(Wnb), tmp_neigh_count(neigh_count),
      tmp_Wt(Wt), tmp_Wt_ix(Wt_ix), tmp_Yt(Yt), tmp_Aw(Aw){}
  
  void operator()(std::size_t begin, std::size_t end) {
    
    for (std::size_t i=begin; i < end; i++) {
      
      int m_x = std::min<unsigned int>(tmp_n_neighbors, i);
      double source = tmp_Wnb(i, 0) - 1;
      double other = 0;
      if (source == 0) {
        other = 1;
      }
      int m_u = tmp_neigh_count(i);
      
      // extract w
      arma::colvec w_neighbors_c(m_x, fill::zeros);
      arma::uvec idx = arma::conv_to<arma::uvec>::from(tmp_W.submat(i, 1, i, m_x));
      w_neighbors_c = tmp_w.elem(idx);

      arma::colvec gamma_u_si;
      arma::colvec b_u_si;
      arma::colvec phi_u_si;
      // // extract the information about future locations that use i as a neighbor
      if(m_u == 0){
        gamma_u_si.set_size(1);
        gamma_u_si(0) = 0;
        b_u_si.set_size(1);
        b_u_si(0) = 0;
        phi_u_si.set_size(1);
        phi_u_si(0) = 1;
      } else {
        arma::uvec idu = arma::conv_to<arma::uvec>::from(tmp_Wt.submat(i, 0, i, m_u - 1));
        arma::uvec iduc = arma::conv_to<arma::uvec>::from(tmp_Wt_ix.submat(i, 0, i, m_u - 1));
        gamma_u_si.set_size(m_u);
        b_u_si.set_size(m_u);
        for (int u = 0; u < m_u; u++) {
          int ux = idu(u);
          int m_ux = std::min<unsigned int>(tmp_n_neighbors, ux);
          arma::colvec w_neighbors_uxc(m_ux, fill::zeros);
          arma::uvec idux = arma::conv_to<arma::uvec>::from(tmp_W.submat(ux, 1, ux, m_ux));
          w_neighbors_uxc = tmp_w.elem(idux);
          gamma_u_si(u) = tmp_gamma(ux, iduc(u));
          b_u_si(u) = tmp_w(ux) - dot(tmp_gamma.submat(ux, 0, ux, m_ux - 1), w_neighbors_uxc) + gamma_u_si(u) * tmp_w(i);
        }
        phi_u_si = tmp_phi.elem(idu);
      }

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
      double F = conditional_covariance(C, A, E(0, 0));

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
                  tmp_sig);

      // sample random effects w
      tmp_samples(i, 0) = mv_w_rcpp_arm(tmp_samples.submat(i, 2, i, m_x + 1),
                  tmp_samples(i, 1),
                  w_neighbors_c,
                  tmp_y,
                  tmp_tau,
                  tmp_beta(i),
                  gamma_u_si,
                  b_u_si,
                  phi_u_si,
                  tmp_Yt.row(i),
                  tmp_Aw.col(source),
                  tmp_Aw.col(other),
                  tmp_w(i - source + other),
                  source,
                  other,
                  i);

      // write the covariance elements into the outputs
      tmp_cov_inverse.subcube(0, 0, i, m_x - 1, m_x - 1, i) = B;
      tmp_cov_vector.submat(0, i, m_x - 1, i) = A;
      tmp_cov_cond(i) = F;
      
    }
    
  }
  
};

// [[Rcpp::export]]
arma::mat mv_mcmc_loop(double a, double n_obs, Rcpp::NumericVector& D,
                     arma::colvec w, int n_neighbors,
                     arma::Mat<int> W, arma::colvec tau,  arma::colvec phi, arma::mat gamma,
                     arma::colvec y, arma::colvec beta, arma::colvec sig, arma::colvec kappa,
                     double nu, arma::colvec tau_prior,
                     double beta_prior, int n_covariates, arma::mat X, arma::mat Aw,
                     int mcmc_samples, arma::colvec nugget, 
                     double a_prior, double a_tuning, double nu_tuning, arma::colvec nu_prior,
                     double a_nu_interval, arma::colvec y1_ix, arma::colvec y2_ix,
                     arma::Mat<int> Wnb, arma::colvec neigh_count,
                     arma::Mat<int> Wt, arma::Mat<int> Wt_ix, arma::Mat<int> Yt) {
  
  // extract parameters
  int m2 = 2 * n_neighbors;
  arma::colvec range_trip(3, fill::zeros);
  range_trip(0) = nu;
  range_trip(2) = nu;
  range_trip(1) = nu;
  
  // Create cubes
  arma::mat samples_full(n_obs + n_covariates + 4, mcmc_samples, fill::zeros);
  arma::cube cube_D = Rcpp::as<arma::cube>(D);
  arma::cube cube_C_i(m2 + 1, m2 + 1, n_obs, fill::zeros);
  
  // Create the sliced samples
  arma::mat samples(n_obs, m2 + n_covariates + 4, fill::zeros);
  arma::colvec samples_light(n_obs + n_covariates + 4, fill::zeros);
  arma::cube cov_inv(m2, m2, n_obs, fill::zeros);
  arma::mat cov_vec(m2, n_obs, fill::zeros);
  arma::colvec cov_con(n_obs, fill::zeros);
  
  // create the workers
  bool pred_ic = false;
  mv_covariance mv_covariance(a, n_obs, Wnb, cube_D, n_neighbors,
                        nugget, range_trip, kappa,
                        sig, cube_C_i, pred_ic);
  
  // call the loop with parellelFor
  parallelFor(2, n_obs, mv_covariance, n_obs / 10);
  
  // Create the sequential MCMC loop
  double sig_unit = 1;
  for (int p=0; p < mcmc_samples; p++) {
    
    // create the worker
    mv_MCMC_update mv_MCMC_update(a, n_obs, cube_C_i,
                            w, m2, gamma, W, tau, phi, y, beta, sig_unit,
                            samples, cov_inv, cov_vec, cov_con, Wnb,
                            neigh_count, Wt, Wt_ix, Yt, Aw);

    // call the loop with parellelFor
    parallelFor(2, n_obs, mv_MCMC_update, n_obs / 10);
    
    // sample tau^2 the observational error
    samples.submat(0, m2 + 2, 1, m2 + 2) = mv_tau_sample(tau_prior, y, samples, beta, y1_ix, y2_ix, Aw);
    //samples.submat(0, m2 + 2, 1, m2 + 2) = tau;
    
    // sample sigma^2 the partial sill
    // samples.submat(2, m2 + 2, 4, m2 + 2) = mv_sig_sample(sigma_prior, W, samples, m2, n_obs);
    samples.submat(2, m2 + 2, 4, m2 + 2) = sig;
    
    // // sample fixed effects
    // samples.submat(0, m2 + 4, 0, m2 + 3 + n_covariates) = 
    //   beta_sample(beta_prior, y, samples.col(0), X, samples(0, m2 + 2), n_covariates);
    // 
    // // compute XB
    // samples.col(m2 + 3) = 
    //   X * samples.submat(0, m2 + 4, 0, m2 + 3 + n_covariates).t();
    // samples(0, m2 + 3) = y(0);
    // samples(1, m2 + 3) = y(1);
    
    if(std::remainder(p + 2, a_nu_interval) == 0) {
      
      // sample alpha and range
      samples(5, m2 + 2) = alpha_sample(a, a_prior, a_tuning, n_obs,
              cov_inv, cov_con, m2, samples.submat(0, 2, n_obs - 1, m2 + 1),
              cov_vec, samples.col(1), cube_C_i, W, samples.col(0));
      
      cube_C_i = (samples(5, m2 + 2) - n_obs - 1) / (a - n_obs - 1) * cube_C_i;
      a = samples(5, m2 + 2);
      
      
      cube_C_i = mv_range_sample(nu, nu_tuning, nu_prior, n_obs,
                              cov_inv, cov_con, m2, samples.submat(0, 2, n_obs - 1, m2 + 1),
                              cov_vec, samples.col(1), cube_C_i, a, cube_D, nugget, kappa,
                              W, samples.col(0), sig, Wnb);
      
      samples(6, m2 + 2) = cube_C_i(0,0,0);
      
      // cube_C_i = nugget_sample(nugget, nugget_prior, nugget_tuning,
      //         n_obs, a,
      //         cov_inv, cube_C_i, cov_con, n_neighbors,
      //         samples.submat(0, 2, n_obs - 1, n_neighbors + 1), cov_vec, samples.col(1),
      //         samples(1, n_neighbors + 2));
      // 
      // samples(7, n_neighbors + 2) = cube_C_i(0,0,0);
      
      
      nu = samples(6, m2 + 2);
      range_trip(0) = nu;
      range_trip(1) = nu;
      range_trip(2) = nu;
      // nugget = samples(4, n_neighbors + 2);
      
    } else {
      samples(5, m2 + 2) = a;
      samples(6, m2 + 2) = nu;
    }
    
    // Output to the posterior samples array
    samples_light.subvec(0, n_obs - 1) = samples.col(0);
    samples_light.subvec(n_obs, n_covariates + n_obs - 1) = samples.submat(0, m2 + 4, 0, m2 + 3 + n_covariates).t();
    samples_light.subvec(n_covariates + n_obs, n_covariates + n_obs + 1) = samples.submat(0, m2 + 2, 1, m2 + 2);
    samples_light(n_covariates + n_obs + 2) = samples(5, m2 + 2);
    samples_light(n_covariates + n_obs + 3) = samples(6, m2 + 2);
    samples_full.col(p) = samples_light;
    
    // Update the starting values;
    w = samples.col(0);
    tau = samples.submat(0, m2 + 2, 1, m2 + 2);
    phi = samples.col(1);
    sig = samples.submat(2, m2 + 2, 4, m2 + 2);
    beta = samples.col(m2 + 3);
    gamma = samples.cols(2, m2 + 1);
  }
  
  return samples_full;
}

// [[Rcpp::export]]
arma::mat mv_posterior_hierarchical_prediction(int n_neighbors, double n_pred, Rcpp::NumericVector& D,
                                            Rcpp::NumericVector& w_post, double a, arma::colvec kappa, 
                                            double nu, arma::colvec nugget, arma::colvec sigma, arma::mat tau_v,
                                            int mcmc_samples, double n_obs, arma::mat X, arma::mat beta,
                                            arma::Mat<int> Wnb, arma::mat Aw) {
  
  // extract values
  int m2= 2 * n_neighbors;
  arma::colvec range_triple(3, fill::zeros);
  range_triple(0) = nu;
  range_triple(2) = nu;
  range_triple(1) = nu;
  
  // Create cubes and matrices
  arma::cube cube_D = Rcpp::as<arma::cube>(D);
  arma::cube mat_w = Rcpp::as<arma::cube>(w_post);
  arma::cube cube_C(m2 + 1, m2 + 1, n_pred, fill::zeros);
  arma::mat mat_Y_star(n_pred, mcmc_samples, fill::zeros);
  arma::mat mat_w_star(n_pred, mcmc_samples, fill::zeros);
  bool pred_ic = true;
  //arma::rowvec sig = sigma.submat(0, 0, 0, 2); // pick only one sigma for now
  
  // compute the matern for m+1 X m+1 matrix
  mv_covariance mv_covariance(a, n_obs, Wnb, cube_D, m2,
                        nugget, range_triple, kappa,
                        sigma, cube_C, pred_ic);
  
  parallelFor(0, n_pred, mv_covariance, n_pred / 10);
  
  // sample w*
  mv_hierarchical_predictive mv_hierarchical_predictive(m2, cube_C, mat_w, mat_w_star,
                                                  mcmc_samples, a, n_obs, tau_v, X, beta,
                                                  Wnb, Aw);
  
  parallelFor(0, n_pred, mv_hierarchical_predictive, n_pred / 10);
  
  // sample y*
  // mv_hierarchical_predictive_final_step
  //   mv_hierarchical_predictive_final_step(mat_w_star, mat_Y_star, mcmc_samples,
  //                                         tau_v, X, beta,
  //                                         Wnb, Aw);
  // 
  // parallelFor(0, n_pred / 2 - 1, mv_hierarchical_predictive_final_step, n_pred / 10);

  return mat_w_star;
  
}