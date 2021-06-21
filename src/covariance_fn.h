#ifndef COVARIANCE_FN_H
#define COVARIANCE_FN_H

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

  arma::mat matern_cov(arma::mat distance, double kappa, double phi);
  
  double conditional_covariance(arma::mat& C, arma::colvec vec_C, double C_i);
  
  struct covariance : public Worker {
    
    double& tmp_a, tmp_n_obs, tmp_tau, tmp_sig, tmp_phi, tmp_kappa;
    arma::cube& tmp_vec_D;
    int& tmp_n_neighbors;
    arma::cube& tmp_covariances;       // output cube to write to
    
    covariance(double& a, double& n_obs, arma::cube& vec_D,
               int& n_neighbors,
               double& tau, double& phi, double& kappa,
               double& sig, arma::cube& covariances)
      : tmp_a(a), tmp_n_obs(n_obs), tmp_vec_D(vec_D),
        tmp_n_neighbors(n_neighbors),
        tmp_tau(tau), tmp_phi(phi), tmp_kappa(kappa), tmp_sig(sig),
        tmp_covariances(covariances) {}
    
    void operator()(std::size_t begin, std::size_t end) {
      
      for (std::size_t i=begin; i < end; i++) {
        
        int m_x = std::min<unsigned int>(tmp_n_neighbors, i);
        
        tmp_covariances.subcube(0, 0, i, m_x, m_x, i) =
          (tmp_a - tmp_n_obs - 1) * tmp_sig *
          matern_cov(tmp_vec_D.subcube(0, 0, i, m_x, m_x, i),
                     tmp_kappa,
                     tmp_phi) +
                       tmp_tau * arma::eye(m_x + 1, m_x + 1);
        
      }
    }
  };
  
  struct covariance_posterior : public Worker {
    
    arma::cube& tmp_covariances;
    int& tmp_n_neighbors;
    arma::Mat<int>& tmp_W;
    arma::cube& tmp_y;
    arma::cube& tmp_covariances_y;       // output cube to write to
    
    covariance_posterior(int& n_neighbors, arma::cube& covariances,
                         arma::Mat<int>& W, arma::cube& y, arma::cube& covariances_y)
      : tmp_n_neighbors(n_neighbors),
        tmp_covariances(covariances),
        tmp_W(W),
        tmp_y(y),
        tmp_covariances_y(covariances_y){}
    
    void operator()(std::size_t begin, std::size_t end) {
      
      for (std::size_t i=begin; i < end; i++) {
        
        int m_x = std::min<unsigned int>(tmp_n_neighbors, i);
        
        // the code to have the YY^T part in here is not working.
        // arma::colvec y_neighbors_c = tmp_y;
        
        // arma::uvec idx = arma::conv_to<arma::uvec>::from(tmp_W.submat(i, 0, i, m_x));
        // y_neighbors_c = tmp_y.elem(idx);
        // arma::mat observations = y_neighbors_c * y_neighbors_c.t();
        
        arma::mat cov_prior = tmp_covariances.subcube(0, 0, i, m_x, m_x, i);
        arma::mat y_post = tmp_y.subcube(0, 0, i, m_x, m_x, i);
        tmp_covariances_y.subcube(0, 0, i, m_x, m_x, i) = cov_prior + y_post;
        
      }
    }
  };

#endif /* COVARIANCE_FN_H */