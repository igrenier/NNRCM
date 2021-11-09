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
  
  const arma::rowvec mvrnormArma(const arma::rowvec mean, const arma::mat variance);
  
  const double rnormArma(const double mean, const double sd);
  
  const double mvnpdf(const arma::colvec x, const arma::colvec mean, const arma::mat sigma,
                      const int n);
  
  struct covariance : public Worker {
    
    double& tmp_a, tmp_n_obs, tmp_tau, tmp_sig, tmp_phi, tmp_kappa;
    bool& tmp_prediction;
    arma::cube& tmp_vec_D;
    int& tmp_n_neighbors;
    arma::cube& tmp_covariances;       // output cube to write to
    
    covariance(double& a, double& n_obs, arma::cube& vec_D,
               int& n_neighbors,
               double& tau, double& phi, double& kappa,
               double& sig, arma::cube& covariances, bool& pred_ict)
      : tmp_a(a), tmp_n_obs(n_obs), tmp_vec_D(vec_D),
        tmp_n_neighbors(n_neighbors),
        tmp_tau(tau), tmp_phi(phi), tmp_kappa(kappa), tmp_sig(sig),
        tmp_covariances(covariances),
        tmp_prediction(pred_ict){}
    
    void operator()(std::size_t begin, std::size_t end) {
      
      for (std::size_t i=begin; i < end; i++) {
        
        int m_x = tmp_n_neighbors;
        
      if (!tmp_prediction){
        m_x = std::min<unsigned int>(tmp_n_neighbors, i);
      }
      // this was changed from tmp_a - tmp_n_obs - 1)
        tmp_covariances.subcube(0, 0, i, m_x, m_x, i) =
          (tmp_a - tmp_n_obs - 1) * tmp_sig * (
          matern_cov(tmp_vec_D.subcube(0, 0, i, m_x, m_x, i),
                     tmp_kappa,
                     tmp_phi) +
                       tmp_tau * arma::eye(m_x + 1, m_x + 1));
        
      }
    }
  };
  
  struct covariance_posterior : public Worker {
    
    arma::cube& tmp_covariances;
    int& tmp_n_neighbors;
    arma::Mat<int>& tmp_W;
    arma::cube& tmp_y;
    arma::colvec& tmp_data;
    arma::cube& tmp_covariances_y;       // output cube to write to
    
    covariance_posterior(int& n_neighbors, arma::cube& covariances,
                         arma::Mat<int>& W, arma::cube& y, arma::cube& covariances_y,
                         arma::colvec& data)
      : tmp_n_neighbors(n_neighbors),
        tmp_covariances(covariances),
        tmp_W(W),
        tmp_y(y),
        tmp_covariances_y(covariances_y),
        tmp_data(data){}
    
    void operator()(std::size_t begin, std::size_t end) {
      
      for (std::size_t i=begin; i < end; i++) {
        
        int m_x = std::min<unsigned int>(tmp_n_neighbors, i);

        // the code to have the YY^T part in here is not working.
        arma::colvec y_neighbors_c(m_x + 1, fill::zeros);

        arma::uvec idx = arma::conv_to<arma::uvec>::from(tmp_W.submat(i, 0, i, m_x));
        y_neighbors_c = tmp_data.elem(idx);
        arma::mat y_post = y_neighbors_c * y_neighbors_c.t();

        arma::mat cov_prior = tmp_covariances.subcube(0, 0, i, m_x, m_x, i);
        //arma::mat y_post = tmp_y.subcube(0, 0, i, m_x, m_x, i);
        tmp_covariances_y.subcube(0, 0, i, m_x, m_x, i) = cov_prior + y_post;
        
      }
    }
  };
  
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
        
        // // extract the m X 1 observations to neighbors covarianc vector
        // arma::colvec C_neighbors_c = C_i_neighbors.submat(1, 0, m_x, 0);
        // arma::colvec C_neighbors_posterior_c = C_i_neighbors_y.submat(1, 0, m_x, 0);
        // 
        // // compute conditional of C
        // double det_cond = conditional_covariance(C_neighbors,
        //                                          C_neighbors_c,
        //                                          C_i_neighbors(0, 0));
        // 
        // // compute conditional of C-posterior
        // double det_posterior_cond = conditional_covariance(C_neighbors_posterior,
        //                                                    C_neighbors_posterior_c,
        //                                                    C_i_neighbors_y(0, 0));
        
        // compute the first expression determinant
        double det_C = arma::det(C_i_neighbors.submat(0, 0, m_x, m_x));
        double det_C_post = arma::det(C_i_neighbors_y.submat(0, 0, m_x, m_x));
        // double det_sqrt = 1 / 2 * (log(arma::det(C_i_neighbors(0, 0))) - log(arma::det(C_i_neighbors_y(0, 0))));
        //double det_sqrt = 1 / 2 * (log(C_i_neighbors(0, 0)) - log(C_i_neighbors_y(0, 0)));
        //double det_sqrt = 1 / 2 * (log(det_C) - log(det_C_post));
        
        // compute the second expression
        double det_C_n = arma::det(C_neighbors);
        double det_C_n_post = arma::det(C_neighbors_posterior);
        
        // copmute marginal terms
        // tmp_marginal += (tmp_a - tmp_n_obs + m_x) / 2 * log(det_cond) - 
        //   (tmp_a - tmp_n_obs + m_x + 1) / 2 * log(det_posterior_cond) + 
        //   det_sqrt;
        
        tmp_marginal += (tmp_a - tmp_n_obs + m_x + 1) / 2 * log(det_C) -
          (tmp_a - tmp_n_obs + m_x + 2) / 2 * log(det_C_post) -
          (tmp_a - tmp_n_obs + m_x) / 2 * log(det_C_n) +
          (tmp_a - tmp_n_obs + m_x + 1) / 2 * log(det_C_n_post);
        
      }
    } 
    
    // join my value with that of another Sum
    void join(const marginal_likelihood& rhs) {
      tmp_marginal += rhs.tmp_marginal;
    }
  }; 
  
  struct marginal_likelihood2 : public Worker {
    
    arma::cube& tmp_covariances;
    arma::cube& tmp_covariances_posterior;
    double& tmp_a, tmp_n_obs;
    int& tmp_n_neighbors;
    double tmp_marginal;       // output of the loop
    
    // constructors  
    marginal_likelihood2(arma::cube& covariances, arma::cube& covariances_posterior,
                        double& a, int& n_neighbors, double& n_obs) 
      : tmp_covariances(covariances),
        tmp_covariances_posterior(covariances_posterior),
        tmp_a(a),
        tmp_n_neighbors(n_neighbors),
        tmp_n_obs(n_obs),
        tmp_marginal(0){}
    
    marginal_likelihood2(marginal_likelihood2& marg, Split) : 
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
        
        // // extract the m X 1 observations to neighbors covarianc vector
        // arma::colvec C_neighbors_c = C_i_neighbors.submat(1, 0, m_x, 0);
        // arma::colvec C_neighbors_posterior_c = C_i_neighbors_y.submat(1, 0, m_x, 0);
        // 
        // // compute conditional of C
        // double det_cond = conditional_covariance(C_neighbors,
        //                                          C_neighbors_c,
        //                                          C_i_neighbors(0, 0));
        // 
        // // compute conditional of C-posterior
        // double det_posterior_cond = conditional_covariance(C_neighbors_posterior,
        //                                                    C_neighbors_posterior_c,
        //                                                    C_i_neighbors_y(0, 0));
        
        // compute the first expression determinant
        double det_C = arma::det(C_i_neighbors.submat(0, 0, m_x, m_x));
        double det_C_post = arma::det(C_i_neighbors_y.submat(0, 0, m_x, m_x));
        // double det_sqrt = 1 / 2 * (log(arma::det(C_i_neighbors(0, 0))) - log(arma::det(C_i_neighbors_y(0, 0))));
        //double det_sqrt = 1 / 2 * (log(C_i_neighbors(0, 0)) - log(C_i_neighbors_y(0, 0)));
        //double det_sqrt = 1 / 2 * (log(det_C) - log(det_C_post));
        
        // compute the second expression
        double det_C_n = arma::det(C_neighbors);
        double det_C_n_post = arma::det(C_neighbors_posterior);
        
        // copmute marginal terms
        // tmp_marginal += (tmp_a - tmp_n_obs + m_x) / 2 * log(det_cond) - 
        //   (tmp_a - tmp_n_obs + m_x + 1) / 2 * log(det_posterior_cond) + 
        //   det_sqrt;
        
        tmp_marginal += (tmp_a - tmp_n_obs + m_x + 1) / 2 * log(det_C) -
          (tmp_a - tmp_n_obs + m_x + 2) / 2 * log(det_C_post) -
          (tmp_a - tmp_n_obs + m_x) / 2 * log(det_C_n) +
          (tmp_a - tmp_n_obs + m_x + 1) / 2 * log(det_C_n_post);
        
      }
    } 
    
    // join my value with that of another Sum
    void join(const marginal_likelihood2& rhs) {
      tmp_marginal += rhs.tmp_marginal;
    }
  }; 
  
  struct mv_covariance : public Worker {
    
    double& tmp_a, tmp_n_obs;
    arma::Mat<int> tmp_Wnb;
    arma::colvec& tmp_tau, tmp_sig, tmp_phi, tmp_kappa;
    bool& tmp_prediction;
    arma::cube& tmp_vec_D;
    int& tmp_n_neighbors;
    arma::cube& tmp_covariances;       // output cube to write to
    
    mv_covariance(double& a, double& n_obs, arma::Mat<int> Wnb, ::cube& vec_D,
               int& n_neighbors,
               arma::colvec& tau, arma::colvec& phi, arma::colvec& kappa,
               arma::colvec& sig, arma::cube& covariances, bool& pred_ict)
      : tmp_a(a), tmp_n_obs(n_obs), tmp_Wnb(Wnb), tmp_vec_D(vec_D),
        tmp_n_neighbors(n_neighbors),
        tmp_tau(tau), tmp_phi(phi), tmp_kappa(kappa), tmp_sig(sig),
        tmp_covariances(covariances),
        tmp_prediction(pred_ict){}
    
    void operator()(std::size_t begin, std::size_t end) {
      
      for (std::size_t i=begin; i < end; i++) {
        
        int m_x1 = tmp_Wnb(i, 1);
        int m_x2 = tmp_Wnb(i, 2);
        int m_x = m_x1 + m_x2;

        // extract which source observation is from
        double source = tmp_Wnb(i, 0) - 1;
        double other = 0;
        if (source == 0) {
          other = 1;
        }
        // Component 2: cross covariance
        arma::mat full_cross_matrix(m_x + 1, m_x + 1, fill::zeros);
        // check if m_x1 or m_x2 is 0.
        if(m_x1 == 0){
          
          // Component 2: cross covariance
          full_cross_matrix = (tmp_a - tmp_n_obs - 1) * tmp_sig(2) * (
            matern_cov(tmp_vec_D.subcube(0, 0, i, m_x, m_x, i),
                       tmp_kappa(2),
                       tmp_phi(2)));
          tmp_covariances.subcube(0, m_x1 + 1, i, m_x1 + 1, m_x, i) =
            full_cross_matrix.submat(0, m_x1 + 1, m_x1 + 1, m_x);
          tmp_covariances.subcube(m_x1 + 1, 0, i, m_x, m_x1 + 1, i) =
            full_cross_matrix.submat(m_x1 + 1, 0, m_x, m_x1 + 1);
          
          // Component 3: matrix C(N_s(s), N_2(s))
          tmp_covariances.subcube(m_x1 + 1, m_x1 + 1, i, m_x, m_x, i) =
            (tmp_a - tmp_n_obs - 1) * tmp_sig(other) * (
                matern_cov(tmp_vec_D.subcube(m_x1 + 1, m_x1 + 1, i, m_x, m_x, i),
                           tmp_kappa(other),
                           tmp_phi(other)) +
                             tmp_tau(other) * arma::eye(m_x2, m_x2));
        } else if (m_x2 == 0) {
          // Component 1: matrix C(s, N_1(s), s, N_1(s))
          tmp_covariances.subcube(0, 0, i, m_x1, m_x1, i) =
            (tmp_a - tmp_n_obs - 1) * tmp_sig(source) * (
                matern_cov(tmp_vec_D.subcube(0, 0, i, m_x1, m_x1, i),
                           tmp_kappa(source),
                           tmp_phi(source)) +
                             tmp_tau(source) * arma::eye(m_x1 + 1, m_x1 + 1));
          
        } else {
          
          // Component 1: matrix C(s, N_1(s), s, N_1(s))
          tmp_covariances.subcube(0, 0, i, m_x1, m_x1, i) =
            (tmp_a - tmp_n_obs - 1) * tmp_sig(source) * (
                matern_cov(tmp_vec_D.subcube(0, 0, i, m_x1, m_x1, i),
                           tmp_kappa(source),
                           tmp_phi(source)) +
                             tmp_tau(source) * arma::eye(m_x1 + 1, m_x1 + 1));
          
          // Component 2: cross covariance
          full_cross_matrix = (tmp_a - tmp_n_obs - 1) * tmp_sig(2) * (
            matern_cov(tmp_vec_D.subcube(0, 0, i, m_x, m_x, i),
                       tmp_kappa(2),
                       tmp_phi(2)));
          tmp_covariances.subcube(0, m_x1 + 1, i, m_x1 + 1, m_x, i) =
            full_cross_matrix.submat(0, m_x1 + 1, m_x1 + 1, m_x);
          tmp_covariances.subcube(m_x1 + 1, 0, i, m_x, m_x1 + 1, i) =
            full_cross_matrix.submat(m_x1 + 1, 0, m_x, m_x1 + 1);
          
          // Component 3: matrix C(N_s(s), N_2(s))
          tmp_covariances.subcube(m_x1 + 1, m_x1 + 1, i, m_x, m_x, i) =
            (tmp_a - tmp_n_obs - 1) * tmp_sig(other) * (
                matern_cov(tmp_vec_D.subcube(m_x1 + 1, m_x1 + 1, i, m_x, m_x, i),
                           tmp_kappa(other),
                           tmp_phi(other)) +
                             tmp_tau(other) * arma::eye(m_x2, m_x2));
          
        }
      }
    }
  };
  
  struct mv_coregionalization : public Worker {
    
    double& tmp_a, tmp_n_obs;
    arma::Mat<int> tmp_Wnb;
    arma::colvec& tmp_tau, tmp_phi, tmp_kappa;
    bool& tmp_prediction;
    arma::colvec& tmp_A;
    arma::cube& tmp_vec_D;
    int& tmp_n_neighbors;
    arma::cube& tmp_covariances;       // output cube to write to
    
    mv_coregionalization(double& a, double& n_obs, arma::Mat<int> Wnb, ::cube& vec_D,
                  int& n_neighbors,
                  arma::colvec& tau, arma::colvec& phi, arma::colvec& kappa,
                  arma::cube& covariances, bool& pred_ict,
                  arma::colvec& A)
      : tmp_a(a), tmp_n_obs(n_obs), tmp_Wnb(Wnb), tmp_vec_D(vec_D),
        tmp_n_neighbors(n_neighbors),
        tmp_tau(tau), tmp_phi(phi), tmp_kappa(kappa),
        tmp_covariances(covariances),
        tmp_prediction(pred_ict), tmp_A(A){}
    
    void operator()(std::size_t begin, std::size_t end) {
      
      for (std::size_t i=begin; i < end; i++) {
        
        int m_x1 = tmp_Wnb(i, 1);
        int m_x2 = tmp_Wnb(i, 2);
        int m_x = m_x1 + m_x2;
        
        // extract which source observation is from
        double source = tmp_Wnb(i, 0) - 1;
        double other = 0;
        if (source == 0) {
          other = 1;
        }
        
        // Component 1: matrix C(s, N_1(s), s, N_1(s))
        tmp_covariances.subcube(0, 0, i, m_x1, m_x1, i) =
          (tmp_a - tmp_n_obs - 1) * tmp_A(source) * tmp_A(source) * (
              matern_cov(tmp_vec_D.subcube(0, 0, i, m_x1, m_x1, i),
                         tmp_kappa(source),
                         tmp_phi(source)) +
                           tmp_tau(source) * arma::eye(m_x1 + 1, m_x1 + 1));
        
        // Component 2: cross covariance
        arma::mat full_cross_matrix(m_x + 1, m_x + 1, fill::zeros);
        full_cross_matrix = (tmp_a - tmp_n_obs - 1) * tmp_A(2) * tmp_A(0) * (
          matern_cov(tmp_vec_D.subcube(0, 0, i, m_x, m_x, i),
                     tmp_kappa(0),
                     tmp_phi(0)));
        tmp_covariances.subcube(0, m_x1 + 1, i, m_x1 + 1, m_x, i) =
          full_cross_matrix.submat(0, m_x1 + 1, m_x1 + 1, m_x);
        tmp_covariances.subcube(m_x1 + 1, 0, i, m_x, m_x1 + 1, i) =
          full_cross_matrix.submat(m_x1 + 1, 0, m_x, m_x1 + 1);
        
        // Component 3: matrix C(N_s(s), N_2(s))
        tmp_covariances.subcube(m_x1 + 1, m_x1 + 1, i, m_x, m_x, i) =
          (tmp_a - tmp_n_obs - 1) * tmp_A(other) * tmp_A(other) * (
              matern_cov(tmp_vec_D.subcube(m_x1 + 1, m_x1 + 1, i, m_x, m_x, i),
                         tmp_kappa(other),
                         tmp_phi(other)) +
                           tmp_tau(other) * arma::eye(m_x2, m_x2));
        
        // Add a^2_{21} * C_1(s, s') to a^2_{22} C_2(s, s')
        if(source == 0){
          tmp_covariances.subcube(m_x1 + 1, m_x1 + 1, i, m_x, m_x, i) +=
            (tmp_a - tmp_n_obs - 1) * tmp_A(2) * tmp_A(2) * (
                matern_cov(tmp_vec_D.subcube(m_x1 + 1, m_x1 + 1, i, m_x, m_x, i),
                           tmp_kappa(0),
                           tmp_phi(0)) +
                             tmp_tau(0) * arma::eye(m_x2, m_x2));
        } else {
          tmp_covariances.subcube(0, 0, i, m_x1, m_x1, i) +=
            (tmp_a - tmp_n_obs - 1) * tmp_A(2) * tmp_A(2) * (
                matern_cov(tmp_vec_D.subcube(0, 0, i, m_x1, m_x1, i),
                           tmp_kappa(0),
                           tmp_phi(0)) +
                             tmp_tau(0) * arma::eye(m_x1 + 1, m_x1 + 1));
        }
        
      }
    }
  };
  
#endif /* COVARIANCE_FN_H */