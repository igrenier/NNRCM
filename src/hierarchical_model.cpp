#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

#include <RcppParallel.h>
// [[Rcpp::depends(RcppParallel)]]
using namespace RcppParallel;

#include <Rcpp.h>
using namespace Rcpp;


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
  arma::mat& tmp_vec_C;
  arma::cube& tmp_solve_C;
  arma::cube& tmp_cov_C;
  arma::colvec& tmp_cond_C;
  arma::colvec& tmp_w;
  int& tmp_n_neighbors;
  arma::colvec& tmp_m_location;
  arma::Mat<int>& tmp_W;
  arma::colvec& tmp_phi;
  arma::colvec& tmp_y;
  arma::mat& tmp_samples;       // output matrix to write to

  MCMC_update(double& a, double& n_obs, arma::mat& vec_C, arma::cube& solve_C,
              arma::cube& cov_C, arma::colvec& cond_C, arma::colvec& w, int& n_neighbors,
              arma::colvec& m_location,
              arma::Mat<int>& W, double& tau, arma::colvec& phi,
              arma::colvec& y, double& beta, double& sig, arma::mat& samples)
    : tmp_a(a), tmp_n_obs(n_obs), tmp_vec_C(vec_C), tmp_solve_C(solve_C),
      tmp_cov_C(cov_C), tmp_cond_C(cond_C), tmp_w(w), tmp_n_neighbors(n_neighbors),
      tmp_m_location(m_location),
      tmp_W(W), tmp_tau(tau), tmp_phi(phi), tmp_y(y), tmp_beta(beta), tmp_sig(sig),
      tmp_samples(samples){}

  void operator()(std::size_t begin, std::size_t end) {

    for (std::size_t i=begin; i < end; i++) {
      arma::colvec w_neighbors_c = tmp_w;
      // if (i < tmp_n_neighbors) {
      //   w_neighbors_c = tmp_w.subvec(0, i - 1);
      // } else {
      arma::uvec idx = arma::conv_to<arma::uvec>::from(tmp_W.submat(i, 0, i, tmp_n_neighbors - 1));
      w_neighbors_c = tmp_w.elem(idx);
      //}
      //arma::colvec w_neighbors_c(tmp_n_neighbors, arma::fill::zeros);
      //Rcpp::Rcout << "i made it";

      // Crop to only needed
      arma::rowvec A = tmp_vec_C.row(i);
      arma::mat B = tmp_solve_C.slice(i);
      arma::mat C = tmp_cov_C.slice(i);

      // below, where it says tmp_n_neighbors, it may need to be tmp_m_locations(i) to define the
      // length of the vector gamma
      tmp_samples.submat(i, 2, i, tmp_n_neighbors + 1) =
        gamma_rcpp_arm(tmp_a,
                       tmp_n_obs,
                       A.t(),
                       B,
                       C,
                       tmp_phi(i),
                       w_neighbors_c,
                       tmp_w(i),
                       tmp_sig);
      //Rcpp::Rcout << "i made it";
      tmp_samples(i, 1) = phi_rcpp_arm(tmp_a,
                  tmp_n_neighbors,
                  tmp_w(i),
                  w_neighbors_c,
                  B,
                  C,
                  tmp_cond_C(i),
                  A.t(),
                  tmp_n_obs,
                  tmp_samples.submat(i, 2, i, tmp_n_neighbors + 1),
                  tmp_tau,
                  tmp_y(i),
                  tmp_sig);

      tmp_samples(i, 0) = w_rcpp_arm(tmp_samples.submat(i, 2, i, tmp_n_neighbors + 1),
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
arma::mat data_loop_rcpp_arm(double a, double n_obs, arma::mat& vec_C, Rcpp::NumericVector& solve_C,
                             Rcpp::NumericVector& cov_C, arma::colvec cond_C, arma::colvec w, int n_neighbors,
                             arma::colvec m_location,
                             arma::Mat<int> W, double tau,  arma::colvec phi,
                             arma::colvec y, double beta, double sig) {


  // Gibbs sampling
  // create the output matrix
  arma::mat samples;
  samples.zeros(n_obs, n_neighbors + 2);

  // Create cubes
  arma::cube cube_solve_C = Rcpp::as<arma::cube>(solve_C);
  arma::cube cube_cov_C = Rcpp::as<arma::cube>(cov_C);

  // create the worker
  MCMC_update MCMC_update(a, n_obs, vec_C, cube_solve_C, cube_cov_C, cond_C,
                          w, n_neighbors, m_location, W, tau, phi, y, beta, sig,
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

