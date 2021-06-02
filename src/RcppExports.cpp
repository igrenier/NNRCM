// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// phi_rcpp_arm
double phi_rcpp_arm(double a, int x_m, double w, arma::colvec& w_neighbors, const arma::mat& solve_C, const arma::mat& cov_C, double cond_C, const arma::colvec& vec_C, int n_obs, const arma::rowvec& gammar, double tau, double y, double sig);
RcppExport SEXP _spatialT2_phi_rcpp_arm(SEXP aSEXP, SEXP x_mSEXP, SEXP wSEXP, SEXP w_neighborsSEXP, SEXP solve_CSEXP, SEXP cov_CSEXP, SEXP cond_CSEXP, SEXP vec_CSEXP, SEXP n_obsSEXP, SEXP gammarSEXP, SEXP tauSEXP, SEXP ySEXP, SEXP sigSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< int >::type x_m(x_mSEXP);
    Rcpp::traits::input_parameter< double >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type w_neighbors(w_neighborsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type solve_C(solve_CSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type cov_C(cov_CSEXP);
    Rcpp::traits::input_parameter< double >::type cond_C(cond_CSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type vec_C(vec_CSEXP);
    Rcpp::traits::input_parameter< int >::type n_obs(n_obsSEXP);
    Rcpp::traits::input_parameter< const arma::rowvec& >::type gammar(gammarSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< double >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type sig(sigSEXP);
    rcpp_result_gen = Rcpp::wrap(phi_rcpp_arm(a, x_m, w, w_neighbors, solve_C, cov_C, cond_C, vec_C, n_obs, gammar, tau, y, sig));
    return rcpp_result_gen;
END_RCPP
}
// w_rcpp_arm
double w_rcpp_arm(arma::rowvec gammar, double phi, arma::colvec& w_neighbors, double y, double tau, double beta, double sig);
RcppExport SEXP _spatialT2_w_rcpp_arm(SEXP gammarSEXP, SEXP phiSEXP, SEXP w_neighborsSEXP, SEXP ySEXP, SEXP tauSEXP, SEXP betaSEXP, SEXP sigSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::rowvec >::type gammar(gammarSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type w_neighbors(w_neighborsSEXP);
    Rcpp::traits::input_parameter< double >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type sig(sigSEXP);
    rcpp_result_gen = Rcpp::wrap(w_rcpp_arm(gammar, phi, w_neighbors, y, tau, beta, sig));
    return rcpp_result_gen;
END_RCPP
}
// mvrnormArma
const arma::rowvec mvrnormArma(const arma::rowvec mean, const arma::mat variance);
RcppExport SEXP _spatialT2_mvrnormArma(SEXP meanSEXP, SEXP varianceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::rowvec >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type variance(varianceSEXP);
    rcpp_result_gen = Rcpp::wrap(mvrnormArma(mean, variance));
    return rcpp_result_gen;
END_RCPP
}
// gamma_rcpp_arm
arma::rowvec gamma_rcpp_arm(double a, double n_obs, const arma::colvec& vec_C, const arma::mat& solve_C, const arma::mat& cov_C, double phi, arma::colvec& w_neighbors, double w, double sig);
RcppExport SEXP _spatialT2_gamma_rcpp_arm(SEXP aSEXP, SEXP n_obsSEXP, SEXP vec_CSEXP, SEXP solve_CSEXP, SEXP cov_CSEXP, SEXP phiSEXP, SEXP w_neighborsSEXP, SEXP wSEXP, SEXP sigSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type n_obs(n_obsSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type vec_C(vec_CSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type solve_C(solve_CSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type cov_C(cov_CSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type w_neighbors(w_neighborsSEXP);
    Rcpp::traits::input_parameter< double >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type sig(sigSEXP);
    rcpp_result_gen = Rcpp::wrap(gamma_rcpp_arm(a, n_obs, vec_C, solve_C, cov_C, phi, w_neighbors, w, sig));
    return rcpp_result_gen;
END_RCPP
}
// data_loop_rcpp_arm
arma::mat data_loop_rcpp_arm(double a, double n_obs, arma::mat& vec_C, Rcpp::NumericVector& solve_C, Rcpp::NumericVector& cov_C, arma::colvec cond_C, arma::colvec w, int n_neighbors, arma::colvec m_location, arma::Mat<int> W, double tau, arma::colvec phi, arma::colvec y, double beta, double sig);
RcppExport SEXP _spatialT2_data_loop_rcpp_arm(SEXP aSEXP, SEXP n_obsSEXP, SEXP vec_CSEXP, SEXP solve_CSEXP, SEXP cov_CSEXP, SEXP cond_CSEXP, SEXP wSEXP, SEXP n_neighborsSEXP, SEXP m_locationSEXP, SEXP WSEXP, SEXP tauSEXP, SEXP phiSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP sigSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type n_obs(n_obsSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type vec_C(vec_CSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector& >::type solve_C(solve_CSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector& >::type cov_C(cov_CSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type cond_C(cond_CSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type w(wSEXP);
    Rcpp::traits::input_parameter< int >::type n_neighbors(n_neighborsSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type m_location(m_locationSEXP);
    Rcpp::traits::input_parameter< arma::Mat<int> >::type W(WSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type sig(sigSEXP);
    rcpp_result_gen = Rcpp::wrap(data_loop_rcpp_arm(a, n_obs, vec_C, solve_C, cov_C, cond_C, w, n_neighbors, m_location, W, tau, phi, y, beta, sig));
    return rcpp_result_gen;
END_RCPP
}
// data_loop_rcpp_no_nugget
arma::mat data_loop_rcpp_no_nugget(double a, double n_obs, const Rcpp::List vec_C, Rcpp::List solve_C, Rcpp::List cov_C, arma::colvec& cond_C, int n_neighbors, const arma::colvec& m_location, const arma::Mat<int> W, double tau, const arma::colvec& phi, const arma::colvec y, double beta, double sig);
RcppExport SEXP _spatialT2_data_loop_rcpp_no_nugget(SEXP aSEXP, SEXP n_obsSEXP, SEXP vec_CSEXP, SEXP solve_CSEXP, SEXP cov_CSEXP, SEXP cond_CSEXP, SEXP n_neighborsSEXP, SEXP m_locationSEXP, SEXP WSEXP, SEXP tauSEXP, SEXP phiSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP sigSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type n_obs(n_obsSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type vec_C(vec_CSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type solve_C(solve_CSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type cov_C(cov_CSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type cond_C(cond_CSEXP);
    Rcpp::traits::input_parameter< int >::type n_neighbors(n_neighborsSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type m_location(m_locationSEXP);
    Rcpp::traits::input_parameter< const arma::Mat<int> >::type W(WSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< const arma::colvec >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type sig(sigSEXP);
    rcpp_result_gen = Rcpp::wrap(data_loop_rcpp_no_nugget(a, n_obs, vec_C, solve_C, cov_C, cond_C, n_neighbors, m_location, W, tau, phi, y, beta, sig));
    return rcpp_result_gen;
END_RCPP
}
// determinant_rcpp_arm
double determinant_rcpp_arm(arma::mat& C, arma::colvec vec_C, double C_i);
RcppExport SEXP _spatialT2_determinant_rcpp_arm(SEXP CSEXP, SEXP vec_CSEXP, SEXP C_iSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type C(CSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type vec_C(vec_CSEXP);
    Rcpp::traits::input_parameter< double >::type C_i(C_iSEXP);
    rcpp_result_gen = Rcpp::wrap(determinant_rcpp_arm(C, vec_C, C_i));
    return rcpp_result_gen;
END_RCPP
}
// marginal_rcpp_arm
double marginal_rcpp_arm(int n_neighbors, double n_obs, Rcpp::NumericVector& C, Rcpp::NumericVector& C_y, double a, double phi, double sigma, double tau, double small);
RcppExport SEXP _spatialT2_marginal_rcpp_arm(SEXP n_neighborsSEXP, SEXP n_obsSEXP, SEXP CSEXP, SEXP C_ySEXP, SEXP aSEXP, SEXP phiSEXP, SEXP sigmaSEXP, SEXP tauSEXP, SEXP smallSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n_neighbors(n_neighborsSEXP);
    Rcpp::traits::input_parameter< double >::type n_obs(n_obsSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector& >::type C(CSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector& >::type C_y(C_ySEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< double >::type small(smallSEXP);
    rcpp_result_gen = Rcpp::wrap(marginal_rcpp_arm(n_neighbors, n_obs, C, C_y, a, phi, sigma, tau, small));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_hello_world
arma::mat rcpparma_hello_world();
RcppExport SEXP _spatialT2_rcpparma_hello_world() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcpparma_hello_world());
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_outerproduct
arma::mat rcpparma_outerproduct(const arma::colvec& x);
RcppExport SEXP _spatialT2_rcpparma_outerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_outerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_innerproduct
double rcpparma_innerproduct(const arma::colvec& x);
RcppExport SEXP _spatialT2_rcpparma_innerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_innerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_bothproducts
Rcpp::List rcpparma_bothproducts(const arma::colvec& x);
RcppExport SEXP _spatialT2_rcpparma_bothproducts(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_bothproducts(x));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_spatialT2_phi_rcpp_arm", (DL_FUNC) &_spatialT2_phi_rcpp_arm, 13},
    {"_spatialT2_w_rcpp_arm", (DL_FUNC) &_spatialT2_w_rcpp_arm, 7},
    {"_spatialT2_mvrnormArma", (DL_FUNC) &_spatialT2_mvrnormArma, 2},
    {"_spatialT2_gamma_rcpp_arm", (DL_FUNC) &_spatialT2_gamma_rcpp_arm, 9},
    {"_spatialT2_data_loop_rcpp_arm", (DL_FUNC) &_spatialT2_data_loop_rcpp_arm, 15},
    {"_spatialT2_data_loop_rcpp_no_nugget", (DL_FUNC) &_spatialT2_data_loop_rcpp_no_nugget, 14},
    {"_spatialT2_determinant_rcpp_arm", (DL_FUNC) &_spatialT2_determinant_rcpp_arm, 3},
    {"_spatialT2_marginal_rcpp_arm", (DL_FUNC) &_spatialT2_marginal_rcpp_arm, 9},
    {"_spatialT2_rcpparma_hello_world", (DL_FUNC) &_spatialT2_rcpparma_hello_world, 0},
    {"_spatialT2_rcpparma_outerproduct", (DL_FUNC) &_spatialT2_rcpparma_outerproduct, 1},
    {"_spatialT2_rcpparma_innerproduct", (DL_FUNC) &_spatialT2_rcpparma_innerproduct, 1},
    {"_spatialT2_rcpparma_bothproducts", (DL_FUNC) &_spatialT2_rcpparma_bothproducts, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_spatialT2(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
