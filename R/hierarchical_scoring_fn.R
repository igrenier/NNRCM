#' =============================================================================
#'
#' SCORING OF THE HIERARCHICAL SPATIAL-T MODEL
#' author: IG
#' date: 04-20-21
#'
#' =============================================================================

#' CRPS scoring function
#'
#' This function computes the continuous ranked probability score (CRPS) given
#' observations and posterior predictive samples.
#'
#' @param Y (vector) response value for our dataset
#' @param posterior.predictive.samples (matrix) posterior predictive samples
#'   associated with the response vector Y. The number of rows must correspond
#'   to the length of the vector Y.
#' @return (numeric) the function returns the crps value

NIGP.crps <- function(Y, posterior.predictive.samples) {

  n.obs <- length(Y)
  mcmc.samples <- ncol(posterior.predictive.samples)

  # term 1: mean absolute predictive error
  abs.mean.diff <- 1 / mcmc.samples * rowSums(abs(posterior.predictive.samples - Y))

  # term 2: within sample variability
  abs.within.var <- rep(0, n.obs)
  for(i in 1:n.obs) {
    abs.within.var[i] <- 1 / (2 * mcmc.samples^2) *
      sum(abs(sapply(posterior.predictive.samples[i, ],
                     "-",
                     posterior.predictive.samples[i, ])))
  }

  # compute the mean crps score over all observed values
  mean.crps <- mean(abs.mean.diff - abs.within.var)

  return(mean.crps)

}

#' PMSE scoring function
#'
#' This function computes the predictive mean square error (PMSE) given
#' observations and posterior predictive samples.
#'
#' @param Y (vector) response value for our dataset
#' @param posterior.predictive.samples (matrix) posterior predictive samples
#'   associated with the response vector Y. The number of rows must correspond
#'   to the length of the vector Y.
#' @return (numeric) the function returns the pmse value

NIGP.pmse <- function(Y, posterior.predictive.samples) {

  n.obs <- length(Y)
  mcmc.samples <- ncol(posterior.predictive.samples)

  # compute the predicted value estimate
  mu.rep <- apply(posterior.predictive.samples, 1, mean)

  # compute the mean squared difference between predictions and observations
  pmse <- 1 / n.obs * sum((mu.rep - Y)^2)

  return (pmse)

}

#' GG scoring function
#'
#' This function computes the Gelfand & Ghosh (GG) score given a specific k,
#' observations and posterior predictive samples.
#'
#' @param Y (vector) response value for our dataset
#' @param posterior.predictive.samples (matrix) posterior predictive samples
#'   associated with the response vector Y. The number of rows must correspond
#'   to the length of the vector Y.
#' @param k (numeric) nonnegative penalty term from GG criterion
#' @return (numeric) the function returns the GG value for the specified penalty k

NIGP.GG <- function(Y, posterior.predictive.samples, k) {

  # term 1: within sample variance
  var.rep <- apply(posterior.predictive.samples, 1, var)

  # term 2: model variancce
  mu.rep <- apply(posterior.predictive.samples, 1, mean)

  # GG
  Dk.m <- sum(var.rep) + k / (k + 1) * sum((Y - mu.rep)^2)

  return(Dk.m)

}

#' Coverage scoring function
#'
#' This function computes the average coverage of the 95% predictive intervals.
#'
#' @param Y (vector) response value for our dataset
#' @param posterior.predictive.samples (matrix) posterior predictive samples
#'   associated with the response vector Y. The number of rows must correspond
#'   to the length of the vector Y.
#' @return (numeric) the function returns the average coverage for the sample

NIGP.coverage <- function(Y, pp.samples) {
  
  ci.bounds <- cbind(apply(pp.samples, 1, function(x) quantile(x, p = 0.025)),
                     apply(pp.samples, 1, function(x) quantile(x, p = 0.975)))
  
  coverage <- mean(Y >= ci.bounds[, 1] & Y <= ci.bounds[, 2])
  
  return(coverage)
  
}


#' Scoring function
#'
#' This function computes the Gelfand & Ghosh (GG) score, the continuous ranked 
#' probability score (CRPS), the predictive mean squared error (PMSE) value 
#' and the coverage given the true observations and the posterior predictive samples.
#'
#' @param Y (vector) response value for our dataset
#' @param posterior.predictive.samples (matrix) posterior predictive samples
#'   associated with the response vector Y. The number of rows must correspond
#'   to the length of the vector Y.
#' @param k (numeric) nonnegative penalty term from GG criterion
#' @return (numeric) the function returns the GG value for the specified penalty k
#' @export
NNRCM.scores <- function(Y, pp.samples, k = 1) {
  
  scores <- rep(0, 4)
  
  scores[1] <- NIGP.pmse(Y, pp.samples)
  scores[2] <- NIGP.crps(Y, pp.samples)
  scores[3] <- NIGP.GG(Y, pp.samples, k)
  scores[4] <- NIGP.coverage(Y, pp.samples)
  
  names(scores) <- c("PMSE", "CRPS", "PPLC", "Coverage")
  
  return (scores)
}