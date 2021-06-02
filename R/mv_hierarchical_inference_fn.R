#' =============================================================================
#'
#' INFERENCE (MCMC) FOR THE MULTIVARIATE HIERARCHICAL SPATIAL-T MODEL
#' author: IG
#' date: 05-18-21
#'
#' =============================================================================

#' Cross-Covariance elements list
#' 
#' Create a list with the covariance inverse, vector and conditional covariance 
#' elements for each observed location
#'
#' @param observed.coordinates (matrix) observed locations (n.obs X 2)
#' @param cov.family (string) name of the covariance function (e.g. "matern")
#' @param nu (numeric) covariance matrix range
#' @param kappa (numeric) smoothness parameter for the Matern covariance function
#' @param nugget (numeric) covariance matrix nugget
#' @param n.neighbors (integer) number of neighbors to include in N(s)
#' @param W (matrix) matrix where each row s identifies the members of N(s)
#' @return (list) list of four lists: covariance, covariance vector, inverse covariance
#'   and conditional covariance for each location and its neighbors
mv.compute.C.lists <- function(observed.coordinates, q, cov.family, nu, kappa, nugget, 
                            n.neighbors, W) {
  
  n.obs <- dim(observed.distance)[1]
  
  # Find C
  C.posterior <- mkSpCov(coords = observed.coordinates,
                         cov.model = cov.family,
                         K = matrix(1, nrow = q, ncol = q), # partial sill
                         Psi = matrix(nugget, nrow = q, col = q), # nugget
                         theta = c(rep(nu, q), rep(kappa, q)))
  
  # Create reduced C list needed for MCMC
  solve.C.list <- array(0, dim = c(n.neighbors * q, n.neighbors * q, n.obs))
  vec.C.list <- array(0, dim = c(n.obs * q, n.neighbors * q))
  cond.C.list <- rep(0, n.obs * q)
  cov.C.list <- array(0, dim = c(n.neighbors * q, n.neighbors * q, n.obs))
  for(x in 2:n.obs) {
    if (x < n.neighbors + 1) {
      n.ind <- seq(1, x - 1)
    } else {
      n.ind <- W[x, W[x, ] != 0]
    }
    x.m <- min(n.neighbors, x - 1)
    x.m.1 <- x.m + 1
    
    neighbors.vec <- vec(seq((n.ind - 1) * q + 1, n.ind * q))
    index.vec <- c(seq((x - 1) * q + 1, x * q), neighbors.vec)

    C.posterior.neighbors <- C.posterior[index.vec, index.vec]
    
    cov.C.list[1:(x.m * q), 1:(x.m * q), x] <- C.posterior.neighbors[2:(x.m.1 * q), 2:(x.m.1 * q)]
    solve.C.list[1:(x.m * q), 1:(x.m * q), x] <- solve(C.posterior.neighbors[2:(x.m.1 * q), 2:(x.m.1 * q)])
    vec.C.list[x, 1:(x.m * q)] <- C.posterior.neighbors[1:q, 2:(x.m.1 * q)]
    cond.C.list[x] <- C.posterior.neighbors[1:q,1:q] -  C.posterior.neighbors[1:q, 2:(x.m.1 * q)] %*%
      solve.C.list[1:(x.m * q), (x.m * q), x] %*% C.posterior.neighbors[2:(x.m.1 * q), 1:q]
  }
  
  return(list(cov.C.list, vec.C.list, solve.C.list, cond.C.list))
}

#' Observational Error Sampling (tau^2)
#' 
#' sample observational error variance (tau^2) with Inverse Gamma prior
#'
#' @param tau.prior (vector) shape and scale of inverse gamma prior on tau^2
#' @param y (vector) data
#' @param w (vector) spatial random effects
#' @param beta (vector) fixed effects mean (Xb)
#' @return (numeric) Gibbs posterior sample for the observational error tau^2 
mv.sample.tau.gibbs <- function(tau.prior, y, w, beta) {
  
  n.obs <- length(y)
  
  shape <- tau.prior[1] + n.obs / 2
  rate <- tau.prior[2] + sum((y - w - beta)^2) / 2
  
  tau <- rinvgamma(1, shape, rate)
  
  return(tau)
}
#' Partial Sill Sampling (sigma^2)
#' 
#' sample variance (sigma^2) with Inverse Gamma prior
#'
#' @param sigma.prior (vector) shape and scale of inverse gamma prior on sigma^2
#' @param w (vector) spatial random effects
#' @param W (matrix) neighbor matrix
#' @param B (matrix) matrix of hidden neighbor effects
#' @param phi (list) list of matrix with the hidden variance.  
#' @return (numeric) Gibbs posterior sample for the partial sill tau^2 
mv.sample.sig.gibbs <- function(sigma.prior, w, W, B, phi) {
  
  n.obs <- length(w)
  shape <- sigma.prior[1] + n.obs / 2
  
  # mean.neighbors <- rowSums(gamma[11:1000,] * t(apply(W[11:1000,], 1, function(s) w[s])))
  # rate <- sigma.prior[2] + sum(((w[11:1000] - mean.neighbors)^2)/phi[11:1000])/ 2
  mean.neighbors <- rowSums(B * t(apply(W, 1, function(s) w[s]))) # this needs to be updated
  rate <- sigma.prior[2] + sum(((w - mean.neighbors)^2)/phi)/ 2 # this needs to be updated
  
  sigma.new <- rinvgamma(1, shape, rate)
  
  return(sigma.new)
}

#' Inference for the hierarchical spatial-t model
#' 
#' the function uses MCMC to obtain posterior inference on the parameters of the
#' hierarchical model
#' 
#' @param Y 
#' @param observed.locations
#' @param model
#' @param alpha 
#' @param tau.prior
#' @param sig.prior
#' @param sigma.start
#' @param n.neighbors
#' @param cov.pars
#' @param cov.family
#' @param smoothness
#' @param nugget
#' @param mcmc.samples

#'    
#' @export
NIGP.hierarchical.rcpp <- function(Y, observed.locations, model, alpha,
                                   tau.prior, sig.prior, sigma.start, n.neighbors, 
                                   cov.pars, cov.family, smoothness, 
                                   nugget, mcmc.samples) {
  
  # Error checks
  if (!(model %in% c("full", "reduced"))) {
    stop("Invalid model choice. The model choices are 'full' and 'reduced'.")
  }
  
  # Set up neighbors
  n.obs <- dim(Y)[1]
  q <- dim(Y)[2]
  Y <- vec(Y)
  nq <- n.obs * q
  x.m.all <- c(seq(0, n.neighbors), rep(n.neighbors, n.obs - n.neighbors - 1))
  observed.distance <- as.matrix(fields::rdist(observed.locations, observed.locations))
  W <- create.W.neighbors.matrix(observed.distance, n.neighbors)
  
  # Set up initial values
  nu.cur <- cov.pars[2]
  tau.cur <- 0.5
  sig.cur <- sigma.start
  alpha.cur <- alpha
  w.cur <- Y
  phi.cur <- rep(0.1, n.obs)
  beta.cur <- 0
  gamma.cur <- array(0.1, dim = c(n.obs, n.neighbors))
  
  # Compute the initial Covariance matrices
  C.lists.cur <- compute.C.lists(observed.distance, cov.family, nu.cur, kappa, nugget, n.neighbors, W)
  cov.C.array <- C.lists.cur[[1]]
  vec.C.array <- C.lists.cur[[2]]
  cond.C.array <- C.lists.cur[[4]]
  solve.C.array <- C.lists.cur[[3]]
  
  # Store samples
  posterior.samples <- list("tau" = rep(NA, mcmc.samples),
                            "sigma" = rep(NA, mcmc.samples),
                            "beta" = rep(NA, mcmc.samples),
                            "phi" = array(NA, dim = c(mcmc.samples, n.obs)),
                            "w" = array(NA, dim = c(mcmc.samples, n.obs)),
                            "gamma" = array(NA, dim = c(mcmc.samples, n.obs, n.neighbors)),
                            "alpha" = rep(NA, mcmc.samples),
                            "nu" = rep(NA, mcmc.samples))
  
  for(i in 1:mcmc.samples) {
    if(i %% 100 == 0) {print(i)}
    
    samples <- data_loop_rcpp_arm(alpha.cur, n.obs,
                                  vec.C.array, solve.C.array, cov.C.array, cond.C.array,
                                  w.cur, n.neighbors, x.m.all,
                                  W - 1 , tau.cur, phi.cur, Y, beta.cur, sig.cur)
    
    
    w.cur[(n.neighbors + 1):n.obs] <- samples[(n.neighbors + 1):n.obs, 1]
    phi.cur[(n.neighbors + 1):n.obs] <- samples[(n.neighbors + 1):n.obs, 2]
    gamma.cur[(n.neighbors + 1):n.obs, ] <- samples[(n.neighbors + 1):n.obs, 3:(n.neighbors + 2)]
    
    # Posterior sample for fixed effects (beta)
    # v <- 1 / (1 + n.obs / tau.cur)
    # mea <- v / tau.cur * sum(y - w.cur)
    # beta.cur <- rnorm(1, mea, sqrt(v))
    
    if(model == "reduced") {
      # alpha (degrees of freedom) and nu (range parameter) are fixed
      tau.cur <- sample.tau.gibbs(tau.prior, Y, w.cur, beta.cur)
      sig.cur <- sample.sig.gibbs(sig.prior, w.cur, W, gamma.cur, phi.cur)
      
    } else {
      # full model
      tau.cur <- sample.tau.gibbs(tau.prior, Y, w.cur, beta.cur)
      sig.cur <- sample.sig.gibbs(sig.prior, w.cur, W, gamma.cur, phi.cur)
      alpha.cur <- sample.alpha.metropolis(1, alpha.cur, phi.cur, gamma.cur,
                                           cond.C.array, solve.C.array, vec.C.array, x.m.all)
      nu.samples <- sample.nu.metropolis(c(3,1), nu.cur, cov.C.array, cond.C.array,
                                         solve.C.array, vec.C.array, observed.distance,
                                         cov.family, kappa, n.neighbors, W, n.obs,
                                         alpha.cur, phi.cur, gamma.cur, x.m.all, nugget)
      nu.cur <- nu.samples[[1]]
      cov.C.array <- nu.samples[[2]]
      vec.C.array <- nu.samples[[3]]
      cond.C.array <- nu.samples[[4]]
      solve.C.array <- nu.samples[[5]]
    }
    
    # Store values
    posterior.samples[["tau"]][i] <- tau.cur
    posterior.samples[["sigma"]][i] <- sig.cur
    posterior.samples[["phi"]][i, ] <- phi.cur
    posterior.samples[["w"]][i, ] <- w.cur
    posterior.samples[["gamma"]][i, , ] <- gamma.cur
    posterior.samples[["beta"]][i] <- beta.cur
    posterior.samples[["alpha"]][i] <- alpha.cur
    posterior.samples[["nu"]][i] <- nu.cur
  }
  
  return(posterior.samples)
}
