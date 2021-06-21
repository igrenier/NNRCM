#' =============================================================================
#'
#' INFERENCE (MCMC) FOR THE HIERARCHICAL SPATIAL-T MODEL
#' author: IG
#' date: 04-20-21
#'
#' =============================================================================

#' Covariance elements list
#' 
#' Create a list with the covariance inverse, vector and conditional covariance 
#' elements for each observed location
#'
#' @param observed.distance (matrix) observed distance between all locations (n.obs X n.obs)
#' @param cov.family (string) name of the covariance function (e.g. "matern")
#' @param nu (numeric) covariance matrix range
#' @param kappa (numeric) smoothness parameter for the Matern covariance function
#' @param nugget (numeric) covariance matrix nugget
#' @param n.neighbors (integer) number of neighbors to include in N(s)
#' @param W (matrix) matrix where each row s identifies the members of N(s)
#' @return (list) list of four lists: covariance, covariance vector, inverse covariance
#'   and conditional covariance for each location and its neighbors
compute.C.lists <- function(observed.distance, cov.family, nu, kappa, nugget, 
                            n.neighbors, W) {

  n.obs <- dim(observed.distance)[1]
  
  # Find C
  C.posterior <- cov.spatial(observed.distance,
                             cov.model = cov.family,
                             cov.pars = c(1, nu),
                             kappa = kappa) + nugget * diag(n.obs) # this part is to make the marginal and the hierarchical match

  # Create reduced C list needed for MCMC
  solve.C.list <- array(0, dim = c(n.neighbors, n.neighbors, n.obs))
  vec.C.list <- array(0, dim = c(n.obs, n.neighbors))
  cond.C.list <- rep(0, n.obs)
  cov.C.list <- array(0, dim = c(n.neighbors, n.neighbors, n.obs))
  for(x in 2:n.obs) {
    if (x < n.neighbors + 1) {
      n.ind <- seq(1, x - 1)
    } else {
      n.ind <- W[x, W[x, ] != 0]
    }
    x.m <- min(n.neighbors, x - 1)
    x.m.1 <- x.m + 1

    C.posterior.neighbors <- C.posterior[c(x, n.ind), c(x, n.ind)]


    cov.C.list[1:x.m, 1:x.m, x] <- C.posterior.neighbors[2:x.m.1, 2:x.m.1]
    solve.C.list[1:x.m, 1:x.m, x] <- solve(C.posterior.neighbors[2:x.m.1, 2:x.m.1])
    vec.C.list[x, 1:x.m] <- C.posterior.neighbors[1, 2:x.m.1]
    cond.C.list[x] <- C.posterior.neighbors[1,1] -  C.posterior.neighbors[1, 2:x.m.1] %*%
      solve.C.list[1:x.m, 1:x.m, x] %*% C.posterior.neighbors[2:x.m.1, 1]
  }

  return(list(cov.C.list, vec.C.list, solve.C.list, cond.C.list))
}


# sample nu parameter with Gamma prior
sample.nu.metropolis <- function(nu.prior, nu.cur, C.cov.cur, C.cond.cur, C.inv.cur, C.vec.cur,
                                 distance, cov.family, kappa, n.neighbors, W, n.obs, alpha,
                                 phi, gamma, x.m, prior.nugget) {

  # generate potential nu sample
  nu.new <- abs(rnorm(1, nu.cur, 0.25))
  C.lists.new <- compute.C.lists(distance, cov.family, nu.new, kappa, prior.nugget, n.neighbors, W)
  C.cov.new <- C.lists.new[[1]]
  C.vec.new <- C.lists.new[[2]]
  C.cond.new <- C.lists.new[[4]]
  C.inv.new <- C.lists.new[[3]]

  # compute prior and likelihood
  prior.cur <- dgamma(nu.cur, nu.prior[1], nu.prior[2], log = TRUE)
  prior.new <- dgamma(nu.new, nu.prior[1], nu.prior[2], log = TRUE)

  likelihood.cur <- sum(dinvgamma(phi[2:n.obs], alpha - n.obs + 1 + x.m[2:n.obs],
                                  (alpha - n.obs - 1) * (C.cond.cur)[2:n.obs],
                                  log = TRUE)) +
    sum(sapply(seq(3, n.obs), function(g) dmvn(gamma[g, 1:(x.m[g])], t(C.inv.cur[1:(x.m[g]),1:(x.m[g]), g] %*% C.vec.cur[g,1:(x.m[g])]),
                                               phi[g] / (alpha - n.obs - 1) * C.inv.cur[1:(x.m[g]),1:(x.m[g]), g], log = TRUE)))
  likelihood.new <- sum(dinvgamma(phi[2:n.obs], alpha - n.obs + 1 + x.m[2:n.obs],
                                  (alpha - n.obs - 1) * (C.cond.new)[2:n.obs],
                                  log = TRUE)) +
    sum(sapply(seq(3, n.obs), function(g) dmvn(gamma[g, 1:(x.m[g])], t(C.inv.new[1:(x.m[g]),1:(x.m[g]), g] %*% C.vec.new[g,1:(x.m[g])]),
                                               phi[g] / (alpha - n.obs - 1) * C.inv.new[1:(x.m[g]),1:(x.m[g]), g], log = TRUE)))


  # update the sample
  posterior.ratio <- prior.new + likelihood.new - prior.cur - likelihood.cur
  if(log(runif(1)) < posterior.ratio) {
    nu.cur <- nu.new
    C.cov.cur <- C.cov.new
    C.vec.cur <- C.vec.new
    C.cond.cur <- C.cond.new
    C.inv.cur <- C.inv.new
  }

  return(list(nu.cur, C.cov.cur, C.vec.cur, C.cond.cur, C.inv.cur))

}

# sample alpha with Pareto prior
sample.alpha.metropolis <- function(alpha.prior, alpha.cur, phi, gamma, C.cond, C.inv, C.vec, x.m){

  n.obs <- length(phi)

  # new alpha:
  alpha.new <- max(rnorm(1, alpha.cur, 5), n.obs + 2)

  # prior
  prior.cur <- - (alpha.prior + 1) * log(alpha.cur)
  prior.new <- - (alpha.prior + 1) * log(alpha.new)

  # likelihood
  likelihood.cur <- sum(dinvgamma(phi[2:n.obs], alpha.cur - n.obs + 1 + x.m[2:n.obs],
                                  (alpha.cur - n.obs - 1) * unlist(C.cond)[2:n.obs],
                                  log = TRUE)) +
    sum(sapply(seq(3, n.obs), function(g) dmvn(gamma[g, 1:(x.m[g])], t(C.inv[1:(x.m[g]),1:(x.m[g]), g] %*% C.vec[g, 1:(x.m[g])]),
                                               phi[g] / (alpha.cur - n.obs - 1) * C.inv[1:(x.m[g]),1:(x.m[g]), g], log = TRUE)))
  likelihood.new <- sum(dinvgamma(phi[2:n.obs], alpha.new - n.obs + 1 + x.m[2:n.obs],
                                  (alpha.new - n.obs - 1) * unlist(C.cond)[2:n.obs],
                                  log = TRUE)) +
    sum(sapply(seq(3, n.obs), function(g) dmvn(gamma[g, 1:(x.m[g])], t(C.inv[1:(x.m[g]),1:(x.m[g]), g] %*% C.vec[g,1:(x.m[g])]),
                                               phi[g] / (alpha.new - n.obs - 1) * C.inv[1:(x.m[g]),1:(x.m[g]), g], log = TRUE)))

  # update
  posterior.ratio <- prior.new + likelihood.new - prior.cur - likelihood.cur
  if(log(runif(1)) < posterior.ratio) {
    alpha.cur <- alpha.new
  }

  return(alpha.cur)


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

sample.tau.gibbs <- function(tau.prior, y, w, beta) {

  n.obs <- length(y)

  shape <- tau.prior[1] + n.obs / 2
  rate <- tau.prior[2] + sum((y - w - beta)^2)/ 2

  tau <- rinvgamma(1, shape, rate)

  return(tau)
}

# sample variance (sigma^2) with Inverse Gamma prior
sample.sig.gibbs <- function(sigma.prior, w, W, gamma, phi) {

  n.obs <- length(w)
  shape <- sigma.prior[1] + n.obs / 2

  mean.neighbors <- rowSums(gamma[11:1000,] * t(apply(W[11:1000,], 1, function(s) w[s])))
  rate <- sigma.prior[2] + sum(((w[11:1000] - mean.neighbors)^2)/phi[11:1000])/ 2

  sigma.new <- rinvgamma(1, shape, rate)

  return(sigma.new)
}

#' Inference for the hierarchical spatial-t model
#' 
#' the function uses MCMC to obtain posterior inference on the parameters of the
#' hierarchical model
#' 
#' @param Y 
#' @param observed.distance
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
NIGP.hierarchical.rcpp <- function(Y, observed.distance, model, alpha,
                                   tau.prior, sig.prior, sigma.start, n.neighbors, 
                                   cov.pars, cov.family, smoothness, 
                                   nugget, mcmc.samples) {

  # Error checks
  if (!(model %in% c("full", "reduced"))) {
    stop("Invalid model choice. The model choices are 'full' and 'reduced'.")
  }
  
  # extract dimensions
  n.obs <- length(Y)
  x.m.all <- c(seq(0, n.neighbors), rep(n.neighbors, n.obs - n.neighbors - 1))
  
  # Create the neighbor structure
  W.list <- create.W.neighbors.matrix.spConjNNGP(observed.locations, n.neighbors)
  W <- array(0, dim = c(n.obs, n.neighbors))
  for(i in 2:(m+1)) {
    W[i, 1:(i-1)] <- seq(1, i - 1) 
  }
  W[(m+2):n.obs, ] <- matrix(unlist(W.list)[(m * (m+1) / 2 + 2) : ((n.obs - m - 1) * m + m * (m+1) / 2 + 1)], ncol = 10, byrow = TRUE)
  W <- cbind(1:n.obs, W)

  # Set up initial values
  nu.cur <- cov.pars[2]
  tau.cur <- 0.5
  sig.cur <- sigma.start
  alpha.cur <- alpha
  w.cur <- Y
  phi.cur <- rep(0.1, n.obs)
  beta.cur <- 0
  gamma.cur <- array(0.1, dim = c(n.obs, n.neighbors))

  # Compute a list of the distance matrices
  D.array <- array(0, dim = c(n.neighbors + 1, n.neighbors + 1, n.obs))
  for(x in 2:n.obs) {
    
    n.ind <- W.list[[x]]
    m.x <- min(n.neighbors + 1, x)
    
    D.array[1:m.x, 1:m.x, x] <- fields::rdist(observed.locations[c(x, n.ind), ], 
                                              observed.locations[c(x, n.ind), ])
    
  }

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

    samples <- data_loop_rcpp_arm(alpha.cur, n.obs, observed.distance,
                                  w.cur, n.neighbors, x.m.all,
                                  W - 1 , tau.cur, phi.cur, Y, beta.cur, sig.cur,
                                  smoothness, nu.cur)


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
      # full model (not working currently, need to re-add computing all the C matrices and vectors)
      tau.cur <- sample.tau.gibbs(tau.prior, Y, w.cur, beta.cur)
      sig.cur <- sample.sig.gibbs(sig.prior, w.cur, W, gamma.cur, phi.cur)
      # alpha.cur <- sample.alpha.metropolis(1, alpha.cur, phi.cur, gamma.cur,
      #                                      cond.C.array, solve.C.array, vec.C.array, x.m.all)
      # nu.samples <- sample.nu.metropolis(c(3,1), nu.cur, cov.C.array, cond.C.array,
      #                                    solve.C.array, vec.C.array, observed.distance,
      #                                    cov.family, kappa, n.neighbors, W, n.obs,
      #                                    alpha.cur, phi.cur, gamma.cur, x.m.all, nugget)
      # nu.cur <- nu.samples[[1]]
      # cov.C.array <- nu.samples[[2]]
      # vec.C.array <- nu.samples[[3]]
      # cond.C.array <- nu.samples[[4]]
      # solve.C.array <- nu.samples[[5]]
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
