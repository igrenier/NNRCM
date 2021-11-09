#' =============================================================================
#'
#' INFERENCE (MCMC) FOR THE HIERARCHICAL SPATIAL-T MODEL
#' author: IG
#' date: 04-20-21
#'
#' =============================================================================
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
NNRCM.hierarchical.infer <- function(Y, observed.locations, X,
                                   starting.values,
                                   smoothness, 
                                   nugget,  
                                   a.nu.update.interval = 0, 
                                   priors = list(tau.prior = c(3,2),
                                                 sig.prior = c(3,2),
                                                 a.prior = 1,
                                                 nu.prior = c(5,2),
                                                 beta.prior = 0.1), 
                                   tuning = list(a.tuning = 5,
                                                 nu.tuning = 0.01),
                                   n.neighbors = 10,
                                   mcmc.samples = 1000) {

  # to be removed
  nugget.prior <- c(3,2)
  nugget.tuning <- 0.1
  
  # check the update interval
  if(a.nu.update.interval == 0){
    a.nu.update.interval <- mcmc.samples + 5
  }
  
  # extract dimensions and shorthands
  n.obs <- length(Y)
  nb.covariates <- ncol(as.matrix(X))
  m <- n.neighbors
  x.m.all <- c(seq(0, n.neighbors), rep(n.neighbors, n.obs - n.neighbors - 1))
  
  # Create the neighbor structure
  model <- create.W.neighbors.matrix.spConjNNGP(observed.locations, n.neighbors)
  W.list <- model$neighbor.info$n.indx
  W <- array(0, dim = c(n.obs, n.neighbors))
  for(i in 2:(m+1)) {
    W[i, 1:(i-1)] <- seq(1, i - 1) 
  }
  W[(m+2):n.obs, ] <- matrix(unlist(W.list)[(m * (m+1) / 2 + 2) : ((n.obs - m - 1) * m + m * (m+1) / 2 + 1)], ncol = m, byrow = TRUE)
  W <- cbind(1:n.obs, W)
  
  print("Neighbor structure retrieved")
  
  # Set up initial values
  nu.cur <- starting.values[3]
  tau.cur <- 0.5
  sig.cur <- starting.values[2]
  alpha.cur <- starting.values[1]
  w.cur <- Y
  gamma.cur <- array(0.1, dim = c(n.obs, n.neighbors))
  phi.cur <- rep(0.1, n.obs)
  beta.cur <- rep(0.01, nb.covariates)
  if(nb.covariates == 1) {
    beta.X <- as.matrix(X * beta.cur)
    X <- as.matrix(X)
  } else {
    beta.X <- X %*% beta.cur
  }
  
  gamma.cur <- array(0.1, dim = c(n.obs, n.neighbors))

  # Compute a list of the distance matrices
  D.array <- array(0, dim = c(n.neighbors + 1, n.neighbors + 1, n.obs))
  max.neighbors <- max(table(model$neighbor.info$nn.indx))
  W2.array <- array(0, dim = c(n.obs, max.neighbors))
  W2i.array <- array(0, dim = c(n.obs, max.neighbors))
  w2count.array <- rep(0, n.obs)
  for(x in 2:n.obs) {
    
    n.ind <- W.list[[x]]
    m.x <- min(n.neighbors + 1, x)
    
    D.array[1:m.x, 1:m.x, x] <- fields::rdist(observed.locations[c(x, n.ind), ], 
                                              observed.locations[c(x, n.ind), ])
    
    w2count.array[n.ind] <- w2count.array[n.ind] + 1
    pairs <- seq(1:m.x)
    for(i in 1:m.x) {
      W2.array[n.ind[i], w2count.array[n.ind[i]]] <- x
      W2i.array[n.ind[i], w2count.array[n.ind[i]]] <- pairs[i]
    }
  }

  print("Distance arrays built")
  
  # Store samples
  posterior.samples <- list("tau" = rep(NA, mcmc.samples),
                            "sigma" = rep(NA, mcmc.samples),
                            "beta" = array(NA, dim = c(nb.covariates, mcmc.samples)),
                            "phi" = array(NA, dim = c(n.obs, mcmc.samples)),
                            "w" = array(NA, dim = c(n.obs, mcmc.samples)),
                            # "gamma" = array(NA, dim = c(n.obs, n.neighbors, mcmc.samples)),
                            "alpha" = rep(NA, mcmc.samples),
                            "nu" = rep(NA, mcmc.samples),
                            "nugget" = rep(NA, mcmc.samples))

  print("Starting c++...")
  samples <- mcmc_loop(alpha.cur, n.obs, D.array,
                       w.cur, n.neighbors, x.m.all,
                       W - 1 , tau.cur, phi.cur, gamma.cur, Y, beta.X, sig.cur,
                       smoothness, nu.cur, priors$tau.prior, priors$sig.prior,
                       priors$beta.prior, nb.covariates, X, mcmc.samples,
                       nugget, priors$a.prior, tuning$a.tuning,
                       tuning$nu.tuning, priors$nu.prior, nugget.tuning, 
                       nugget.prior, a.nu.update.interval,
                       w2count.array, W2.array - 1, W2i.array - 1)
  
  # Store values
  # posterior.samples[["tau"]] <- samples[1, n.neighbors + 3, ]
  # posterior.samples[["sigma"]] <- samples[2, n.neighbors + 3, ]
  # posterior.samples[["phi"]] <- samples[, 2, ]
  # posterior.samples[["w"]] <- samples[, 1, ]
  # # posterior.samples[["gamma"]] <- samples[, 3:(n.neighbors + 2), ]
  # posterior.samples[["beta"]] <- samples[1, (n.neighbors + 5):(n.neighbors + nb.covariates + 4), ]
  # posterior.samples[["alpha"]] <- samples[3, n.neighbors + 3, ]
  # posterior.samples[["nu"]] <- samples[4, n.neighbors + 3, ]
  # posterior.samples[["nugget"]] <- samples[5, n.neighbors + 3, ]
  
  posterior.samples[["tau"]] <- samples[1, 3, ]
  posterior.samples[["sigma"]] <- samples[2, 3, ]
  posterior.samples[["phi"]] <- samples[, 2, ]
  posterior.samples[["w"]] <- samples[, 1, ]
  # posterior.samples[["gamma"]] <- samples[, 3:(n.neighbors + 2), ]
  posterior.samples[["beta"]] <- samples[1, 5:(nb.covariates + 4), ]
  posterior.samples[["alpha"]] <- samples[3, 3, ]
  posterior.samples[["nu"]] <- samples[4, 3, ]
  posterior.samples[["nugget"]] <- samples[5,  3, ]
  
  return(posterior.samples)
}


#' Inference for the bivariate hierarchical spatial-t model
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
NNRCM.bv.hierarchical.infer <- function(Y, observed.locations, X = 0, 
                                      starting.values = list(alpha = 0,
                                                             sigma = 0,
                                                             nu = 0,
                                                             nugget = 0),
                                      smoothness, 
                                      a.nu.update.interval,
                                      priors = list(tau.prior = c(3,2),
                                                    a.prior = 1,
                                                    nu.prior = c(5,2),
                                                    beta.prior = 0.1), 
                                      tuning = list(a.tuning = 5,
                                                    nu.tuning = 0.01),
                                      n.neighbors = 10, mcmc.samples = 1000, 
                                      A = diag(2)) {
  
  # Check if A is invertible
  out <- tryCatch(solve(A) %*% A, error = function(e) e)
  if(any(class(out) == "error")){
    print("The A you supplied is singular, you moron. Please try again.")
    stop()
  }
  
  # Check if any covariates are given
  if(X == 0){
    X <- rep(1, nrow(Y))
  }
  
  # Check that point estimates have been given
  if(starting.values$alpha == 0){
    print("You need to supply starting values and point estimates for the parameters")
    stop()
  }
  
  # # Melt the data by alternative sources/variate
  n.locations <- nrow(Y)
  nb.covariates <- ncol(as.matrix(X))
  total.n <- 2 * n.locations

  seq.1 <- seq(1, total.n, by = 2)
  seq.2 <- seq(2, total.n, by = 2)
  mydata.melt.wNA <- rep(0, total.n)
  mydata.melt.wNA[seq.1] <- Y[, 1]
  mydata.melt.wNA[seq.2] <- Y[, 2]
  mydata.melt <- na.omit(mydata.melt.wNA)
  Ynb <- array(1, dim = c(total.n, 3))
  Ynb[seq.1, 2] <- seq.1
  Ynb[seq.2, 2] <- seq.1
  Ynb[seq.1, 3] <- seq.2
  Ynb[seq.2, 3] <- seq.2
  Ynb[seq.1[is.na(Y[, 1])], 1] <- 2
  Ynb[seq.1[is.na(Y[, 2])], 1] <- 3
  Ynb[seq.2[is.na(Y[, 1])], 1] <- 2
  Ynb[seq.2[is.na(Y[, 2])], 1] <- 3
  seq.1.noNA <- seq.1[!is.na(Y[, 1])]
  seq.2.noNA <- seq.2[!is.na(Y[, 2])]

  observed.locations.melt <- matrix(0, nrow = total.n, ncol= 2)
  observed.locations.melt[seq.1, ] <- as.vector(as.matrix(observed.locations))
  observed.locations.melt[seq.2, ] <- as.vector(as.matrix(observed.locations))
  # # observed.locations.melt <- observed.locations.melt[-is.na(mydata.melt.wNA), ]

  # Extract values
  m <- n.neighbors
  
  # Create the neighbor structure
  W <- array(0, dim = c(total.n, 2 * n.neighbors))
  Wnb <- array(n.neighbors, dim = c(total.n , 3))
  Wnb[1:(2 * m), 2] <- sort(rep(seq(0, m-1), 2))
  Wnb[1:(2 * m), 3] <- sort(c(seq(0, m-1), seq(1, m-1), m))
  Wnb[seq.1, 1] <- 1
  Wnb[seq.2, 1] <- 2 
  model <- create.W.neighbors.matrix.spConjNNGP(observed.locations, n.neighbors)
  W.list <- model$neighbor.info$n.indx
  W.1 <- array(0, dim = c(n.locations, n.neighbors))
  W.1[(m+1):n.locations, ] <- 2 * matrix(unlist(W.list)[((m - 1) * m / 2 + 2) : ((n.locations - m - 1) * m + m * (m+1) / 2 + 1)], ncol = m, byrow = TRUE) - 1
  W.2 <- array(0, dim = c(n.locations, n.neighbors))
  W.2[m:n.locations, ] <- cbind(seq(2 * m - 1, total.n - 1, by = 2), 
                                    W.1[m:n.locations, 1:(m-1)])
  W[seq.1, 1:m] <- W.1
  W[seq.1, (m+1):(2 *m)] <- W.1 + 1
  W[seq.2, 1:m] <- W.1 + 1
  W[seq.2, (m+1):(2 *m)] <- W.2
  W[1:(2 *m), 1:(2*m)] <- 0
  W[2, 1] <- 1
  for(i in 3:(2 * m + 1)) {
    if((i %% 2) == 0){
      W[i, 1:(i-1)] <- c(seq(2, i - 1, by = 2), seq(1, i - 1, by = 2))
    } else {
      W[i, 1:(i-1)] <- c(seq(1, i - 1, by = 2), seq(2, i - 1, by = 2))
    }
  }
  W <- cbind(1:total.n, W)

  # # Create the neighbor structure
  # W <- array(0, dim = c(total.n, 2 * n.neighbors))
  # Wnb <- array(10, dim = c(total.n , 3))
  # for(i in 3:total.n) {
  #   if(i <= (2 * m + 1)) {
  #     prev.1 <- seq.1 < i
  #     prev.2 <- seq.2 < i
  #     if(i %in% seq.1) {
  #       W[i, 1:(i-1)] <- c(seq.1[prev.1], seq.2[prev.2])
  #       Wnb[i, 1] <- 1
  #       Wnb[i, 2] <- sum(prev.1)
  #       Wnb[i, 3] <- sum(prev.2)
  #     } else {
  #       W[i, 1:(i-1)] <- c(seq.2[prev.2], seq.1[prev.1])
  #       Wnb[i, 1] <- 2
  #       Wnb[i, 2] <- sum(prev.2)
  #       Wnb[i, 3] <- sum(prev.1)}
  #   } else {
  #     prev.1 <- seq.1 < i
  #     prev.2 <- seq.2 < i
  #     WN.1 <- RANN::nn2(observed.locations.melt[seq.1[prev.1], ],
  #                       observed.locations.melt[i, , drop = FALSE],
  #                       n.neighbors)
  #     WN.2 <- RANN::nn2(observed.locations.melt[seq.2[prev.2], ],
  #                       observed.locations.melt[i, , drop = FALSE],
  #                       n.neighbors)
  #     if(i %in% seq.1) {
  #       Wnb[i, 1] <- 1
  #       W[i, ] <- c(seq.1[WN.1$nn.idx], seq.2[WN.2$nn.idx])
  #     } else {
  #       Wnb[i, 1] <- 2
  #       W[i, ] <- c(seq.2[WN.2$nn.idx], seq.1[WN.1$nn.idx]) }
  #   }
  # }
  # W <- cbind(1:total.n, W)

  
  
  print("Neighbor structure retrieved")
  # Compute a list of the distance matrices
  D.array <- array(0, dim = c(2 * n.neighbors + 1, 2 * n.neighbors + 1, total.n))
  # Y.array <- array(0, dim = c(2 * n.neighbors + 1, 2 * n.neighbors + 1, total.n))
  max.neighbors <- max(table(W[, 2:(2 * n.neighbors + 1)]))
  W2.array <- array(0, dim = c(total.n, max.neighbors))
  W2i.array <- array(0, dim = c(total.n, max.neighbors))
  w2count.array <- rep(0, total.n)
  
  for(x in 3:total.n) {
    
    m.x <- min(2 * n.neighbors + 1, x)
    n.ind <- W[x, 1:m.x]
    
    D.array[1:m.x, 1:m.x, x] <- fields::rdist(observed.locations.melt[n.ind, ], 
                                              observed.locations.melt[n.ind, ])
    
    # Y.array[1:m.x, 1:m.x, x] <- mydata.melt[n.ind] %*% t(mydata.melt[n.ind])
    n.ind2 <- W[x, 2:m.x]
    w2count.array[n.ind2] <- w2count.array[n.ind2] + 1
    pairs <- seq(1:m.x)
    for(i in 1:m.x) {
      W2.array[n.ind2[i], w2count.array[n.ind2[i]]] <- x
      W2i.array[n.ind2[i], w2count.array[n.ind2[i]]] <- pairs[i]
    }
  }
 
 
  
  print("Distance arrays built")

  # Set up initial values
  nu.cur <- starting.values$nu
  tau.cur <- c(0.1, 0.1)
  sig.cur <- starting.values$sigma
  alpha.cur <- starting.values$alpha
  # w.cur <- mydata.melt.wNA
  w.cur <- rep(0.1, total.n)
  phi.cur <- rep(0.1, total.n)
  beta.cur <- rep(0.01, nb.covariates)
  if(nb.covariates == 1) {
    beta.X <- as.matrix(X * beta.cur)
    X <- as.matrix(X)
  } else {
    beta.X <- X %*% beta.cur
  }
  beta.X <- rep(0, total.n)
  gamma.cur <- array(0.1, dim = c(total.n, 2 * n.neighbors))
  
  # Store samples
  posterior.samples <- list("tau" = rep(NA, 2, mcmc.samples),
                            #"sigma" = rep(NA, 3, mcmc.samples),
                            "beta" = array(NA, dim = c(nb.covariates, mcmc.samples)),
                            # "phi" = array(NA, dim = c(total.n, mcmc.samples)),
                            "w" = array(NA, dim = c(total.n, mcmc.samples)),
                            # "gamma" = array(NA, dim = c(total.n, 2 * n.neighbors, mcmc.samples)),
                            "alpha" = rep(NA, mcmc.samples),
                            "nu" = rep(NA, mcmc.samples))#,
                            #"nugget" = rep(NA, mcmc.samples))
  
  samples <- mv_mcmc_loop(alpha.cur, total.n, D.array,
                       w.cur, n.neighbors,
                       W - 1 , tau.cur, phi.cur, gamma.cur, mydata.melt.wNA, beta.X, sig.cur,
                       smoothness, nu.cur, priors$tau.prior,
                       priors$beta.prior, nb.covariates, X, A, mcmc.samples,
                       starting.values$nugget, priors$a.prior, tuning$a.tuning,
                       tuning$nu.tuning, priors$nu.prior, 
                       a.nu.update.interval, seq.1.noNA - 1, seq.2.noNA - 1, Wnb,
                       w2count.array, W2.array - 1, W2i.array - 1, Ynb - 1)
  
  # Store values
  #posterior.samples[["sigma"]] <- samples[3:5,  3, ] 
  # posterior.samples[["phi"]] <- samples[, 2, ]
  posterior.samples[["w"]] <- samples[1:total.n, ] 
  # posterior.samples[["gamma"]] <- samples[, 3:(2 * n.neighbors + 2), ] 
  posterior.samples[["beta"]] <- samples[(total.n + 1):(total.n + nb.covariates), ] 
  posterior.samples[["tau"]] <- samples[(total.n + nb.covariates + 1):(total.n + nb.covariates + 2), ] 
  posterior.samples[["alpha"]] <- samples[3 + total.n + nb.covariates, ]
  posterior.samples[["nu"]] <- samples[4 + total.n + nb.covariates,  ]
  #posterior.samples[["nugget"]] <- samples[8, 3, ]
  
  return(posterior.samples)
}