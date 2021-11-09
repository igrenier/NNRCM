#' =============================================================================
#'
#' PREDICTION FOR THE HIERARCHICAL SPATIAL-T MODEL
#' author: IG
#' date: 04-20-21
#'
#' =============================================================================

NNRCM.hierarchical.predict <- function(Y, X, 
                                              observed.locations, 
                                              predicted.locations,
                                              posterior.samples,
                                              point.estimates,
                                              smoothness,
                                              n.neighbors = 10) {

  # remove location 1 and 2 from the pool of observed location
  observed.locations <- observed.locations[-c(1,2), ]
  Y <- Y[-c(1,2)]
  posterior.samples$w <- posterior.samples$w[-c(1,2), ]
  
  # adjust objects if there is only one covariate
  nb.covariates <- ncol(as.matrix(X))
  if(nb.covariates == 1) {
    X <- as.matrix(X)
    Beta <- t(as.matrix(posterior.samples$beta))
  } else {
    Beta <- posterior.samples$beta
  }
  
  n.pred <- nrow(predicted.locations)
  n.obs <- nrow(observed.locations)
  mcmc.samples <- length(posterior.samples$tau)

  # Obtain distance and neighbor structure
  WN <- RANN::nn2(observed.locations,
                  predicted.locations,
                  n.neighbors)
  
  W.array <- WN$nn.idx
  
  # Compute a list of the distance matrices
  D.array <- array(0, dim = c(n.neighbors + 1, n.neighbors + 1, n.pred))
  Y.array <- array(0, dim = c(n.pred, n.neighbors))
  w.array <- array(0, dim = c(mcmc.samples, n.neighbors, n.pred))
  
  for(x in 1:n.pred) {
    
    n.ind <- W.array[x, ]
    
    D.array[, , x] <- fields::rdist(rbind(predicted.locations[x, ], 
                                          observed.locations[n.ind, ]), 
                                    rbind(predicted.locations[x, ], 
                                          observed.locations[n.ind, ]))
    
    Y.array[x, ] <- Y[n.ind]
    w.array[, , x] <- t(posterior.samples$w[n.ind, ])
    
  }
  
  y.pred <-
    posterior_hierarchical_prediction(n.neighbors, n.pred, D.array, w.array,
                                      point.estimates[1], 
                                      smoothness, point.estimates[3], point.estimates[4], 
                                      posterior.samples$sigma,
                                      posterior.samples$tau, mcmc.samples, n.obs, 
                                      X, Beta
    )
  
  return(y.pred)
}

NNRCM.bv.hierarchical.predict <- function(Y, X = 0, A = diag(2), kappa,
                                              observed.locations, predicted.locations,
                                              posterior.samples, 
                                              point.estimates = list(alpha = 0,
                                                                  sigma = 0,
                                                                  nu = 0,
                                                                  nugget = 0), 
                                              n.neighbors = 10) {
  
  # Check if any covariates are given
  if(X == 0){
    X <- rep(1, nrow(predicted.locations))
  }
  
  if(point.estimates$alpha == 0){
    print("You need to supply point estimates for the parameters")
    stop()
  }
  
  # adjust objects if there is only one covariate
  nb.covariates <- ncol(as.matrix(X))
  if(nb.covariates == 1) {
    X <- as.matrix(X)
    Beta <- t(as.matrix(posterior.samples$beta))
  } else {
    Beta <- posterior.samples$beta
  }
  
  # Melt the data by alternative sources/variate
  n.locations <- nrow(Y)
  total.n <- 2 * n.locations
  
  seq.1 <- seq(1, total.n, by = 2)
  seq.2 <- seq(2, total.n, by = 2)
  
  observed.locations.melt <- matrix(0, nrow = total.n, ncol= 2)
  observed.locations.melt[seq.1, ] <- as.vector(as.matrix(observed.locations))
  observed.locations.melt[seq.2, ] <- as.vector(as.matrix(observed.locations))
  observed.locations.melt <- observed.locations.melt[-c(1,2), ]
  
  observed.locations.melt <- data.frame("Var1" = observed.locations.melt[, 1],
                                        "Var2" = observed.locations.melt[, 2])
  observed.locations <- data.frame("Var1"= observed.locations[, 1],
                                    "Var2" = observed.locations[, 2])
  predicted.locations <- data.frame("Var1"= predicted.locations[, 1],
                                    "Var2" = predicted.locations[, 2])
  
  # remove location 1 and 2 from the pool of observed location
  posterior.samples$w <- posterior.samples$w[-c(1,2), ]
  
  # Extract values
  n.pred <- dim(predicted.locations)[1]
  n.pred2 <- 2 * n.pred
  m <- n.neighbors
  mcmc.samples <- length(posterior.samples$alpha)
  
  # Create the neighbor structure for the predictions
  W <- array(0, dim = c(n.pred, 2 * n.neighbors))
  Wnb <- array(n.neighbors, dim = c(n.pred2 , 3))
  Wnb[seq(1, n.pred2, by = 2), 1] <- 1
  Wnb[seq(2, n.pred2, by = 2), 1] <- 2
  WN <- RANN::nn2(observed.locations[-1, ],
                    predicted.locations,
                    n.neighbors)
  
  W[, 1:n.neighbors] <- 2 * WN$nn.idx - 1
  W[, (n.neighbors + 1):(2 * n.neighbors)] <- 2 * WN$nn.idx
  # for(i in 1:n.pred) {
  #   WN.1 <- RANN::nn2(observed.locations[-1, ],
  #                     predicted.locations[i, ],
  #                     n.neighbors)
  #   WN.2 <- RANN::nn2(observed.locations[-1, ],
  #                     predicted.locations[i, ],
  #                     n.neighbors)
  #   Wnb[2 * i - 1, 1] <- 1
  #   Wnb[2 * i, 1] <- 2
  #   W[2 * i - 1, ] <- c(seq.1[WN.1$nn.idx], seq.2[WN.2$nn.idx]) 
  #   W[2 * i, ] <- c(seq.2[WN.1$nn.idx], seq.1[WN.2$nn.idx])
  # } 
  
  print("Neighbor structure retrieved")
  # Compute a list of the distance matrices
  D.array <- array(0, dim = c(2 * n.neighbors + 1, 2 * n.neighbors + 1, n.pred2))
  # Y.array <- array(0, dim = c(n.pred2, 2 * n.neighbors))
  w.array <- array(0, dim = c(mcmc.samples, 2 * n.neighbors, n.pred2))
  
  for(x in 1:n.pred) {
    
    n.ind <- W[x, ]
    distance <- fields::rdist(rbind(predicted.locations[x, ],
                                    observed.locations.melt[n.ind, ]),
                              rbind(predicted.locations[x, ],
                                    observed.locations.melt[n.ind, ]))
    
    D.array[, , 2 * x - 1] <- distance
    D.array[, , 2 * x] <- distance
    # Y.array[2 * x - 1, ] <- mydata.melt[n.ind] 
    w.array[, , 2 * x - 1] <- t(posterior.samples$w[n.ind, ])
    
    n.ind <- W[x, c((n.neighbors + 1):(2 * n.neighbors), 1:n.neighbors)]
    
    # D.array[, , 2 * x] <- fields::rdist(rbind(predicted.locations[x, ],
    #                                           observed.locations.melt[n.ind, ]),
    #                                     rbind(predicted.locations[x, ],
    #                                           observed.locations.melt[n.ind, ]))
    # 
    # Y.array[2 * x, ] <- mydata.melt[n.ind] 
    w.array[, , 2 * x] <- t(posterior.samples$w[n.ind, ])
    
  }
  
  print("Distance arrays built")
  print("Starting c++...")

  # y.pred <- array(NA, dim = c(n.pred2, mcmc.samples))
  y.pred <-
    as.array(mv_posterior_hierarchical_prediction(n.neighbors, n.pred2, D.array, w.array, point.estimates$alpha, 
                                         kappa, point.estimates$nu, point.estimates$nugget, point.estimates$sigma,
                                         posterior.samples$tau, mcmc.samples, total.n, X, Beta,
                                         Wnb, A
    ))

  seq.odd <- seq(1, n.pred2, by = 2)
  seq.even <- seq(2, n.pred2, by = 2)
  posterior.predictive.Y.array <- array(0, dim=c(n.pred, mcmc.samples, 2))
  posterior.predictive.Y.array[,,1] <- y.pred[seq.odd, ]
  posterior.predictive.Y.array[,,2] <- y.pred[seq.even, ]
  
  
  return(posterior.predictive.Y.array)
}
