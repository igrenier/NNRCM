create.W.neighbors.predictive.matrix <- function(matrix.distance, n.neighbors) {
  n.obs <- dim(matrix.distance)[2]
  n.pred <- dim(matrix.distance)[1]
  W.dist <- matrix.distance
  W.neighbors <- matrix(0, nrow = n.pred, ncol = n.obs)
  
  for(x in 1:n.pred){
    lst.neighbors <- order(W.dist[x, ])[1:n.neighbors] 
    W.neighbors[x, lst.neighbors] <- rep(1, n.neighbors)
  }
  
  return(W.neighbors)
}


NIGP.posterior.predictive <- function(Y, observed.locations, predicted.locations, 
                                      cov.pars, cov.family, n.sim, n.neighbors, 
                                      smoothness, alpha, hierarchical = FALSE) {
  
  predicted.locations <- as.matrix(predicted.locations)
  
  n.obs <- length(Y)
  n.pred <- nrow(predicted.locations)
  
  # posterior
  distance <- as.matrix(fields::rdist(predicted.locations, observed.locations))
  
  W <- create.W.neighbors.predictive.matrix(distance, n.neighbors)
  posterior.predictive.Y <- array(0, dim = c(n.pred, n.sim))
  
  for (x in 1:n.pred) {
    if((x %% 10) == 0) {print(x)}
    n.ind <- W[x, ]!=0
    Y.neighbors <- Y[n.ind]
    
    distance.neighbors <- as.matrix(dist(rbind(predicted.locations[x, ], observed.locations[seq(1,n.obs)[n.ind], ])))
    
    C <- geoR::cov.spatial(distance.neighbors,
                           cov.model = cov.family,
                           cov.pars = cov.pars[1:2],
                           kappa = smoothness)
    C <- C + cov.pars[3] * diag(1 + sum(n.ind))
    # print(distance.neighbors)
    C.posterior.neighbors <- C[2:(n.neighbors + 1), 2:(n.neighbors + 1)] * (alpha - n.obs - 1) # + Y.neighbors %*% t(Y.neighbors)
    
    # covariance
    C.posterior <- C * (alpha - n.obs - 1)
    C.posterior[2:(n.neighbors + 1), 2:(n.neighbors + 1)] <- C.posterior.neighbors
    
    # compute C's
    C.neighbors.vec <- C.posterior[1, 2:(n.neighbors + 1)]
    C.neighbors <- C.posterior[2:(n.neighbors + 1), 2:(n.neighbors + 1)]
    C.neighbors.inv <- qr.solve(C.neighbors)
    C.neighbors.cond <- C.posterior[1,1] - t(C.neighbors.vec) %*% C.neighbors.inv %*% C.neighbors.vec
    
    # Sample phi
    phi.array <- rinvgamma(n.sim, alpha + n.neighbors, C.neighbors.cond)
    
    # Sample gamma
    gamma.array <- sapply(phi.array, function(p) mvrnorm(1, 
                                                         C.neighbors.inv %*% C.neighbors.vec, 
                                                         p * C.neighbors.inv,
                                                         tol = 1e-01))
    
    # sample prediction
    posterior.predictive.mean <- t(gamma.array) %*% Y.neighbors
    posterior.predictive.sd <- sqrt(phi.array)
    posterior.predictive.Y[x, ] <- rnorm(n.sim, posterior.predictive.mean, posterior.predictive.sd)
  }
  
  return(posterior.predictive.Y)
}