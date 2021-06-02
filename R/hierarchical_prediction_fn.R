#' =============================================================================
#'
#' PREDICTION FOR THE HIERARCHICAL SPATIAL-T MODEL
#' author: IG
#' date: 04-20-21
#'
#' =============================================================================

NIGP.hierarchical.prediction.rcpp <- function(alpha, n.neighbors, y, cov.pars, cov.family, kappa,
                                              prior.nugget, observed.locations, predicted.locations,
                                              posterior.samples, fixed.neighborhood = FALSE) {

  n.pred <- nrow(predicted.locations)
  n.obs <- nrow(observed.locations)
  mcmc.samples <- length(posterior.samples$tau)

  distance <- as.matrix(fields::rdist(predicted.locations, observed.locations))

  W <- create.W.neighbors.predictive.matrix(distance, n.neighbors, fixed.neighborhood)

  y.pred <- array(0, dim = c(n.pred, mcmc.samples))

  for(x in 1:n.pred) {
    if(x %% 100 == 0){print(x)}
    n.ind <- W[x, ]!=0
    Y.neighbors <- y[n.ind]
    distance.neighbors <- as.matrix(dist(rbind(predicted.locations[x, ], observed.locations[n.ind, ])))

    C <- geoR::cov.spatial(distance.neighbors,
                           cov.model = cov.family,
                           cov.pars = cov.pars,
                           kappa = kappa) + prior.nugget * diag(n.neighbors + 1)

    cov.C <- C[2:(n.neighbors + 1), 2:(n.neighbors + 1)]
    solve.C <- solve(cov.C)
    vec.C <- C[1, 2:(n.neighbors + 1)]
    cond.C <- C[1,1] - t(vec.C) %*% solve.C %*% vec.C

    # sample phi
    phi <- rinvgamma(mcmc.samples, alpha - n.obs + 1 + n.neighbors, (alpha - n.obs - 1) * cond.C)

    # sample gamma
    gamma <- do.call("rbind", lapply(phi, function(p) mvrnorm(1,
                                                              solve.C %*% vec.C,
                                                              p / (alpha - n.obs - 1) * solve.C)))

    # sample w
    w <- rnorm(mcmc.samples,
               posterior.samples$beta + rowSums(gamma * (posterior.samples$w[, n.ind])),
               sqrt(phi * posterior.samples$sigma))

    # sample y
    y.pred[x, ] <- rnorm(mcmc.samples, w, sqrt(posterior.samples$tau))
  }

  return(y.pred)
}
