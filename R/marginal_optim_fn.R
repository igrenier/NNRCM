#' =============================================================================
#'
#' OPTIMIZATION OF THE MARGINAL SPATIAL-T MODEL
#' author: IG
#' date: 04-28-21
#'
#' =============================================================================


#' Functions to combine lists obtained from doPar
add <- function(x) Reduce("+", x)

comb <- function(x, ...) {
  lapply(seq_along(x),
         function(i) c(x[[i]], lapply(list(...), function(y) y[[i]])))
}

#' conditional covariance matrix
#'
#' compute the conditional covariance of s given its neighbors N(s)
#'
#' @param C (matrix) full covariance matrix for the dataset
#' @param m (integer) number of neighbors to include in N(s)
#' @return (numeric) the function computes and returns the conditional value of
#'                   the covariance of s | N(s).
conditional.covariance <- function(C, m) {
  C.cond <- C[1, 1] - C[1, 2:(m+1)] %*% solve(C[2:(m+1), 2:(m+1)]) %*% C[2:(m+1), 1]
  return(C.cond)
}

#' posterior marginal likelihood
#'
#' compute the value of the posterior marginal likelihood
#'
#' @param pars (vector)
#' @param Y (vector) response value for our dataset
#' @param D (matrix) observed distance between all locations (n.obs X n.obs)
#' @param smallest.distance (numeric) smallest observed distance between two 
#'  locations in D
#' @param W (matrix) matrix where each row s identifies the members of N(s)
#' @param n.obs (integer) number of observations in our dataset
#' @param m (integer) number of neighbors to include in N(s)
#' @param kappa (numeric) smoothness parameter for the Matern covariance function
#' @return (numeric) the function computes and returns the value of the
#'                   posterior marginal likelihood for a given set of parameters
marginal.likelihood.optim <- function(pars, Y, D, Y_post, smallest.distance, W, n.obs, 
                                      m, kappa, cov.family) {

  # Name the parameters for ease of understanding
  alpha <- as.numeric(pars[1])
  sig <- as.numeric(pars[2])
  nu <- as.numeric(pars[3])
  xi <- as.numeric(pars[4])

  # try coding it using rcpp:
  marginal <- posterior_marginal(m, n.obs, D, Y_post,
                                 alpha, kappa, Y, W, nu, sig, xi, smallest.distance)
  
  return(marginal)
}

#' optim wrapper for marginal likelihood
#'
#' optimizes the posterior marginal likelihood
#'
#' @param mydata (vector) response value for our dataset
#' @param observed.distance (matrix) observed distance between all locations (n.obs X n.obs)
#' @param n.neighbors (integer) number of neighbors to include in N(s)
#' @param smoothness (numeric) smoothness parameter for the Matern covariance function
#' @param cov.family (string) name of the covariance function (e.g. "matern")
#' @param starting.values (vector) starting values for the optimization, must be of
#'   length four in the following order: alpha, sigma, phi, xi.
#' @return (list) the function returns a list with the output from the R::optim function
#'
#' @export
NIGP.marginal.optim.fn <- function(mydata, observed.locations, n.neighbors,
                                   smoothness, cov.family,
                                   starting.values) {

  # Extract values
  n.obs <- length(mydata)
  m <- n.neighbors
  smallest.distance <- min(fields::rdist(observed.locations[1, ], observed.locations[2:n.obs,]))
  
  # Create the neighbor structure
  W.list <- create.W.neighbors.matrix.spConjNNGP(observed.locations, n.neighbors)
  W <- array(0, dim = c(n.obs, n.neighbors))
  for(i in 2:(m+1)) {
    W[i, 1:(i-1)] <- seq(1, i - 1) 
  }
  W[(m+2):n.obs, ] <- matrix(unlist(W.list)[(m * (m+1) / 2 + 2) : ((n.obs - m - 1) * m + m * (m+1) / 2 + 1)], ncol = 10, byrow = TRUE)
  W <- cbind(1:n.obs, W)
  
  print("I made it past 1")
  # Compute a list of the distance matrices
  D.array <- array(0, dim = c(n.neighbors + 1, n.neighbors + 1, n.obs))
  Y.array <- array(0, dim = c(n.neighbors + 1, n.neighbors + 1, n.obs))
  
  for(x in 2:n.obs) {
    
    n.ind <- W.list[[x]]
    m.x <- min(n.neighbors + 1, x)
    
    D.array[1:m.x, 1:m.x, x] <- fields::rdist(observed.locations[c(x, n.ind), ], 
                                    observed.locations[c(x, n.ind), ])
    
    Y.array[1:m.x, 1:m.x, x] <- mydata[c(x, n.ind)] %*% t(mydata[c(x, n.ind)])
    
  }
  print("I made it past 2")
  # Fit the optim function
  # cl <- makeCluster(12)
  # registerDoParallel(cl)
  optim.results <- 
    optim(par = starting.values,
                   fn = marginal.likelihood.optim,
                   Y = mydata,
                   D = D.array,
                   Y_post = Y.array,
                   smallest.distance = smallest.distance,
                   m = n.neighbors,
                   W = W - 1,
                   n.obs = n.obs,
                   kappa = smoothness,
                   cov.family = cov.family,
                   method = "L-BFGS-B",
                   lower = c(n.obs + 2, 0.01, 0.01, 0.0001),
                   upper = c(Inf, Inf, Inf, Inf),
                   control = list(fnscale = -1, maxit=10))
  print("I made it past 3")
  # stopCluster(cl)
  return(optim.results)

}

#' posterior prediction function
#'
#' predicts the response values for the requested set of predicted locations.
#'
#' @param Y (vector) response value for our dataset
#' @param observed.locations (matrix) observed locations (n.obs X 2)
#' @param predicted.locations (matrix) predicted locations (n.pred X 2)
#' @param cov.pars (vector) vector of length two where the first element is the partial
#    sill (sigma) and the second element is the range (nu)
#' @param cov.family (string) name of the covariance function (e.g. "matern")
#' @param n.neighbors (integer) number of neighbors to include in N(s)
#' @param smoothness (numeric) smoothness parameter for the Matern covariance function
#' @param alpha (numeric) degrees of freedom of the Wishart distribution
#' @return (list) the function returns a list with the output from the R::optim function
#' @export
NIGP.posterior.predictive <- function(Y, observed.locations, predicted.locations,
                                      cov.pars, cov.family, n.sim, n.neighbors,
                                      smoothness, alpha) {

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
                           cov.pars = c(cov.pars["sigma"], cov.pars["nu"]),
                           kappa = smoothness)
    C <- C + cov.pars["tau"] * diag(1 + sum(n.ind))

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

# # Create list C
# COVARIANCE <- foreach(x = 1:n.obs,.packages = "geoR",.combine='comb',
#                    .multicombine=TRUE,.init=list(list(), list())) %dopar%{
#   
#      V.x <- array(0, dim = c(m + 1, m + 1))
#      V.x.Y <- array(0, dim = c(m + 1, m + 1))
#                      
#     if(x == 1){
#       V.x[1,1] <- xi
#       V.x.Y[1,1] <- xi + Y[x]^2
#     } else {
#       
#       m.x <- min(m + 1, x)
#       n.ind <- W[[x]]
#       
#       V.x[1:m.x, 1:m.x] <- (alpha - n.obs - 1) * (geoR::cov.spatial(D[1:m.x,1:m.x, x],
#                                                       cov.model = cov.family,
#                                                       cov.pars = c(sig, nu),
#                                                       kappa = kappa) + xi * diag(m.x))
#       
#       V.x.Y[1:m.x, 1:m.x] <- V.x[1:m.x, 1:m.x] + Y[c(x, n.ind)] %*% t(Y[c(x, n.ind)])
#      
#     }
#      
#     list(V.x, V.x.Y)
#                      
# }

# V.x <- array(as.numeric(unlist(COVARIANCE[[1]])), dim=c(m + 1, m + 1, n.obs))
# V.x.Y <- array(as.numeric(unlist(COVARIANCE[[2]])), dim=c(m + 1, m + 1, n.obs))

