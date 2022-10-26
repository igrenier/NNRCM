#' =============================================================================
#'
#' OPTIMIZATION OF THE MARGINAL SPATIAL-T MODEL
#' author: IG
#' date: 04-28-21
#'
#' =============================================================================

#' posterior marginal likelihood
#'
#' compute the value of the posterior marginal likelihood
#'
#' @param pars (vector) of length four with the parameter values in the following 
#'   order: alpha, sigma, phi, xi
#' @param Y (vector) response value for our dataset
#' @param D (array) observed distance between each locations and its neighbors
#'   (n.neighbors + 1 X n.neighbors + 1 X n.obs)
#' @param smallest.distance (numeric) approximate smallest observed distance between two 
#'  locations. This is used as a hyperparameter for the prior on the range parameter
#' @param Y.post (array) contains the matrix Y_{s, N(s)}'Y_{s, N(s)} for each s
#'   (n.neighbors + 1 X n.neighbors + 1 X n.obs)
#' @param W (matrix) matrix where each row s identifies the members of N(s)
#' @param n.obs (integer) number of observations in our dataset
#' @param m (integer) number of neighbors to include in N(s)
#' @param kappa (numeric) smoothness parameter for the Matern covariance function
#' @return (numeric) the function computes and returns the value of the
#'                   posterior marginal likelihood for a given set of parameters
marginal.likelihood.optim <- function(pars, Y, D, Y_post, smallest.distance, W, n.obs, 
                                      m, kappa, 
                                      prior.list = NULL
                                     ) {

  # Name the parameters for ease of understanding
  alpha <- as.numeric(pars[1])
  sig <- as.numeric(pars[2])
  phi <- as.numeric(pars[3])
  tau <- as.numeric(pars[4])
  
  # Set Value for prior parameters of sigma^2 and tau^2
  if(is.null(prior.list)){
    prior.list <- list(a_sig = 3, b_sig = 2, a_tau = 3, b_tau = 2)}
  a_sig <- prior.list$a_sig
  b_sig <- prior.list$b_sig
  a_tau <- prior.list$a_tau
  b_tau <- prior.list$b_tau
  
  # try coding it using rcpp:
  marginal <- posterior_marginal(m, n.obs, D, Y_post,
                                 alpha, kappa, Y, W, phi, 
                                 sig, tau, smallest.distance,
                                a_sig, b_sig, a_tau, b_tau)
  
  return(marginal)
}

#' optim wrapper for marginal likelihood
#'
#' optimizes the posterior marginal likelihood
#'
#' @param Y (vector) response value for our dataset
#' @param observed.locations (matrix) observed locations for each observation 
#'   of mydata (n.obs X 2)
#' @param smoothness (numeric) smoothness parameter for the Matern covariance function
#' @param n.neighbors (integer) number of neighbors to include in N(s) (default: 10)
#' @param starting.values (vector) starting values for the optimization, must be of
#'   length four in the following order: alpha, sigma, phi, xi
#'   (default: (0, 0.5, 0.5, 0.5))
#' @return (list) the function returns a list with the output from the R::optim 
#'   function
#' @export
NNRCM.marginal.infer <- function(Y, observed.locations, smoothness, 
                                   n.neighbors = 10, 
                                   starting.values = c(0, 0.5, 0.5, 0.5)) {

  # Extract values
  n.obs <- length(Y)
  m <- n.neighbors
  distance.first <- fields::rdist(observed.locations[1:2, ], observed.locations[3:n.obs,])
  smallest.distance <- min(distance.first[distance.first !=0])
  
  # check that the degrees of freedom are in the correct range:
  if(starting.values[1] < n.obs + 2) {
    print(paste0("The starting values for the degrees of freedom were set to ", n.obs +2))
    starting.values[1] <- n.obs + 2
  }
  
  # Create the neighbor structure
  model <- create.W.neighbors.matrix.spConjNNGP(observed.locations, m)
  W.list <- model$neighbor.info$n.indx
  W <- array(0, dim = c(n.obs, m))
  for(i in 2:(m+1)) {
    W[i, 1:(i-1)] <- seq(1, i - 1) 
  }
  W[(m+2):n.obs, ] <- matrix(unlist(W.list)[(m * (m+1) / 2 + 2) : ((n.obs - m - 1) * m + m * (m+1) / 2 + 1)], ncol = m, byrow = TRUE)
  W <- cbind(1:n.obs, W)
  
  print("Neighbor structure retrieved")
  # Compute a list of the distance matrices
  D.array <- array(0, dim = c(n.neighbors + 1, n.neighbors + 1, n.obs))
  Y.array <- array(0, dim = c(n.neighbors + 1, n.neighbors + 1, n.obs))
  
  for(x in 2:n.obs) {
    
    n.ind <- W.list[[x]]
    m.x <- min(n.neighbors + 1, x)
    
    D.array[1:m.x, 1:m.x, x] <- fields::rdist(observed.locations[c(x, n.ind), ], 
                                    observed.locations[c(x, n.ind), ])
    
    Y.array[1:m.x, 1:m.x, x] <- Y[c(x, n.ind)] %*% t(Y[c(x, n.ind)])
    
  }
  print("Distance arrays built")

  optim.results <- 
    optim(par = starting.values,
                   fn = marginal.likelihood.optim,
                   Y = Y,
                   D = D.array,
                   Y_post = Y.array,
                   smallest.distance = smallest.distance,
                   m = n.neighbors,
                   W = W - 1,
                   n.obs = n.obs,
                   kappa = smoothness,
                   method = "L-BFGS-B",
                   lower = c(n.obs + 2, 0.0001, 0.0001, 0.0001),
                   upper = c(Inf, Inf, Inf, Inf),
                   control = list(fnscale = -1))
  print("Optimization completed")

  return(optim.results)

}

#' posterior prediction function
#'
#' predicts the response values for the requested set of predicted locations.
#'
#' @param Y (vector) response value for our dataset
#' @param observed.locations (matrix) observed locations (n.obs X 2)
#' @param predicted.locations (matrix) predicted locations (n.pred X 2)
#' @param optim.pars (vector) vector of length four in the following order: 
#'   alpha, sigma, phi, tau
#' @param smoothness (numeric) smoothness parameter for the Matern covariance function
#' @param n.neighbors (integer) number of neighbors to include in N(s) (default: 10)
#' @param n.sim (integer) prediction sample size to output (default: 1000)
#' @return (matrix) the function returns a matrix where each row is the prediction
#'   samples for each predicted location.
#' @export
NNRCM.marginal.predict <- function(Y, observed.locations, predicted.locations,
                                   optim.pars, smoothness, n.neighbors = 10,
                                   n.sim = 1000) {

  
  # Extract values
  n.obs <- length(Y)
  n.pred <- nrow(predicted.locations)

  # Obtain distance and neighbor structure
  WN <- RANN::nn2(observed.locations,
            predicted.locations,
            n.neighbors)

  W.array <- WN$nn.idx
  
  # Compute a list of the distance matrices
  D.array <- array(0, dim = c(n.neighbors + 1, n.neighbors + 1, n.pred))
  Y.array <- array(0, dim = c(n.pred, n.neighbors))
  
  for(x in 1:n.pred) {
    
    n.ind <- W.array[x, ]
    
    D.array[, , x] <- fields::rdist(rbind(predicted.locations[x, ], 
                                          observed.locations[n.ind, ]), 
                                    rbind(predicted.locations[x, ], 
                                          observed.locations[n.ind, ]))
    
    Y.array[x, ] <- Y[n.ind]
    
  }
  
  print("Distance arrays built")
  print("Starting c++...")
  
  posterior.predictive.Y <- 
    posterior_marginal_prediction(n.neighbors, n.pred, D.array, Y.array,
                                  optim.pars[1],
                                  smoothness, optim.pars[3], optim.pars[2],
                                  optim.pars[4], n.sim, n.obs)

  return(posterior.predictive.Y)
  
}

#' mv posterior marginal likelihood
#'
#' compute the value of the bivariate posterior marginal likelihood
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
mv.marginal.likelihood.optim <- function(pars, Y, D, Y_post, smallest.distance, W, Wnb, n.obs, 
                                         m, kappa, cov.family, parsimonious , 
                                         coregionalization) {
  
  # Name the parameters for ease of understanding
  alpha <- as.numeric(pars[1])
  sig <- as.numeric(pars[2:4])
  xi <- as.numeric(pars[7:8]) 
  
  if(coregionalization) {
    
    # compute the posterior marginal value
    A <- sig
    nu <- c(pars[5:6], 0)
    marginal <- mv_posterior_marginal_coregionalization(m, n.obs, D, Y_post,
                                      alpha, kappa, Y, W, Wnb, nu, 
                                      A, xi, smallest.distance)
    
  } else {
    if(parsimonious ){
      nu <- rep(as.numeric(pars[5]), 3)
      rho <- sig[3] * (sqrt(kappa[1] * kappa[2]) / mean(c(kappa[1], kappa[2])))
      sig[3] <- sqrt(sig[1] * sig[2]) * rho
    } else {
      nu <- c(pars[5:6], 0)
      nu[3] <- sqrt(0.5 * (nu[1]^2 + nu[2]^2))
      rho <- sig[3] * (nu[1]^kappa[1] * nu[2]^kappa[2]) / (nu[3]^(2 * kappa[3])) *
        gamma(kappa[3]) / sqrt(gamma(kappa[1]) * gamma(kappa[2]))
      sig[3] <- sqrt(sig[1] * sig[2]) * rho
    }
    
    # compute the posterior marginal value
    marginal <- mv_posterior_marginal(m, n.obs, D, Y_post,
                                      alpha, kappa, Y, W, Wnb, nu, 
                                      sig, xi, smallest.distance)
  }


  
  return(marginal)
}

#' bivariate optim wrapper for marginal likelihood
#'
#' optimizes the posterior marginal likelihood of bivariate datasets
#'
#' @param mydata (matrix) bivariate response value for our dataset. Use NA to represent missing data
#' @param observed.locations (matrix) observed locations of bivariate data
#' @param n.neighbors (integer) number of neighbors to include in N(s)
#' @param smoothness (vector) vector of length three (v_{11}, v_r, v_{22}). The first and last terms are the
#'   the Matern smoothness parameters of the responses. The middle term is the a number
#'   between 1 and infinity which is used to compute v_{12}: 
#'   v_{12} = v_r / 2 * (v_{11} + v_{22})
#' @param cov.family (string) name of the covariance function (e.g. "matern")
#' @param starting.values (vector) starting values for the optimization, must be of
#'   length four in the following order: alpha, sigma, phi, xi.
#' @return (list) the function returns a list with the output from the R::optim function
#'
#' @export
NNRCM.bv.marginal.infer <- function(Y, observed.locations, 
                                      smoothness, 
                                      starting.values = rep(0, 8), parsimonious = TRUE, 
                                      mcmc.samples = 1000, n.neighbors = 10,
                                      coregionalization = FALSE, neighbor.info = FALSE,
                                      neighbor.seq = FALSE) {
  
  # add v_r between 1 and infinity check
  
  # Adjust the starting values if they are not improper
  if(starting.values[1] == 0){
    starting.values[1] <- dim(Y)[1] * 2 + 20
    starting.values[2:8] <- rep(0.5, 7)
  }
  
  # Add the smoothness cross-term
  smoothness<- c(smoothness[1], smoothness[2], mean(smoothness[1:2]))
  
  # Create the two objects with the separate coordinates and responses
  mydata.1.available <- !is.na(Y[, 1])
  mydata.1 <- Y[mydata.1.available, 1]
  observed.locations.1 <- observed.locations[mydata.1.available, ]
  n.obs.1 <- sum(mydata.1.available)
  mydata.2.available <- !is.na(Y[, 2])
  mydata.2 <- Y[mydata.2.available, 2]
  observed.locations.2 <- observed.locations[mydata.2.available, ]
  n.obs.2 <- sum(mydata.2.available)
  
  # Melt the data by alternative sources/variate
  if(n.obs.1 == n.obs.2) {
    seq.1 <- seq(1, 2 * n.obs.1, by = 2)
    seq.2 <- seq(2, 2 * n.obs.1, by = 2)
  } else if(n.obs.1 > n.obs.2) {
    seq.1 <- c(seq(1, 2 * n.obs.2, by = 2), seq(2 * n.obs.2 + 1, n.obs.2 + n.obs.1, by = 1))
    seq.2 <- seq(2, 2 * n.obs.2, by = 2)
  } else {
    seq.2 <- c(seq(1, 2 * n.obs.1, by = 2), seq(2 * n.obs.1 + 1, n.obs.2 + n.obs.1, by = 1))
    seq.1 <- seq(2, 2 * n.obs.1, by = 2)
  }
  
  mydata.melt <- rep(0, n.obs.1 + n.obs.2)
  mydata.melt[seq.1] <- mydata.1
  mydata.melt[seq.2] <- mydata.2
  observed.locations.melt <- matrix(0, nrow = n.obs.1 + n.obs.2, ncol= 2)
  observed.locations.melt[seq.1, ] <- as.vector(as.matrix(observed.locations.1))
  observed.locations.melt[seq.2, ] <- as.vector(as.matrix(observed.locations.2))
  
  # Extract values
  n.obs <- n.obs.1 + n.obs.2
  m <- n.neighbors
  smallest.distance <- min(fields::rdist(observed.locations.1[1, ], observed.locations.1[2:n.obs.1,]))
  
  # Create the neighbor structure
  W <- array(0, dim = c(n.obs, 2 * n.neighbors))
  Wnb <- array(10, dim = c(n.obs , 3))
  m2 <- 2 * m
  if(neighbor.seq == TRUE) {
    model <- create.W.neighbors.matrix.spConjNNGP(observed.locations.melt, m2)
    W.list <- model$neighbor.info$n.indx
    Wnb[1:(2 * m), 2] <- sort(rep(seq(0, m-1), 2))
    Wnb[1:(2 * m), 3] <- sort(c(seq(0, m-1), seq(1, m-1), m))
    Wnb[seq.1, 1] <- 1
    Wnb[seq.2, 1] <- 2 
    for(i in 2:(m2 + 1)) {
      W[i, 1:(i-1)] <- seq(1, i - 1) 
    }
    W[(m2 + 2):n.obs, ] <- matrix(unlist(W.list)[(m2 * (m2+1) / 2 + 2) : ((n.obs - m2 - 1) * m2 + m2 * (m2+1) / 2 + 1)], ncol = m2, byrow = TRUE)
    # I don't think this works: would need to go over the whole thing to reorder properly.
  } else {
    for(i in 3:n.obs) {
      if(i <= (2 * m + 1)) {
        prev.1 <- seq.1 < i
        prev.2 <- seq.2 < i
        if(i %in% seq.1) {
          W[i, 1:(i-1)] <- c(seq.1[prev.1], seq.2[prev.2])
          Wnb[i, 1] <- 1
          Wnb[i, 2] <- sum(prev.1)
          Wnb[i, 3] <- sum(prev.2)
        } else {
          W[i, 1:(i-1)] <- c(seq.2[prev.2], seq.1[prev.1])
          Wnb[i, 1] <- 2
          Wnb[i, 2] <- sum(prev.2)
          Wnb[i, 3] <- sum(prev.1)}
        
      } else {
        prev.1 <- seq.1 < i
        prev.2 <- seq.2 < i
        WN.1 <- RANN::nn2(observed.locations.melt[seq.1[prev.1], ],
                          observed.locations.melt[i, , drop = FALSE],
                          n.neighbors)
        WN.2 <- RANN::nn2(observed.locations.melt[seq.2[prev.2], ],
                          observed.locations.melt[i, , drop = FALSE],
                          n.neighbors)
        if(i %in% seq.1) {
          Wnb[i, 1] <- 1
          W[i, ] <- c(seq.1[WN.1$nn.idx], seq.2[WN.2$nn.idx]) 
        } else {
          Wnb[i, 1] <- 2
          W[i, ] <- c(seq.2[WN.2$nn.idx], seq.1[WN.1$nn.idx]) }
      }
    }
  }
  
  W <- cbind(1:n.obs, W)
  
  print("Neighbor structure retrieved")
  # Compute a list of the distance matrices
  D.array <- array(0, dim = c(2 * n.neighbors + 1, 2 * n.neighbors + 1, n.obs))
  Y.array <- array(0, dim = c(2 * n.neighbors + 1, 2 * n.neighbors + 1, n.obs))
  
  for(x in 2:n.obs) {
    
    m.x <- min(2 * n.neighbors + 1, x)
    
    # if neighbor sequence is TRUE, then rearrange the neighbors so the sources are ordered
    if(neighbor.seq == TRUE) {
      n.ind <- W[x, 2:m.x]
      n.ind.seq.1 <- intersect(n.ind, seq.1)
      n.ind.seq.2 <- intersect(n.ind, seq.2)
      if(Wnb[x, 1] == 1) {
        Wnb[x, 2] <- length(n.ind.seq.1)
        Wnb[x, 3] <- length(n.ind.seq.2)
        W[x, 2:m.x] <- c(n.ind.seq.1, n.ind.seq.2)
      } else {
        Wnb[x, 2] <- length(n.ind.seq.2)
        Wnb[x, 3] <- length(n.ind.seq.1)
        W[x, 2:m.x] <- c(n.ind.seq.2, n.ind.seq.1)
      }
      
    }
    
    n.ind <- W[x, 1:m.x]
    D.array[1:m.x, 1:m.x, x] <- fields::rdist(observed.locations.melt[n.ind, ], 
                                              observed.locations.melt[n.ind, ])
    
    Y.array[1:m.x, 1:m.x, x] <- mydata.melt[n.ind] %*% t(mydata.melt[n.ind])
    
  }
  
  print("Distance arrays built")
  
  if(coregionalization){
    optim.results <- 
      optim(par = starting.values,
            fn = mv.marginal.likelihood.optim,
            Y = mydata.melt,
            D = D.array,
            Y_post = Y.array,
            smallest.distance = smallest.distance,
            m = n.neighbors,
            W = W - 1,
            Wnb = Wnb,
            n.obs = n.obs,
            kappa = smoothness,
            cov.family = cov.family,
            parsimonious  = parsimonious ,
            coregionalization = TRUE,
            method = "L-BFGS-B",
            lower = c(n.obs + 2, -Inf, -Inf, -Inf, 0.001,0.001, 0.001, 0.001),
            upper = c(Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf),
            control = list(fnscale = -1))
  } else {
    optim.results <- 
      optim(par = starting.values,
            fn = mv.marginal.likelihood.optim,
            Y = mydata.melt,
            D = D.array,
            Y_post = Y.array,
            smallest.distance = smallest.distance,
            m = n.neighbors,
            W = W - 1,
            Wnb = Wnb,
            n.obs = n.obs,
            kappa = smoothness,
            cov.family = cov.family,
            parsimonious  = parsimonious ,
            coregionalization = FALSE,
            method = "L-BFGS-B",
            lower = c(n.obs + 2, 0.001, 0.001, -1, 0.001,0.001, 0.001, 0.001),
            upper = c(Inf, Inf, Inf, 1, Inf, Inf, Inf, Inf),
            control = list(fnscale = -1, maxit = 10))
  }
  
  print("Optimzation completed")
  
  # Adjust the correlation parameter to be within the proper boundaries
  if (coregionalization) {
    # the parameters can be use as is without modification.
  } else {
    if(parsimonious ) {
      optim.results$par[6] <- optim.results$par[5]
      optim.results$par[4] <- optim.results$par[4] * sqrt(smoothness[1] * smoothness[2]) / smoothness[3]
    } else {
      midrange <- sqrt(1/2 * (optim.results$par[5]^2 + optim.results$par[6]^2))
      optim.results$par[4] <- optim.results$par[4] * ((optim.results$par[5])^(smoothness[1]) *
                                                        optim.results$par[6]^(smoothness[2])) /
        (midrange^(2 * smoothness[3])) *
        gamma(smoothness[3]) / sqrt(gamma(smoothness[1]) * gamma(smoothness[2]))
    }
  }
  
  if(neighbor.info) {
    return(list("optim.results" = optim.results,
                "W" = W,
                "Wnb" = Wnb,
                "D" = D.array,
                "YY" = Y.array))
  } else {
    return(optim.results)
  }

  
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
NNRCM.bv.marginal.predict <- function(Y, observed.locations, predicted.locations,
                                      point.estimates, n.sim = 1000, n.neighbors = 10,
                                      smoothness, coregionalization = FALSE, parsimonious  = TRUE) {
  
  # Adjust the cross smoothness parameter
  smoothness<- c(smoothness[1], smoothness[2], mean(smoothness[1:2]))
  
  # Create the two objects with the separate coordinates and responses
  mydata.1.available <- !is.na(Y[, 1])
  mydata.1 <- Y[mydata.1.available, 1]
  observed.locations.1 <- observed.locations[mydata.1.available, ]
  n.obs.1 <- sum(mydata.1.available)
  mydata.2.available <- !is.na(Y[, 2])
  mydata.2 <- Y[mydata.2.available, 2]
  observed.locations.2 <- observed.locations[mydata.2.available, ]
  n.obs.2 <- sum(mydata.2.available)
  n.obs <- n.obs.1 + n.obs.2
  
  mydata.melt <- c(mydata.1, mydata.2)
  seq.1 <- seq(1, n.obs.1)
  seq.2 <- seq(n.obs.1 + 1, n.obs)
  observed.locations.melt <- rbind(observed.locations.1, observed.locations.2)
  predicted.locations <- data.frame("Var1"= predicted.locations[, 1],
                                    "Var2" = predicted.locations[, 2])
  observed.locations.1 <- data.frame("Var1"= observed.locations.1[, 1],
                                    "Var2" = observed.locations.1[, 2])
  
  observed.locations.2 <- data.frame("Var1"= observed.locations.2[, 1],
                                    "Var2" = observed.locations.2[, 2])
  observed.locations.melt <- data.frame("Var1"= observed.locations.melt[, 1],
                                     "Var2" = observed.locations.melt[, 2])
  # Extract values
  n.pred <- dim(predicted.locations)[1]
  n.pred2 <- 2 * n.pred
  m <- n.neighbors
  
  # Create the neighbor structure for the predictions
  W <- array(0, dim = c(n.pred2, 2 * n.neighbors))
  Wnb <- array(10, dim = c(n.pred2 , 3))
  for(i in 1:n.pred) {
      WN.1 <- RANN::nn2(observed.locations.1,
                        predicted.locations[i, ],
                        n.neighbors)
      WN.2 <- RANN::nn2(observed.locations.2,
                        predicted.locations[i, ],
                        n.neighbors)
      Wnb[2 * i - 1, 1] <- 1
      Wnb[2 * i, 1] <- 2
      W[2 * i - 1, ] <- c(seq.1[WN.1$nn.idx], seq.2[WN.2$nn.idx]) 
      W[2 * i, ] <- c(seq.2[WN.2$nn.idx], seq.1[WN.1$nn.idx])
  } 

  print("Neighbor structure retrieved")
  # Compute a list of the distance matrices
  D.array <- array(0, dim = c(2 * n.neighbors + 1, 2 * n.neighbors + 1, n.pred2))
  Y.array <- array(0, dim = c(n.pred2, 2 * n.neighbors))
  
  for(x in 1:n.pred) {

    n.ind <- W[2 * x - 1, ]
    
    D.array[, , 2 * x - 1] <- fields::rdist(rbind(predicted.locations[x, ],
                                            observed.locations.melt[n.ind, ]),
                                            rbind(predicted.locations[x, ],
                                              observed.locations.melt[n.ind, ]))
    
    Y.array[2 * x - 1, ] <- mydata.melt[n.ind] 
    
    n.ind <- W[2 * x, ]
    
    D.array[, , 2 * x] <- fields::rdist(rbind(predicted.locations[x, ],
                                              observed.locations.melt[n.ind, ]),
                                            rbind(predicted.locations[x, ],
                                              observed.locations.melt[n.ind, ]))
    
    Y.array[2 * x, ] <- mydata.melt[n.ind] 
    
  }
  
  print("Distance arrays built")
  print("Starting c++...")
  
  if(coregionalization) {
    posterior.predictive.Y <- 
      mv_posterior_marginal_prediction_coregionalization(n.neighbors, n.pred2, D.array, Y.array,
                                       smoothness, point.estimates, n.sim, n.obs,
                                       Wnb)
  } else {
    posterior.predictive.Y <- 
      mv_posterior_marginal_prediction(n.neighbors, n.pred2, D.array, Y.array,
                                       smoothness, point.estimates, n.sim, n.obs,
                                       Wnb)
  }

  seq.odd <- seq(1, n.pred2, by = 2)
  seq.even <- seq(2, n.pred2, by = 2)
  posterior.predictive.Y.array <- array(0, dim=c(n.pred, n.sim, 2))
  posterior.predictive.Y.array[,,1] <- posterior.predictive.Y[seq.odd, ]
  posterior.predictive.Y.array[,,2] <- posterior.predictive.Y[seq.even, ]
  
  return(posterior.predictive.Y.array)
  
}
