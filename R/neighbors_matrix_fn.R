#' =============================================================================
#'
#' NEIGHBOR MATRICES CREATION FUNCTIONS
#' author: IG
#' date: 04-20-21
#'
#' =============================================================================

#' neighbor matrix for osberved locations using spConjNNGP
#' 
#' creates matrix of neighbors for observations. The neighborhood of a location
#' "s" must only contain locations with smaller indexes. That is, the order of
#'  the rows of the data matrix matters.
#' 
#' @param matrix.locations (matrix) observed locations (n.obs X 2)
#' @param n.neighbors (integer) number of neighbors to include in N(s)
#' @return (list) the function returns a list (n.obs X n.neighbors) of indexes to
#'    indicate which observed location is a neighbor for each observed location 
create.W.neighbors.matrix.spConjNNGP <- function(matrix.locations, n.neighbors) {
  
  # Create fake data
  fake.df <- data.frame("y" = rep(1, dim(matrix.locations)[1]))
  theta.alpha <- list("phi" = 1, "nu" = 1, "alpha" = 1)
  
  model <- spNNGP::spConjNNGP(formula = y~1, data = fake.df, 
                              coords = matrix.locations,
                              n.neighbors = n.neighbors, cov.model = "matern",
                              theta.alpha = theta.alpha,
                              sigma.sq.IG = c(3, 1),
                              n.omp.threads = 12,
                              return.neighbor.info = TRUE)
  
  return(model$neighbor.info$n.indx)
}

#' neighbor matrix for observed locations
#'
#' creates matrix of neighbors for observations. The neighborhood of a location
#' "s" must only contain locations with smaller indexes. That is, the order of
#'  the rows of the data matrix matters.
#'
#' @param matrix.locations (matrix) observed locations (n.obs X 2)
#' @param n.neighbors (integer) number of neighbors to include in N(s)
#' @param fixed.neighbors (logical) set of 5 fixed neighbors for all locations
#'    (default: FALSE)
#' @return (matrix) the function returns a matrix (n.obs X n.neighbors) of indexes to
#'    indicate which observed location is a neighbor for each observed location
create.W.neighbors.matrix <- function(matrix.locations, n.neighbors,
                                      fixed.neighbors = FALSE) {

  n.obs <- dim(matrix.locations)[1]
  W.neighbors <- matrix(0, nrow = n.obs, ncol = n.neighbors)

  if (!fixed.neighbors) {
    for(x in 2:n.obs){
      if(x <= n.neighbors) {
        W.neighbors[x, 1:(x- 1)] <- seq(1, x - 1)
      } else {
        x.dist <- fields::rdist(matrix.locations[x, ], matrix.locations[1:(x - 1), ])
        lst.neighbors <- order(x.dist)[1:n.neighbors]
        W.neighbors[x, ] <- sort(lst.neighbors)
      }
    }
  } else {
    for(x in 2:n.obs){
      if(x <= n.neighbors) {
        W.neighbors[x, 1:(x- 1)] <- seq(1, x - 1)
      } else {
        x.dist <- fields::rdist(matrix.locations[x, ], matrix.locations[6:(x - 1), ])
        lst.neighbors <- order(x.dist)[1:(n.neighbors - 5)] + 5
        W.neighbors[x, ] <- c(seq(1,5), sort(lst.neighbors))
        
      }
    }
  }

  return(W.neighbors)
}

#' neighbor matrix for predicted locations
#'
#' creates matrix with observed neighbors index for each prediction location
#'
#' @param matrix.distance (matrix) observed distance between the prediction and observed
#'    locations (n.pred X n.obs)
#' @param n.neighbors (integer) number of neighbors to include in N(s)
#' @param fixed.neighbors (logical) set of 5 fixed neighbors for all locations
#'    (default: FALSE)
#' @return (matrix) the function returns a matrix (n.pred X n.obs) of logical to
#'    indicate which observed location is a neighbor for each predicted location
create.W.neighbors.predictive.matrix <- function(matrix.distance, n.neighbors,
                                                 fixed.neighbors = FALSE) {
  n.obs <- dim(matrix.distance)[2]
  n.pred <- dim(matrix.distance)[1]
  W.dist <- matrix.distance
  W.neighbors <- matrix(0, nrow = n.pred, ncol = n.obs)

  if(!fixed.neighbors) {
    for(x in 1:n.pred){
      lst.neighbors <- order(W.dist[x, ])[1:n.neighbors]
      W.neighbors[x, lst.neighbors] <- rep(1, n.neighbors)
    }
  } else {
    for(x in 1:n.pred){
      lst.neighbors <- c(seq(1, 5), order(W.dist[x, 6:n.obs])[1:(n.neighbors - 5)] + 5)
      W.neighbors[x, lst.neighbors] <- rep(1, n.neighbors)
    }
  }

  return(W.neighbors)
}

