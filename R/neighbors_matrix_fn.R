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
#' @return (list) the function returns the output of the spConjNNGP model
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
  
  return(model)
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
create.W.neighbors.predictive.matrix <- function(Y,
                                                 observed.locations, 
                                                 predicted.locations,
                                                 n.neighbors) {
  # Create the output matrices 
  n.pred <- dim(predicted.locations)[1]
  D.array <- array(0, dim = c(n.neighbors + 1, n.neighbors + 1, n.pred))
  Y.array <- array(0, dim = c(n.pred, n.neighbors))
  W.array <- matrix(0, nrow = n.pred, ncol = n.neighbors)
  
  # Fill out the indexes
  for(x in 1:n.pred){
    x.dist <- fields::rdist(predicted.locations[x, ], observed.locations)
    lst.neighbors <- order(x.dist)[1:n.neighbors]
    D.array[, , x] <- fields::rdist(rbind(predicted.locations[x, ],
                                          observed.locations[lst.neighbors, ]),
                                    rbind(predicted.locations[x, ],
                                          observed.locations[lst.neighbors, ]))
    Y.array[x, ] <- Y[lst.neighbors]
    W.array[x, ] <- lst.neighbors
  }
  
  return(list(D.array, Y.array, W.array))
}

