# Vignette libraries:
library(NNRCM)
library(fields)
library(MASS)
library(ggplot2)
library(RandomFields)
library(RConics)
library(spBayes)

# define the parameters
N1 <- 50
N2 <- 50
N <- N1 * N2
n <- N / 2
theta <- c(1, 1)
kappa <- 1/2

# define the locations
set.seed(1)
observed.locations <- expand.grid(seq(0, 10, length.out = N1), 
                                  seq(0, 10, length.out = N2))
observed.distance <- fields::rdist(observed.locations)

# generate the observations
C <- exp(-observed.distance / theta[2])
y.truth <- mvrnorm(1, rep(0, N), C)

# add noise (optional)
y <- y.truth + rnorm(N, 0, 0.5)

# divide the observations into the training and testing sets
training.sample <- sample(1:N, n, replace = FALSE)
training.locations <- observed.locations[training.sample, ]
testing.locations <- observed.locations[-training.sample, ]
training.y <- y[training.sample]
testing.y <- y[-training.sample]

# ==============================================================================
# Univariate Marginal Model
# ==============================================================================

# obtain the point estimates for the marginal model
optim.results <- 
  NNRCM.marginal.infer(Y = training.y,
                       observed.locations = training.locations,
                       smoothness = kappa)

# posterior predictive sampling on the testing set
posterior.predictive.marginal <- 
  NNRCM.marginal.predict(Y = training.y,
                         observed.locations = training.locations, 
                         predicted.locations = testing.locations, 
                         optim.pars = optim.results$par,
                         smoothness = kappa)

# scoring
NNRCM.scores(Y = y.truth[-training.sample], 
             pp.samples = posterior.predictive.marginal,
             k = 1)

# posterior predictive sampling over high resolution surface
predicted.locations <- 
  expand.grid(seq(-0.01, 10.01, length.out = 2 * N1),
              seq(-0.01, 10.01, length.out = 2 * N2))

predicted.surface <- 
  NNRCM.marginal.predict(Y = training.y,
                         observed.locations = training.locations, 
                         predicted.locations = predicted.locations, 
                         optim.pars = optim.results$par,
                         smoothness = kappa)
# plot the true surface
true.df <- data.frame(x = observed.locations[, 1],
                      y = observed.locations[ ,2],
                      z = y.truth)

ggplot2::ggplot(true.df, aes(x = x, y = y, fill = z)) +
  geom_raster(interpolate = T) + 
  scale_fill_gradient2(low = "red", mid = "white", high = "blue",
                       midpoint = 0, space = "Lab", na.value = "grey50",
                       guide = "colourbar", aesthetics = "fill")

# plot the high resolution predictions
predicted.df <- data.frame(x = predicted.locations[, 1],
                           y = predicted.locations[ ,2],
                           z = apply(predicted.surface, 1, mean))

ggplot2::ggplot(predicted.df, aes(x = x, y = y, fill = z)) +
  geom_raster(interpolate = T) + 
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", 
                       midpoint = 0, space = "Lab", na.value = "grey50", 
                       guide = "colourbar", aesthetics = "fill")
ggsave("vignette-uv-marginal-predictions.pdf", width = 6, height = 5, units='in')

# ==============================================================================
# Univariate Hierarchical Model
# ==============================================================================

# define the fixed effects
training.X <- rep(1, n)
testing.X <- rep(1, n)

# inference from the hierarchical model on the training set 
training.samples <-
  NNRCM.hierarchical.infer(Y = training.y,
                           observed.locations = training.locations,
                           X = training.X,
                           starting.values = optim.results$par[1:3],
                           smoothness = kappa,
                           nugget = optim.results$par[4])

# posterior predictive sampling for the testing set
testing.samples <- 
  NNRCM.hierarchical.predict(
    Y = training.y,
    X = testing.X,
    observed.locations = training.locations,
    predicted.locations = testing.locations, 
    posterior.samples = training.samples,
    point.estimates = c(mean(training.samples$alpha),
                        1, 
                        mean(training.samples$nu),
                        optim.results$par[4]),
    smoothness = kappa)

# scoring
NNRCM.scores(Y = y.truth[-training.sample],
             pp.samples = testing.samples,
             k = 1)

# posterior predictive sampling for the high resolution surface
predicted.X <- rep(1, 4 * N1 * N2)

predicted.surface.2 <- 
  NNRCM.hierarchical.predict(
    Y = training.y,
    X = predicted.X,
    observed.locations = training.locations,
    predicted.locations = predicted.locations, 
    posterior.samples = training.samples,
    point.estimates = c(mean(training.samples$alpha),
                        1, 
                        mean(training.samples$nu),
                        optim.results$par[4]),
    smoothness = kappa)

# plot the high resolution predictions
predicted.df <- data.frame(x = predicted.locations[, 1],
                           y = predicted.locations[ ,2],
                           z = apply(predicted.surface.2, 1, mean))

ggplot2::ggplot(predicted.df, aes(x = x, y = y, fill = z)) +
  geom_raster(interpolate = T) +
  scale_fill_gradient2(low = "red", mid = "white", high = "blue",
                       midpoint = 0, space = "Lab", na.value = "grey50",
                       guide = "colourbar", aesthetics = "fill")

# ==============================================================================
# Bivariate Marginal Model
# ==============================================================================

# define the parameters
N1 <- 25
N2 <- 25
N <- N1 * N2
n <- 300
kappa.bv <- c(3, 0.5, 1)
range.bv <- c(2, 1, 0.5)
sill.bv <- c(1, 1, -0.5)

# define the locations
observed.locations.grid <- cbind(seq(0, 10, length.out = N1),
                                 seq(0, 10, length.out = N2))
observed.locations <- expand.grid(seq(0, 10, length.out = N1),
                                  seq(0, 10, length.out = N2))

# define the bivariate Matern model using the RandomFields package
set.seed(1)
model.bv <- RMbiwm(nudiag = c(kappa.bv[1], kappa.bv[2]), 
                   nured = kappa.bv[3],
                   s = range.bv, 
                   cdiag = c(sill.bv[1], sill.bv[2]), 
                   rhored = sill.bv[3], 
                   notinvnu = TRUE)

# generate the observations                   
y.modeled <- RFsimulate(model.bv, x = observed.locations.grid, spConform=FALSE)
y.truth <- cbind(as.vector(y.modeled[,,1]), as.vector(y.modeled[,,2]))

# add noise (optional)
y <- y.truth + array(rnorm(N * 2, 0, 0.5), dim = c(N, 2))

# divide the observations into the training and testing sets
training.sample <- sample(1:N, n, replace = FALSE)
training.y.bv <- y[training.sample, ]
testing.y.bv <- y[-training.sample, ]
training.locations <- observed.locations[training.sample, ]
testing.locations <- observed.locations[-training.sample, ]


# inference using the parsimonious Matern cross-covariance function
optim.results <-
  NNRCM.bv.marginal.infer(Y = training.y.bv,
                          observed.locations = training.locations,
                          smoothness = kappa.bv)

# inference using the flexible Matern cross-covariance function
optim.results.ns <-
  NNRCM.bv.marginal.infer(Y = training.y.bv,
                          observed.locations = training.locations,
                          smoothness = kappa.bv,
                          parsimonious = FALSE)

# inference using coregionalization
optim.results.cr <-
  NNRCM.bv.marginal.infer(Y = training.y.bv,
                          observed.locations = training.locations,
                          smoothness = kappa.bv,
                          coregionalization = TRUE)

# inference using a separable covariance function
optim.results.sp <-
  NNRCM.bv.marginal.infer(Y = training.y.bv,
                          observed.locations = training.locations,
                          smoothness = rep(mean(kappa.bv[1:2]), 3))

# posterior predictive sampling of the testing set
posterior.predictive.marginal <- 
  NNRCM.bv.marginal.predict(Y = training.y.bv, 
                            observed.locations = training.locations, 
                            predicted.locations = testing.locations, 
                            point.estimates = optim.results$par,
                            smoothness = kappa.bv)

# scoring
NNRCM.scores(testing.y.bv[, 1], posterior.predictive.marginal[,,1])
NNRCM.scores(testing.y.bv[, 2], posterior.predictive.marginal[,,2])

# posterior predictive sampling for the high resolution surface
predicted.locations <- expand.grid(seq(-0.0001, 10.0001, length.out = 2 * N1),
                                   seq(-0.0001, 10.0001, length.out = 2 * N2))

predicted.surface <-
  mv.NNRCM.posterior.predictive(Y = training.y.bv,
                               observed.locations = training.locations,
                               predicted.locations = predicted.locations,
                               point.estimates = optim.results$par,
                               smoothness = kappa.bv)

# plot the true surface
true.df <- data.frame(x = observed.locations[, 1],
                      y = observed.locations[ ,2],
                      z1 = y.truth[, 1],
                      z2 = y.truth[, 2])

ggplot2::ggplot(true.df, aes(x = x, y = y, fill = z1)) +
  geom_raster(interpolate = T) +
  scale_fill_gradient2(low = "red", mid = "white",
                       high = "blue", midpoint = 0, space = "Lab",
                       na.value = "grey50", guide = "colourbar", aesthetics = "fill")

ggplot2::ggplot(true.df, aes(x = x, y = y, fill = z2)) +
  geom_raster(interpolate = T) +
  scale_fill_gradient2(low = "red", mid = "white",
                       high = "blue", midpoint = 0, space = "Lab",
                       na.value = "grey50", guide = "colourbar", aesthetics = "fill")

# plot the high resolution surface
predictions.df <- data.frame(x = predicted.locations[, 1],
                             y = predicted.locations[ ,2],
                             z1 = 1 / mcmc.samples * rowSums(predicted.surface[,,1]),
                             z2 = 1 / mcmc.samples * rowSums(predicted.surface[,,2]))

ggplot2::ggplot(predictions.df, aes(x = x, y = y, fill = z1)) +
  geom_raster(interpolate = T) +
  scale_fill_gradient2(low = "red", mid = "white",
                       high = "blue", midpoint = 0, space = "Lab",
                       na.value = "grey50", guide = "colourbar", aesthetics = "fill")

ggplot2::ggplot(predictions.df, aes(x = x, y = y, fill = z2)) +
  geom_raster(interpolate = T) +
  scale_fill_gradient2(low = "red", mid = "white",
                       high = "blue", midpoint = 0, space = "Lab",
                       na.value = "grey50", guide = "colourbar", aesthetics = "fill")

# ==============================================================================
# Bivariate Hierarchical Model
# ==============================================================================

# transform the dataset
A <- matrix(c(1,1,0,1), ncol =2)
transformed.training.y <- training.y.bv %*% t(A)
transformed.testing.y <- testing.y.bv %*% t(A)

# inference from the bivariate hierarchical model on the training set
posterior.samples <- 
  NNRCM.bv.hierarchical.infer(
    Y = transformed.training.y,
    observed.locations = training.locations,
    starting.values = list(alpha = optim.results$par[1],
                           sigma = optim.results$par[2:4],
                           nu = optim.results$par[5],
                           nugget = optim.results$par[7:8]),
    smoothness = kappa.bv, 
    a.nu.update.interval = 1005,
    A = A)

# posterior predictive sampling for the testing set
leave.out.predictive.samples <- 
  NNRCM.bv.hierarchical.predict(
    Y = transformed.training.y,
    kappa = kappa.bv,
    observed.locations = training.locations,
    predicted.locations = testing.locations, 
    posterior.samples = posterior.samples,
    point.estimates = list(alpha = optim.results$par[1],
                           sigma = optim.results$par[2:4],
                           nu = optim.results$par[5],
                           nugget = optim.results$par[7:8]),
    A = A)

# scoring
NNRCM.scores(testing.y.bv[, 1], leave.out.predictive.samples[,,1])
NNRCM.scores(testing.y.bv[, 2], leave.out.predictive.samples[,,2])

# posterior predictive sampling for the high resolution surface
predicted.surface <- 
  NNRCM.bv.hierarchical.predict(Y = transformed.training.y,
                                       kappa = kappa.bv,
                                       observed.locations = training.locations,
                                       predicted.locations = predicted.locations, 
                                       posterior.samples = posterior.samples,
                                       point.estimates = list(alpha = optim.results$par[1],
                                                              sigma = optim.results$par[2:4],
                                                              nu = optim.results$par[5],
                                                              nugget = optim.results$par[7:8]),
                                       A = A)

# plot the high resolution surface
predictions.df <- data.frame(x = predicted.locations[, 1],
                             y = predicted.locations[ ,2],
                             z1 = 1 / mcmc.samples * rowSums(predicted.surface[,,1]),
                             z2 = 1 / mcmc.samples * rowSums(predicted.surface[,,2]))

ggplot2::ggplot(predictions.df, aes(x = x, y = y, fill = z1)) +
  geom_raster(interpolate = T) +
  scale_fill_gradient2(low = "red", mid = "white",
                       high = "blue", midpoint = 0, space = "Lab",
                       na.value = "grey50", guide = "colourbar", aesthetics = "fill")

ggplot2::ggplot(predictions.df, aes(x = x, y = y, fill = z2)) +
  geom_raster(interpolate = T) +
  scale_fill_gradient2(low = "red", mid = "white",
                       high = "blue", midpoint = 0, space = "Lab",
                       na.value = "grey50", guide = "colourbar", aesthetics = "fill")