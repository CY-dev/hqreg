\name{hqreg-package}
\alias{hqreg-package}
\docType{package}
\title{Regularization Paths for Huber Loss Regression and Quantile Regression Penalized by Lasso or Elastic-net}
\description{Efficient algorithms for fitting coefficient paths for regression models with 
  Huber loss, quantile loss or squared loss penalized by lasso or elastic-net.}
\details{
\tabular{ll}{
Package: \tab hqreg\cr
Type: \tab Package\cr
Version: \tab 1.2-1\cr
Date: \tab 2016-03-19\cr
License: \tab GPL-3\cr
}
Very simple to use. Accepts \code{X,y} data for regression models, and
produces the regularization path over a grid of values for the tuning
parameter \code{lambda}. Also provides methods for plotting, prediction and cross-validation.
}
\author{Congrui Yi <congrui-yi@uiowa.edu>}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
beta = rnorm(10)
eps = 4*rnorm(1000)
y = drop(X[,1:10] \%*\% beta + eps) 

# Huber loss
fit1 = hqreg(X, y)
coef(fit1, 0.01)
predict(fit1, X[1:5,], lambda = c(0.02, 0.01))

# Quantile loss
fit2 = hqreg(X, y, method = "quantile", tau = 0.2)
plot(fit2, xvar = "norm")

# Squared loss
fit3 = hqreg(X, y, method = "ls", preprocess = "rescale")
plot(fit3, xvar = "lambda", log.x = TRUE)
}
\keyword{models}
\keyword{regression}
