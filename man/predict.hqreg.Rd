\name{predict.hqreg}
\alias{predict.hqreg}
\alias{coef.hqreg}
\title{Model predictions based on "hqreg" object.}
\description{This function returns fitted values, coefficients and more from a fitted \code{"hqreg"} object.}
\usage{
\method{predict}{hqreg}(object, X, lambda, type = c("response","coefficients","nvars"), 
    exact = FALSE, ...)
\method{coef}{hqreg}(object, lambda, exact = FALSE, ...)
}
\arguments{
  \item{object}{Fitted \code{"hqreg"} model object.}
  \item{X}{Matrix of values at which predictions are to be made. Used only for \code{type = "response"}.}
  \item{lambda}{Values of the regularization parameter \code{lambda} at which predictions 
    are requested. Default is the entire sequence used to create the model.}
  \item{type}{Type of prediction. \code{"response"} returns the fitted values; 
    \code{"coefficients"} returns the coefficients; \code{"nvars"} returns the number of 
    nonzero coefficients at each value of \code{lambda}.}
  \item{exact}{If \code{exact=FALSE} (default), then the function uses linear interpolation 
    to make predictions for values of \code{lambda} that do not coincide with those used to 
    fit the model. If \code{exact=TRUE}, and predictions are requested at values of \code{lambda} 
    not included in the original fit, the model is refit on a lambda sequence consisting 
    \code{object$lambda} and the new ones before predictions are made. }
  \item{\dots}{Not used. Other arguments to predict.}
}
\value{The object returned depends on type.}
\references{Yi, C. and Huang, J. (2016) 
  \emph{Semismooth Newton Coordinate Descent Algorithm for 
  Elastic-Net Penalized Huber Loss Regression and Quantile Regression},
  \url{https://arxiv.org/abs/1509.02957} \cr
  \emph{Journal of Computational and Graphical Statistics, accepted in Nov 2016} \cr
  \url{http://www.tandfonline.com/doi/full/10.1080/10618600.2016.1256816}}
\author{Congrui Yi <congrui-yi@uiowa.edu>}
\seealso{\code{\link{hqreg}}}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
beta = rnorm(10)
eps = 4*rnorm(1000)
y = drop(X[,1:10] \%*\% beta + eps) 
fit = hqreg(X, y, method = "quantile", tau = 0.7)
predict(fit, X[1:5,], lambda = c(0.05, 0.01))
predict(fit, X[1:5,], lambda = 0.05, exact = TRUE)
predict(fit, X[1:5,], lambda = 0.05, type = "nvars")
coef(fit, lambda = 0.05)
}
\keyword{models}
\keyword{regression}
