\name{predict.cv.hqreg}
\alias{predict.cv.hqreg}
\alias{coef.cv.hqreg}
\title{Model predictions based on "cv.hqreg" object.}
\description{This function makes predictions from a cross-validated hqreg model, using the stored fit 
and the optimal value chosen for \code{lambda}.}
\usage{
\method{predict}{cv.hqreg}(object, X, lambda = c("lambda.1se","lambda.min"), 
    type = c("response","coefficients","nvars"), ...)
\method{coef}{cv.hqreg}(object, lambda, ...)
}
\arguments{
  \item{object}{Fitted \code{"hqreg"} model object.}
  \item{X}{Matrix of values at which predictions are to be made. Used only for \code{type = "response"}.}
  \item{lambda}{Values of the regularization parameter \code{lambda} at which predictions 
    are requested. Default is the value \code{"lambda.1se"} stored on the CV \code{object}. 
    Alternatively \code{"lambda.min"} can be used. If \code{lambda} is numeric, it is taken 
    as the value(s) of \code{lambda} to be used.}
  \item{type}{Type of prediction. \code{"response"} returns the fitted values; \code{"coefficients"} 
    returns the coefficients; \code{"nvars"} returns the number of nonzero coefficients at 
    each value of \code{lambda}.}
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
\seealso{\code{\link{hqreg}} \code{\link{cv.hqreg}}}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
beta = rnorm(10)
eps = 4*rnorm(1000)
y = drop(X[,1:10] \%*\% beta + eps) 
cv = cv.hqreg(X, y, seed = 1011)
predict(cv, X[1:5,])
predict(cv, X[1:5,], lambda = "lambda.min")
predict(cv, X[1:5,], lambda = 0.05)
}
\keyword{models}
\keyword{regression}
