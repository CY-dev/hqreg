\name{plot.cv.hqreg}
\alias{plot.cv.hqreg}
\title{Plot the cross-validation curve for a "cv.hqreg" object}
\description{Plot the cross-validation curve for a "cv.hqreg" object against the 
\code{lambda} values used, along with standard error bars.}
\usage{
\method{plot}{cv.hqreg}(x, log.l = TRUE, nvars = TRUE, ...)
}
\arguments{
  \item{x}{A \code{"cv.hqreg"} object.}
  \item{log.l}{Should \code{log(lambda)} be used instead of \code{lambda} for X-axis? Default is TRUE.}
  \item{nvars}{If \code{TRUE} (the default), places an axis on top of the plot denoting 
    the number of variables with nonzero coefficients at each \code{lambda}.}
  \item{\dots}{Other graphical parameters to \code{plot}}
}
\details{Produces a plot of mean cv errors at each \code{lambda} along with upper and lower standard error bars.}
\references{Yi, C. and Huang, J. (2016) 
  \emph{Semismooth Newton Coordinate Descent Algorithm for 
  Elastic-Net Penalized Huber Loss Regression and Quantile Regression},
  \url{https://arxiv.org/abs/1509.02957} \cr
  \emph{Journal of Computational and Graphical Statistics, accepted in Nov 2016} \cr
  \url{http://www.tandfonline.com/doi/full/10.1080/10618600.2016.1256816}}
\author{Congrui Yi <congrui-yi@uiowa.edu>}
\seealso{\code{\link{hqreg}}, \code{\link{cv.hqreg}}}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
beta = rnorm(10)
eps = 4*rnorm(1000)
y = drop(X[,1:10] \%*\% beta + eps)
cv = cv.hqreg(X, y, seed = 123)
plot(cv)
}
\keyword{models}
\keyword{regression}
