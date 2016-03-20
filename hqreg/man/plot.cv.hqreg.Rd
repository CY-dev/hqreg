\name{plot.cv.hqreg}
\alias{plot.cv.hqreg}
\title{Plot the cross-validation curve for a "cv.hqreg" object}
\description{Plot the cross-validation curve for a "cv.hqreg" object against the 
\code{lambda} values used, along with standard error bars.}
\usage{
\method{plot}{cv.hqreg}(x, log.x = TRUE, nvars = TRUE, ...)
}
\arguments{
  \item{x}{A \code{"cv.hqreg"} object.}
  \item{log.x}{Should X-axis be on the log scale?  Default is TRUE.}
  \item{nvars}{If \code{TRUE} (the default), places an axis on top of the plot denoting 
    the number of variables with nonzero coefficients at each \code{lambda}.}
  \item{\dots}{Other graphical parameters to \code{plot}}
}
\details{Produces a plot of mean cv errors at each \code{lambda} along with upper and lower standard error bars.}
\author{Congrui Yi <congrui-yi@uiowa.edu>}
\seealso{\code{\link{hqreg}}, \code{\link{cv.hqreg}}}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
beta = rnorm(10)
eps = 4*rnorm(1000)
y = drop(X[,1:10] \%*\% beta + eps)
cv = cv.hqreg(X, y, seed = 123)
plot(cv, log.x = TRUE)
}
\keyword{models}
\keyword{regression}
