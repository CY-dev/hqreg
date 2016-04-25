\name{plot.hqreg}
\alias{plot.hqreg}
\title{Plot coefficients from a "hqreg" object}
\description{Produce a plot of the coefficient paths for a fitted
  \code{"hqreg"} object.}
\usage{
\method{plot}{hqreg}(x, xvar = c("lambda", "norm"), log.x=FALSE, nvars = TRUE, 
    alpha = 1, ...)
}
\arguments{
  \item{x}{A \code{hqreg} object.}
  \item{xvar}{What is on the X-axis. \code{"lambda"} plots against the lambda sequence, 
    \code{"norm"} against the L1-norm of the coefficients.}
  \item{log.x}{Should X-axis be on the log scale?  Default is FALSE. Must set \code{log.x}
    to be FALSE for \code{xvar = "norm"} since the norm can be exactly 0.}
  \item{nvars}{If \code{TRUE} (the default), places an axis on top of the plot denoting the 
    number of variables with nonzero coefficients at each \code{lambda}.}
  \item{alpha}{A value between 0 and 1 for alpha transparency channel(0 means transparent 
    and 1 means opaque), helpful when the number of variables is large.}
  \item{\dots}{Other graphical parameters to \code{plot}.}
}
\author{Congrui Yi <congrui-yi@uiowa.edu>}
\seealso{\code{\link{hqreg}}}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
beta = rnorm(10)
eps = 4*rnorm(1000)
y = drop(X[,1:10] \%*\% beta + eps) 
fit = hqreg(X, y)
par(mfrow = c(2,2))
plot(fit)
plot(fit, nvars = FALSE, alpha = 0.5)
plot(fit, log.x = TRUE)
plot(fit, xvar = "norm")
}
\keyword{models}
\keyword{regression}
