\name{cv.hqreg}
\alias{cv.hqreg}
\title{Cross-validation for hqreg}
\description{Perform k-fold cross validation for elastic-net penalized Huber loss regression 
  and quantile regression over a sequence of lambda values and find an optimal lambda.}
\usage{
cv.hqreg(X, y, ..., ncores = 1, nfolds = 10, fold.id, type.measure = c("deviance", "mse", "mae"), 
         seed, trace = FALSE)
}
\arguments{
  \item{X}{The input matrix, as in \code{hqreg}.}
  \item{y}{The response vector, as in \code{hqreg}.}
  \item{...}{Additional arguments to \code{hqreg}.}
  \item{ncores}{\code{cv.hqreg} can be run in parallel across a
  cluster using the \code{parallel} package. If \code{ncores > 1},a cluster is 
  created to run \code{cv.hqreg} in parallel. The code is run sequentially if 
  \code{ncores = 1} (the default). A message is printed if \code{ncores} is larger than the
  total number of available cores, and all available cores will be used.}
  \item{nfolds}{The number of cross-validation folds. Default is 10.}
  \item{fold.id}{(Optional) a vector of values between 1 and nfold indicating 
  which fold each observation belongs to. If supplied, nfolds can be missing. 
  By default the observations are randomly assigned by \code{cv.hqreg}. }
  \item{type.measure}{The default is "deviance", which uses the chosen loss function of the model. 
  Other options include "mse" for mean squared error and "mae" for mean absolute error.}
  \item{seed}{(Optional) Seed for the random number generator in order to obtain reproducible results.}
  \item{trace}{If set to TRUE, cv.hqreg will inform the user of its
    progress by announcing the beginning of each CV fold.  Default is
    FALSE.}
  }
\details{
The function randomly partitions the data in \code{nfolds}. It calls \code{hqreg} 
\code{nfolds}+1 times, the first to obtain the \code{lambda} sequence, and the remainder 
to fit with each of the folds left out once for validation. The cross-validation error is 
the average of validation errors for the \code{nfolds} fits.

Note that \code{cv.hqreg} does not search for values of \code{alpha}, \code{gamma} or \code{tau}. 
Specific values should be supplied, otherwise the default ones for \code{hqreg} are used. 
If users would like to cross-validate \code{alpha}, \code{gamma} or \code{tau} as well, 
they should call \code{cv.hqreg} for each combination of these parameters and use the same 
"seed" in these calls so that the partitioning remains the same.
}
\value{
  The function returns an object of S3 class \code{"cv.hqreg"}, which is a list containing:
  \item{cve}{The error for each value of \code{lambda}, averaged across the cross-validation folds.}
  \item{cvse}{The estimated standard error associated with each value of \code{cve}.}
  \item{lambda}{The values of \code{lambda} used in the cross-validation fits.}
  \item{fit}{The fitted \code{hqreg} object for the whole data.}
  \item{lambda.1se}{The largest \code{lambda} such that the error is within 1 standard 
    error of the minimum.}
  \item{lambda.min}{The value of \code{lambda} with the minimum cross-validation error.}
}
\author{Congrui Yi <congrui-yi@uiowa.edu>}
\seealso{\code{hqreg}, \code{plot.cv.hqreg}}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
beta = rnorm(10)
eps = 4*rnorm(1000)
y = drop(X[,1:10] \%*\% beta + eps)
cv = cv.hqreg(X, y, seed = 123)
plot(cv)
predict(cv, X[1:5,])
}
\keyword{models}
\keyword{regression}
