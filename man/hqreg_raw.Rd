\name{hqreg_raw}
\alias{hqreg_raw}
\title{On raw data fit a robust regression model with Huber or quantile loss penalized by lasso or elasti-net}
\description{On raw data fit solution paths for Huber loss regression or quantile regression penalized 
by lasso or elastic-net over a grid of values for the regularization parameter lambda.}
\usage{
hqreg_raw(X, y, method = c("huber", "quantile", "ls"),
    gamma = IQR(y)/10, tau = 0.5, alpha = 1, nlambda = 100, lambda.min = 0.05, lambda, 
    intercept = TRUE, screen = c("ASR", "SR", "none"), 
    max.iter = 10000, eps = 1e-7, dfmax = ncol(X)+1, penalty.factor = rep(1, ncol(X)), 
    message = FALSE)
}

\arguments{
  \item{X}{Input matrix.}
  \item{y}{Response vector.}
  \item{method}{The loss function to be used in the model. Either "huber" (default), 
    "quantile", or "ls" for least squares (see \code{Details}).}
  \item{gamma}{The tuning parameter of Huber loss, with no effect for the other loss 
    functions. Huber loss is quadratic for absolute values less than gamma and linear for those 
    greater than gamma. The default value is IQR(y)/10.}
  \item{tau}{The tuning parameter of the quantile loss, with no effect for the other loss 
    functions. It represents the conditional quantile of the response to be estimated, so 
    must be a number between 0 and 1. It includes the absolute loss when tau = 0.5 (default).}
  \item{alpha}{The elastic-net mixing parameter that controls the relative contribution 
    from the lasso and the ridge penalty. It must be a number between 0 and 1. \code{alpha=1} 
    is the lasso penalty and \code{alpha=0} the ridge penalty.}
  \item{nlambda}{The number of lambda values.  Default is 100.}
  \item{lambda.min}{The smallest value for lambda, as a fraction of lambda.max, the data 
    derived entry value. Default is 0.05.}
  \item{lambda}{A user-specified sequence of lambda values. Typical usage is to leave 
    blank and have the program automatically compute a \code{lambda} sequence based on 
    \code{nlambda} and \code{lambda.min}. Specifying \code{lambda} overrides this. This 
    argument should be used with care and supplied with a decreasing sequence instead of 
    a single value. To get coefficients for a single \code{lambda}, use \code{coef} or 
    \code{predict} instead after fitting the solution path with \code{hqreg} or performing 
    k-fold CV with \code{cv.hqreg}.}
  \item{intercept}{Should an intercept be included? Default is TRUE.}
  \item{screen}{Screening rule to be applied at each \code{lambda} that discards variables 
    for speed. Either "ASR" (default), "SR" or "none". "SR" stands for the strong rule, 
    and "ASR" for the adaptive strong rule. Using "ASR" typically requires fewer iterations 
    to converge than "SR", but the computing time are generally close. Note that the option 
    "none" is used mainly for debugging, which may lead to much longer computing time.}
  \item{max.iter}{Maximum number of iterations. Default is 10000.}
  \item{eps}{Convergence threshold. The algorithms continue until the maximum change in the
    objective after any coefficient update is less than \code{eps} times the null deviance. 
    Default is \code{1E-7}.}
  \item{dfmax}{Upper bound for the number of nonzero coefficients. The algorithm exits and 
    returns a partial path if \code{dfmax} is reached. Useful for very large dimensions.}
  \item{penalty.factor}{A numeric vector of length equal to the number of variables. Each 
    component multiplies \code{lambda} to allow differential penalization. Can be 0 for 
    some variables, in which case the variable is always in the model without penalization. 
    Default is 1 for all variables.}
  \item{message}{If set to TRUE,  hqreg will inform the user of its progress. This argument 
    is kept for debugging. Default is FALSE.}
}

\details{
  The sequence of models indexed by the regularization parameter \code{lambda} is fit 
  using a semismooth Newton coordinate descent algorithm. The objective function is defined 
  to be \deqn{\frac{1}{n} \sum loss_i + \lambda\textrm{penalty}.}{\sum loss_i /n + \lambda*penalty.}
  For \code{method = "huber"}, 
  \deqn{loss(t) = \frac{t^2}{2\gamma} I(|t|\le \gamma) + (|t| - \frac{\gamma}{2};) I(|t|>
  \gamma)}{loss(t) = t^2/(2*\gamma) I(|t|\le \gamma) + (|t| - \gamma/2) I(|t|>\gamma);}
  for \code{method = "quantile"}, \deqn{loss(t) = t (\tau - I(t<0));}
  for \code{method = "ls"}, \deqn{loss(t) = \frac{t^2}{2}}{loss(t) = t^2/2.}
  In the model, "t" is replaced by residuals.
}

\value{
  The function returns an object of S3 class \code{"hqreg"}, which is a list containing:
  \item{call}{The call that produced this object.}
  \item{beta}{The fitted matrix of coefficients.  The number of rows is equal to the number 
    of coefficients, and the number of columns is equal to \code{nlambda}. An intercept is included.}
  \item{iter}{A vector of length \code{nlambda} containing the number of iterations until 
    convergence at each value of \code{lambda}.}
  \item{saturated}{A logical flag for whether the number of nonzero coefficients has reached \code{dfmax}.}
  \item{lambda}{The sequence of regularization parameter values in the path.}
  \item{alpha}{Same as above.}
  \item{gamma}{Same as above. \code{NULL} except when \code{method = "huber"}.}
  \item{tau}{Same as above. \code{NULL} except when \code{method = "quantile"}.}
  \item{penalty.factor}{Same as above.}
  \item{method}{Same as above.}
  \item{nv}{The variable screening rules are accompanied with checks of optimality 
    conditions. When violations occur, the program adds in violating variables and re-runs 
    the inner loop until convergence. \code{nv} is the number of violations.}
}
\author{Congrui Yi <congrui-yi@uiowa.edu>}
\seealso{\code{\link{plot.hqreg}}, \code{\link{cv.hqreg}}}
\examples{
X = matrix(rnorm(1000*100), 1000, 100)
beta = rnorm(10)
eps = 4*rnorm(1000)
y = drop(X[,1:10] \%*\% beta) + eps

# Huber loss
# include an intercept by default
fit1 = hqreg_raw(X, y)
coef(fit1, 0.01)
predict(fit1, X[1:5,], lambda = c(0.02, 0.01))

# no intercept
fit2 = hqreg_raw(X, y, intercept = FALSE)
plot(fit2)
}
\keyword{models}
\keyword{regression}
