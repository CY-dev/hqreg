# Determine vector d used to compute lambda.max
initResidual <- function(yy, X, method, gamma, c, penalty.factor)
{
  ind <- which(penalty.factor!=0)
  if(length(ind) == ncol(X)) {
    yy
  } else {
    fit <- lm(yy~X[,-ind]+0)
    fit$residuals
  }
}

