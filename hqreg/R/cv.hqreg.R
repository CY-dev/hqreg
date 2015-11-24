cv.hqreg <- function(X, y, ..., nfolds=10, seed, trace=FALSE) {
  if (!missing(seed)) set.seed(seed)
  fit <- hqreg(X, y, ...)
  cv.args <- list(...)
  cv.args$lambda <- fit$lambda
  cv.args$gamma <- fit$gamma
  loss.args <- list(method=fit$method, gamma=fit$gamma, tau=fit$tau)
  E <- matrix(NA, nrow=length(y), ncol=length(cv.args$lambda))
  n <- length(y)
  cv.ind <- ceiling(sample(1:n)/n*nfolds)
  for (i in 1:nfolds) {
    if (trace) cat("Starting CV fold #",i,sep="","\n")
    cv.args$X <- X[cv.ind!=i,]
    cv.args$y <- y[cv.ind!=i]
    X2 <- X[cv.ind==i,]
    y2 <- y[cv.ind==i]
    fit.i <- do.call("hqreg", cv.args) # ensure the cross validation uses the same gamma for huber loss
    yhat <- predict(fit.i, X2)
    E[cv.ind==i, 1:ncol(yhat)] <- loss.hqreg(y2, yhat, loss.args)
  }

  ## Eliminate saturated lambda values
  ind <- which(apply(is.finite(E), 2, all))
  E <- E[,ind]
  lambda <- cv.args$lambda[ind]

  ## Results
  cve <- apply(E, 2, mean)
  cvse <- apply(E, 2, sd) / sqrt(n)
  index.min <- which.min(cve)
  # adjust the selection using 1-SD method
  index.1se <- min(which(cve < cve[index.min]+cvse[index.min]))
  val <- list(cve=cve, cvse=cvse, lambda=lambda, fit=fit, lambda.1se = lambda[index.1se], lambda.min=lambda[index.min])
  structure(val, class="cv.hqreg")
}
