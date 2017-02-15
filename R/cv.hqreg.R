cv.hqreg <- function(X, y, ..., FUN = c("hqreg", "hqreg_raw"), ncores = 1, nfolds=10, fold.id, type.measure = c("deviance", "mse", "mae"), seed) {
  FUN <- match.arg(FUN)
  FUN <- switch(FUN, "hqreg"=hqreg, "hqreg_raw"=hqreg_raw)
#  FUN <- hqreg::FUN
#  FUN <- get(FUN)
  type.measure <- match.arg(type.measure)
  n <- length(y)
  if (!missing(seed)) set.seed(seed)
  if(missing(fold.id)) fold.id <- ceiling(sample(1:n)/n*nfolds)

  fit <- FUN(X, y, ...)
  cv.args <- list(...)
  cv.args$lambda <- fit$lambda
  cv.args$alpha <- fit$alpha
  cv.args$gamma <- fit$gamma
  cv.args$tau <- fit$tau
  measure.args <- list(method=fit$method, gamma=fit$gamma, tau=fit$tau, type.measure = type.measure)
  E <- matrix(NA, nrow=length(y), ncol=length(cv.args$lambda))
  
  parallel <- FALSE
  if (ncores > 1) {
    max.cores <- detectCores()
    if (ncores > max.cores) {
      cat("The number of cores specified (", ncores, ") is larger than 
          the number of avaiable cores (", max.cores, "), so", max.cores, "cores are used.", "\n")
      ncores = max.cores
    }
    cluster <- makeCluster(ncores)
    if (!("cluster" %in% class(cluster))) stop("Cluster is not of class 'cluster'; see ?makeCluster")
    parallel <- TRUE
    cat("Start parallel computing for cross-validation...")
    clusterExport(cluster, c("fold.id", "X", "y", "cv.args", "measure.args"), 
                  envir=environment())
    clusterCall(cluster, function() require(FUN))
    fold.results <- parLapply(cl = cluster, X = 1:nfolds, fun = cvf, XX = X, y = y, 
                              fold.id = fold.id, cv.args = cv.args, measure.args = measure.args)
    stopCluster(cluster)
  }
  
  E <- matrix(NA, nrow = n, ncol = length(cv.args$lambda))
  for (i in 1:nfolds) {
    if (parallel) {
      fit.i <- fold.results[[i]]
    } else {
      cat("CV fold #",i,sep="","\n")
      fit.i <- cvf(i, X, y, fold.id, cv.args, measure.args, FUN)
    }
    E[fold.id == i, 1:fit.i$nl] <- fit.i$pe
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
  val <- list(cve = cve, cvse = cvse, type.measure = type.measure, lambda = lambda, fit = fit, 
              lambda.1se = lambda[index.1se], lambda.min = lambda[index.min])
  structure(val, class="cv.hqreg")
  }

cvf <- function(i, XX, y, fold.id, cv.args, measure.args, FUN) {
  cv.args$X <- XX[fold.id != i,,drop = FALSE]
  cv.args$y <- y[fold.id != i]
  X2 <- XX[fold.id == i,,drop = FALSE]
  y2 <- y[fold.id == i]
  fit.i <- do.call(FUN, cv.args)
  yhat <- matrix(predict.hqreg(fit.i, X2), length(y2))
  list(pe = measure.hqreg(y2, yhat, measure.args), nl = length(fit.i$lambda))
}
