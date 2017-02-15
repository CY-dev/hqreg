hqreg_raw <- function (X, y, method = c("huber", "quantile", "ls"), gamma = IQR(y)/10, tau = 0.5, alpha=1, nlambda=100, lambda.min = 0.05, lambda, 
                       intercept = TRUE, screen = c("ASR", "SR", "none"), max.iter = 10000, eps = 1e-7, 
                       dfmax = ncol(X)+1, penalty.factor=rep(1, ncol(X)), message = FALSE) {
  
  # Error checking
  method <- match.arg(method)
  screen <- match.arg(screen)
  if (missing(lambda) && nlambda < 2) stop("nlambda should be at least 2")
  if (alpha < 0 || alpha > 1) stop("alpha should be between 0 and 1")
  if (method == "huber" && !missing(gamma) && gamma <= 0) stop("gamma should be positive for Huber loss")
  if (method == "quantile" && (tau < 0 || tau > 1)) stop("tau should be between 0 and 1 for quantile loss")
  if (length(penalty.factor)!=ncol(X)) stop("the length of penalty.factor should equal the number of columns of X")
  
  call <- match.call()
  if (intercept == TRUE) {
    XX <- cbind(1, X)
    penalty.factor <- c(0, penalty.factor) # no penalty for intercept
    if (method == "huber") {
      shift <- if(gamma > sd(y)) mean(y) else median(y)
    } else if (method == "ls") {
      shift <- mean(y)
    } else if (method == "quantile") {
      shift <- quantile(y, tau)
    }
  } else {
    XX <- X
    shift <- 0
  }
  n <- nrow(XX)
  p <- ncol(XX)
  yy <- y - shift
  
  # Flag for user-supplied lambda
  user <- 0
  if (missing(lambda)) {
    lambda <- double(nlambda)
  } else {
    nlambda <- length(lambda)
    user <- 1
  }
  
  # Flags for preprocessing and screening
  ppflag = 0 # no preprocessing
  scrflag = switch(screen, ASR = 1, SR = 2, none = 0)
  # Fitting
  if (alpha > 0) {
    if (method == "huber") {
      fit <- .C(C_huber, double(p*nlambda), integer(nlambda), as.double(lambda), integer(1), integer(1), as.double(XX), as.double(yy), as.double(penalty.factor), 
                as.double(gamma), as.double(alpha), as.double(eps), as.double(lambda.min), as.integer(nlambda), as.integer(n), as.integer(p), as.integer(ppflag),
                as.integer(scrflag), as.integer(intercept), as.integer(dfmax), as.integer(max.iter), as.integer(user), as.integer(message))
    } else if (method == "quantile") {
      fit <- .C(C_quant, double(p*nlambda), integer(nlambda), as.double(lambda), integer(1), integer(1), as.double(XX), as.double(yy), as.double(penalty.factor), 
                as.double(tau), as.double(alpha), as.double(eps), as.double(lambda.min), as.integer(nlambda), as.integer(n), as.integer(p), 
                as.integer(ppflag), as.integer(scrflag), as.integer(intercept), as.integer(dfmax), as.integer(max.iter), as.integer(user), as.integer(message))
    } else {
      fit <- .C(C_squared, double(p*nlambda), integer(nlambda), as.double(lambda), integer(1), integer(1), as.double(XX), as.double(yy), as.double(penalty.factor), 
                as.double(alpha), as.double(eps), as.double(lambda.min), as.integer(nlambda), as.integer(n), as.integer(p), as.integer(ppflag), as.integer(scrflag),
                as.integer(intercept), as.integer(dfmax), as.integer(max.iter), as.integer(user), as.integer(message))
    }
    beta <- matrix(fit[[1]],nrow = p)
    iter <- fit[[2]]
    lambda <- fit[[3]]
    saturated <- fit[[4]]
    nv <- fit[[5]]
    # Eliminate saturated lambda values
    ind <- !is.na(iter)
    beta <- beta[, ind]
    iter <- iter[ind]
    lambda <- lambda[ind]
  } else {
    if (method == "huber") {
      fit <- .C(C_huber_l2, double(p*nlambda), integer(nlambda), as.double(lambda), as.double(XX), as.double(yy), as.double(penalty.factor), 
                as.double(gamma), as.double(eps), as.double(lambda.min), as.integer(nlambda), as.integer(n), as.integer(p), as.integer(ppflag),
                as.integer(intercept), as.integer(max.iter), as.integer(user), as.integer(message))
    } else if (method == "quantile") {
      fit <- .C(C_quantile_l2, double(p*nlambda), integer(nlambda), as.double(lambda), as.double(XX), as.double(yy), as.double(penalty.factor), 
                as.double(tau), as.double(eps), as.double(lambda.min), as.integer(nlambda), as.integer(n), as.integer(p), as.integer(ppflag),
                as.integer(intercept), as.integer(max.iter), as.integer(user), as.integer(message))      
    } else {
      fit <- .C(C_squared_l2, double(p*nlambda), integer(nlambda), as.double(lambda), as.double(XX), as.double(yy), as.double(penalty.factor), 
                as.double(eps), as.double(lambda.min), as.integer(nlambda), as.integer(n), as.integer(p), as.integer(ppflag),
                as.integer(intercept), as.integer(max.iter), as.integer(user), as.integer(message))      
    }
    beta <- matrix(fit[[1]],nrow = p)
    iter <- fit[[2]]
    lambda <- fit[[3]]
    saturated <- 0
    nv <- 0
  }
  
  # Intercept
  beta[1,] <- beta[1,] + shift
  
  # Names
  vnames <- colnames(X)
  if (intercept == TRUE) {
    if (is.null(vnames)) vnames=paste0("V",seq(p-1))
    vnames <- c("(Intercept)", vnames)
  } else if (is.null(vnames)) {
    vnames=paste0("V",seq(p))
  }
  dimnames(beta) <- list(vnames, paste0("L", 1:length(lambda)))
  
  # Output
  structure(list(call = call,
                 beta = beta,
                 iter = iter,
                 saturated = saturated,
                 lambda = lambda,
                 alpha = alpha,
                 gamma = if (method == "huber") gamma else NULL,
                 tau = if (method == "quantile") tau else NULL,
                 penalty.factor = if (intercept) penalty.factor[-1] else penalty.factor,
                 method = method,
                 nv = nv),
            class = "hqreg")
}
