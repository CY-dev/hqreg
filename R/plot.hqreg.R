plot.hqreg <- function(x, xvar = c("lambda", "norm"), log.l = TRUE, nvars = TRUE, alpha = 1, ...)
{
  xvar <- match.arg(xvar)
  if (nrow(x$beta) == length(x$penalty.factor)) { # no intercept
    Y <- x$beta[,,drop = FALSE]
  } else {
    Y <- x$beta[-1,,drop = FALSE]
  }
  nonzero <- which(rowSums(abs(Y))!=0)
  Y <- Y[nonzero,,drop = FALSE]
  p <- nrow(Y)
  if (xvar == "lambda") {
    X <- x$lambda
    if (log.l) X <- log(X)
  } else {
    X <- colSums(abs(Y))
  }
  
  if (xvar == "lambda") {
    xlab <- if (log.l) expression(log(lambda)) else expression(lambda)
    xlim <- rev(range(X))
  } else {
    xlab <- expression(group("||", hat(beta), "||")[1])
    xlim <- range(X)
  }
  plot.args <- list(x=X, y=seq(X), ylim=range(Y), xlab=xlab, ylab="", 
                    type="n", xlim=xlim)
  new.args <- list(...)
  if (length(new.args)) plot.args[names(new.args)] <- new.args
  do.call("plot", plot.args)
  if (!is.element("ylab", names(new.args))) mtext(expression(hat(beta)), side=2, cex=par("cex"), line=3, las=1)
  
  cols <- hcl(h=seq(15, 375, len=max(4, p+1)), l=60, c=150, alpha=alpha)
  cols <- if (p==2) cols[c(1,3)] else cols[1:p]  
  line.args <- list(col=cols, lwd=1+2*exp(-p/20), lty=1)
  if (length(new.args)) line.args[names(new.args)] <- new.args
  line.args$x <- X
  line.args$y <- t(Y)
  do.call("matlines",line.args)
  abline(h=0)
  if (nvars) {
    nv = predict(x, lambda = x$lambda, type = "nvars")
    axis(3, at=X, labels=nv, tick=FALSE, line=-0.5)
  }
}

