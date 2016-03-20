loss.hqreg <- function(y, yhat, args) {
  r <- y-yhat
<<<<<<< HEAD
  type.measure <- args$type.measure
  if (type.measure == "deviance") {
    method <- args$method
    if (method == "huber") {
      gamma <- args$gamma
      val <- hloss(r, gamma)
    } else if (method == "quantile") {
      tau <- args$tau
      val <- qloss(r, tau)
    } else {
      val <- r^2
    }    
  } else if (type.measure == "mse") {
    val <- r^2
  } else {
    val <- abs(r)
=======
  method <- args$method
  if (method == "huber") {
    gamma <- args$gamma
    val <- hloss(r, gamma)
  } else if (method == "quantile") {
    tau <- args$tau
    val <- qloss(r, tau)
  } else {
    val <- r^2
>>>>>>> e7b4a1a73aa4e9ac7fc87ddc10bbb0cab36afcd7
  }
  val
}

hloss <- function(r, gamma) {
  rr <- abs(r)
  ifelse(rr <= gamma, rr^2/(2*gamma), rr-gamma/2)
}

<<<<<<< HEAD
qloss <- function(r, tau) ifelse(r <= 0, (tau-1)*r, tau*r)
=======
qloss <- function(r, tau) ifelse(r <= 0, (1-tau)*r, tau*r)
>>>>>>> e7b4a1a73aa4e9ac7fc87ddc10bbb0cab36afcd7
