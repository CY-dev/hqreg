
#include <math.h>
#include <time.h>
#include <string.h>
#include <R.h>
#include <R_ext/Applic.h>
#include "Rinternals.h"
#include "R_ext/Rdynload.h"

double sign(double x) {
  if (x > 0) return 1.0;
  else if (x < 0) return -1.0;
  else return 0.0;
}

double crossprod(double *x, double *v, int n, int j) {
  int jn = j*n, i; double sum=0.0;
  for (i=0;i<n;i++) sum += x[jn+i]*v[i];
  return(sum);
}

double maxprod(double *x, double *v, int n, int p, double *pf, int *nonconst) {
  int j; double z, max=0.0;
  for (j=1; j<p; j++) {
    if (pf[j] && nonconst[j]) {
      z = fabs(crossprod(x, v, n, j))/pf[j];
      if (z>max) max = z;
    }
  }
  return(max);
}
