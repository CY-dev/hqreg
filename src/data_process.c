#include <math.h>
#include <time.h>
#include <string.h>
#include <R.h>
#include <R_ext/Applic.h>
#include "Rinternals.h"
#include "R_ext/Rdynload.h"

// standardization for feature matrix
void standardize(double *x, double *x2, double *shift, double *scale, int *nonconst, int n, int p) 
{
  int i, j, jn; double xm, xsd, xvar;
  for (i=0; i<n; i++) x2[i] = 1.0;
  for (j=1; j<p; j++) {
    jn = j*n; xm = 0.0; xsd = 0.0; xvar = 0.0; 
    for (i=0; i<n; i++) xm += x[jn+i];
    xm /= n;
    for (i=0; i<n; i++) {
      x[jn+i] -= xm;
      x2[jn+i] = pow(x[jn+i], 2);
      xvar += x2[jn+i];
    }
    xvar /= n;
    xsd = sqrt(xvar);
    if (xsd > 1e-6) {
      nonconst[j] = 1;
      for (i=0; i<n; i++) {
        x[jn+i] /= xsd;
        x2[jn+i] /= xvar;
      }
      shift[j] = xm;
      scale[j] = xsd;
    }
  }
  nonconst[0] = 1;
} 

// rescaling for feature matrix
void rescale(double *x, double *x2, double *shift, double *scale, int *nonconst, int n, int p) 
{
  int i, j, jn; double cmin, cmax, crange;
  for (i=0; i<n; i++) x2[i] = 1.0;
  for (j=1; j<p; j++) {
    jn = j*n; cmin = x[jn]; cmax = x[jn];
    for (i=0; i<n; i++) {
      if (x[jn+i] < cmin) {
        cmin = x[jn+i];
      } else if (x[jn+i] > cmax) {
        cmax = x[jn+i];
      }
    }
    crange = cmax - cmin;
    if (crange > 1e-6) {
      nonconst[j] = 1;
      for (i=0; i<n; i++) {
        x[jn+i] = (x[jn+i]-cmin)/crange;
        x2[jn+i] = pow(x[jn+i], 2);
      }
      shift[j] = cmin;
      scale[j] = crange;      
    }
  }
  nonconst[0] = 1;
}

// simple processing with assignment of nonconst
void simple_process(double *x, double *x2, int *nonconst, int n, int p, int intercept) 
{
  int i, j, jstart, jn; double cmin, cmax;
  if (intercept) {
    for (i=0; i<n; i++) x2[i] = 1.0;
    nonconst[0] = 1;
    jstart = 1;
  } else {
    jstart = 0;
  }
  for (j=jstart; j<p; j++) {
    jn = j*n; cmin = x[jn]; cmax = x[jn];
    for (i=0; i<n; i++) {
      x2[jn+i] = pow(x[jn+i], 2);
      if (x[jn+i] < cmin) {
        cmin = x[jn+i];
      } else if (x[jn+i] > cmax) {
        cmax = x[jn+i];
      }
    }
    if (cmax - cmin > 1e-6) nonconst[j] = 1;
  }
}

// postprocess feature coefficients
void postprocess(double *beta, double *shift, double *scale, int *nonconst, int nlam, int p) {
  int l, j, lp; double prod;
  for (l = 0; l<nlam; l++) {
    lp = l*p;
    prod = 0.0;
    for (j = 1; j<p; j++) {
      if (nonconst[j]) {
        beta[lp+j] /= scale[j];
        prod += shift[j]*beta[lp+j];
      }
    }
    beta[lp] -= prod;
  }
}
