#include <math.h>
#include <time.h>
#include <string.h>
#include <R.h>
#include <R_ext/Applic.h>
#include "Rinternals.h"
#include "R_ext/Rdynload.h"

double sign(double x);
double crossprod(double *x, double *v, int n, int j);
double maxprod(double *x, double *v, int n, int p, double *pf, int *nonconst);
double ksav(double *a, int size, int K);
void standardize(double *x, double *x2, double *shift, double *scale, int *nonconst, int n, int p);
void rescale(double *x, double *x2, double *shift, double *scale, int *nonconst, int n, int p);
void simple_process(double *x, double *x2, int *nonconst, int n, int p, int intercept);
void postprocess(double *beta, double *shift, double *scale, int *nonconst, int nlam, int p);
void init_huber(double *beta, double *beta_old, int *iter, double *x, double *x2, 
		double *y, double *r, double *pf, double *d1, double *d2, int *nonconst, 
		double gamma, double thresh, int n, int p, int max_iter);
void init_quantile(double *beta, double *beta_old, int *iter, double *x, double *x2, 
		   double *y, double *r, double *pf, double *d1, double *d2, int *nonconst,
		   double gamma, double c, double thresh, int n, int p, int max_iter);
void init_squared(double *beta, double *beta_old, int *iter, double *x, double *x2m, double *y, double *r, 
                  double *pf, int *nonconst, double thresh, int n, int p, int ppflag, int max_iter);

void derivative_huber(double *d1, double *d2, double *r, double gamma, int n) {
  double gi = 1.0/gamma;
  for (int i=0; i<n; i++)
    if (fabs(r[i]) > gamma) {
      d1[i] = sign(r[i]);
      d2[i] = 0.0;
    } else {
      d1[i] = r[i]*gi;
      d2[i] = gi;
    }
}

void derivative_quantapprox(double *d1, double *d2, double *r, double gamma, double c, int n) {
  double gi = 1.0/gamma;
  for (int i=0; i<n; i++) {
    if (fabs(r[i]) > gamma) {
      d1[i] = sign(r[i])+c;
      d2[i] = 0.0;
    } else {
      d1[i] = r[i]*gi+c;
      d2[i] = gi;
    }
  }
}

// Semismooth Newton Coordinate Descent (SNCD)
static void sncd_huber(double *beta, int *iter, double *lambda, int *saturated, int *numv, double *x, double *y, 
		       double *pf, double *gamma_, double *alpha_, double *eps_, double *lambda_min_, int *nlam_, 
		       int *n_, int *p_, int *ppflag_, int *scrflag_, int *intercept_, int *dfmax_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double gamma = gamma_[0]; double alpha = alpha_[0]; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0]; int scrflag = scrflag_[0]; int intercept = intercept_[0];
  int dfmax = dfmax_[0]; int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, ll, lp, jn, lstart, mismatch; 
  double pct, lstep, ldiff = 1.0, l1, l2, v1, v2, v3, tmp, change, nullDev, max_update, update, thresh;
  double gi = 1.0/gamma; // 1/gamma as a multiplier
  double scrfactor = 1.0; // scaling factor used for screening rules
  //scrflag = 0: no screening; scrflag = 1: Adaptive Strong Rule(ASR); scrflag = 2: Strong Rule(SR)
  // ASR fits an appropriate scrfactor adaptively; SR always uses scrfactor = 1
  int nnzero = 0; // number of nonzero variables
  double *x2 = Calloc(n*p, double); // x^2
  double *shift = Calloc(p, double);
  double *scale = Calloc(p, double);
  double *beta_old = Calloc(p, double); 
  double *r = Calloc(n, double);
  double *s = Calloc(p, double);
  double *d1 = Calloc(n, double);
  double *d2 = Calloc(n, double);
  double *z = Calloc(p, double); // partial derivative used for screening: X^t*d1/n
  double cutoff;
  int *include = Calloc(p, int);
  int *nonconst = Calloc(p, int);
  int violations = 0, nv = 0; 
  
  // Preprocessing
  if (ppflag == 1) {
    standardize(x, x2, shift, scale, nonconst, n, p);
  } else if (ppflag == 2) {
    rescale(x, x2, shift, scale, nonconst, n, p);
  } else {
    simple_process(x, x2, nonconst, n, p, intercept);
  }

  if (scrflag == 0) {
    for (j=0; j<p; j++) if (nonconst[j]) include[j] = 1;
  } else {
    for (j=0; j<p; j++) if (pf[j] == 0.0 && nonconst[j]) include[j] = 1; // unpenalized coefficients
  }
  
  // Initialization
  nullDev = 0.0; // not divided by n
  for (i=0; i<n; i++) {
    r[i] = y[i];
    tmp = fabs(r[i]);
    if (tmp > gamma) {
      nullDev += tmp - gamma/2;
    } else {
      nullDev += tmp*tmp/(2*gamma);
    }
  }
  thresh = eps*nullDev;
  if (message) Rprintf("Threshold = %f\nGamma = %f\n", thresh, gamma);
  
  // Initial solution
  derivative_huber(d1, d2, r, gamma, n);
  init_huber(beta, beta_old, iter, x, x2, y, r, pf, d1, d2, nonconst, gamma, thresh, n, p, max_iter);
  
  // Set up lambda
  if (user == 0) {
    lambda[0] = maxprod(x, d1, n, p, pf, nonconst)/(n*alpha);
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
    if (message) Rprintf("Lambda 1\n# iterations = %d\n", iter[0]);
    lstart = 1;
  } else {
    lstart = 0;
  }

  for (j=0; j<p; j++) if (pf[j] && nonconst[j]) z[j] = crossprod(x, d1, n, j)/n;

  // Solution path
  for (l=lstart; l<nlam; l++) {
    if (message) Rprintf("Lambda %d\n", l+1);
    lp = l*p;
    l1 = lambda[l]*alpha;
    l2 = lambda[l]*(1.0-alpha);
    // Variable screening
    if (scrflag != 0) {
      if (scrfactor > 3.0) scrfactor = 3.0;
      if (l == 0) {
      	cutoff = alpha*lambda[0];
      } else {
        cutoff = alpha*((1.0+scrfactor)*lambda[l] - scrfactor*lambda[l-1]);
        ldiff = lambda[l-1] - lambda[l];
      }
      for (j=0; j<p; j++) {
        if (include[j] == 0 && nonconst[j] && fabs(z[j]) > cutoff * pf[j]) include[j] = 1;
      }
      if (scrflag == 1) scrfactor = 0.0; //reset scrfactor for ASR
    }
    while (iter[l] < max_iter) {
      // Check dfmax
      if (nnzero > dfmax) {
        for (ll = l; ll<nlam; ll++) iter[ll] = NA_INTEGER;
        saturated[0] = 1;
        break;
      }
      // Solve KKT equations on eligible predictors
      while (iter[l] < max_iter) {
        iter[l]++;
        max_update = 0.0;
        for (j=0; j<p; j++) {
          if (include[j]) {
            for (k=0; k<5; k++) {
              update = 0.0; mismatch = 0;
              // Calculate v1, v2
	      jn = j*n; v1 = 0.0; v2 = 0.0; pct = 0.0;
              for (i=0; i<n; i++) {
                v1 += x[jn+i]*d1[i];
                v2 += x2[jn+i]*d2[i];
                pct += d2[i];
              }
	      pct *= gamma/n; // percentage of residuals with absolute values below gamma
              if (pct < 0.05 || pct < 1.0/n || v2 == 0.0) {
                // approximate v2 with a continuation technique
                for (i=0; i<n; i++) {
                  tmp = fabs(r[i]);
                  if (tmp > gamma) v2 += x2[jn+i]/tmp;
                }
              }
              v1 /= n; v2 /= n;
              // Update beta_j
              if (pf[j] == 0.0) { // unpenalized
	        beta[lp+j] = beta_old[j] + v1/v2; 
              } else if (fabs(beta_old[j]+s[j]) > 1.0) { // active
                s[j] = sign(beta_old[j]+s[j]);
                beta[lp+j] = beta_old[j] + (v1-l1*pf[j]*s[j]-l2*pf[j]*beta_old[j])/(v2+l2*pf[j]); 
              } else { // inactive
                s[j] = (v1+v2*beta_old[j])/(l1*pf[j]);
                beta[lp+j] = 0.0;
              }
              // mismatch between beta and s
	      if (pf[j] > 0) {
                if (fabs(s[j]) > 1 || (beta[lp+j] != 0 && s[j] != sign(beta[lp+j]))) mismatch = 1;
              }
              // Update r, d1, d2 and compute candidate of max_update
              change = beta[lp+j]-beta_old[j];
              if (fabs(change) > 1e-6) {
                for (i=0; i<n; i++) {
		  r[i] -= x[jn+i]*change;
                  if (fabs(r[i]) > gamma) {
                    d1[i] = sign(r[i]);
                    d2[i] = 0.0;
                  } else {
	            d1[i] = r[i]*gi;
              	    d2[i] = gi;
                  }
                }
                update = (v2+l2*pf[j])*change*change*n;
                if (update > max_update) max_update = update;
                beta_old[j] = beta[lp+j];
              }
              if (!mismatch && update < thresh) break;
            }
          }
        }
        // Check for convergence
        if (max_update < thresh) break;
      }
      // Scan for violations of the screening rule and count nonzero variables
      violations = 0; nnzero = 0;
      if (scrflag != 0) {
        for (j=0; j<p; j++) {
	  if (include[j] == 0 && nonconst[j]) {
	    // pf[j] > 0, beta_old = beta = s = 0
            v1 = crossprod(x, d1, n, j)/n;
	    // Check for KKT conditions
	    if (fabs(v1) > l1*pf[j]) { 
	      include[j]=1; 
	      violations++;
              s[j] = v1/(l1*pf[j]);
              if (message) Rprintf("+V%d", j);
	    } else if (scrflag == 1) {
	      v3 = fabs(v1-z[j]);
              if (v3 > scrfactor) scrfactor = v3;
	    }
	    z[j] = v1;
	  }
	  if (beta_old[j] != 0) nnzero++;
        }
        scrfactor /= alpha*ldiff;
        if (message && violations > 0) Rprintf("\n");
      } else {
        for (j=0; j<p; j++) {
          if (beta_old[j] != 0) nnzero++;
        }
      }
      if (message) Rprintf("# iterations = %d\n", iter[l]);
      if (violations == 0) break;
      nv += violations;
    }
  }
  if (scrflag != 0 && message) Rprintf("# violations detected and fixed: %d\n", nv);
  numv[0] = nv;
  // Postprocessing
  if (ppflag) postprocess(beta, shift, scale, nonconst, nlam, p);

  Free(x2);
  Free(shift);
  Free(scale);
  Free(beta_old);
  Free(r);
  Free(s);
  Free(d1);
  Free(d2);
  Free(z);
  Free(include);
  Free(nonconst);
}

static void sncd_quantile(double *beta, int *iter, double *lambda, int *saturated, int *numv, double *x, double *y, 
			  double *pf, double *tau_, double *alpha_, double *eps_, double *lambda_min_, int *nlam_, 
			  int *n_, int *p_, int *ppflag_, int *scrflag_, int *intercept_, int *dfmax_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double tau = tau_[0]; double alpha = alpha_[0]; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0]; int scrflag = scrflag_[0]; int intercept = intercept_[0];
  int dfmax = dfmax_[0]; int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int m, i, j, k, l, ll, lp, jn, lstart, mismatch; 
  double lo, gamma, gi, pct, lstep, ldiff = 1.0, l1, l2, v1, v2, v3, tmp, change, nullDev, max_update, update, thresh; 
  double c = 2*tau-1.0; // coefficient for the linear term in quantile loss
  double scrfactor = 1.0; // variable screening factor
  //scrflag = 0: no screening; scrflag = 1: Adaptive Strong Rule(ASR); scrflag = 2: Strong Rule(SR)
  // ASR fits an appropriate scrfactor adaptively; SR always uses scrfactor = 1
  int nnzero = 0; // number of nonzero variables
  double *x2 = Calloc(n*p, double); // x^2
  double *shift = Calloc(p, double);
  double *scale = Calloc(p, double);
  double *beta_old = Calloc(p, double);
  double *r = Calloc(n, double);
  double *s = Calloc(p, double);
  double *d = Calloc(n, double);
  double *d1 = Calloc(n, double);
  double *d2 = Calloc(n, double);
  double *z = Calloc(p, double); // partial derivative used for screening: X^t*d1/n
  double cutoff;
  int *include = Calloc(p, int);
  int *nonconst = Calloc(p, int);
  int violations = 0, nv = 0;
  if (tau >= 0.05 && tau <= 0.95) {
      m = n/10 + 1;
      lo = 0.001;
  } else {
      m = n/100 + 1;
      lo = 0.0001;
  }
  // Preprocessing
  if (ppflag == 1) {
    standardize(x, x2, shift, scale, nonconst, n, p);
  } else if (ppflag == 2) {
    rescale(x, x2, shift, scale, nonconst, n, p);
  } else {
    simple_process(x, x2, nonconst, n, p, intercept);
  }
  
  if (scrflag == 0) {
    for (j=0; j<p; j++) if (nonconst[j]) include[j] = 1;
  } else {
    for (j=0; j<p; j++) if (pf[j] == 0.0 && nonconst[j]) include[j] = 1; // unpenalized coefficients
  }
  
  // Initialization
  nullDev = 0.0; // not divided by n
  for (i=0;i<n;i++) {
    r[i] = y[i];
    nullDev += fabs(r[i]) + c*r[i];
  }
  thresh = eps*nullDev;
  if (message) Rprintf("Threshold = %f\n", thresh);

  // Initial solution
  gamma = ksav(r, n, m);
  if (gamma < lo) gamma = lo;
  derivative_quantapprox(d1, d2, r, gamma, c, n);
  init_quantile(beta, beta_old, iter, x, x2, y, r, pf, d1, d2, nonconst, gamma, c, thresh, n, p, max_iter);

  // Set up lambda
  if (user == 0) {
    lambda[0] = maxprod(x, d1, n, p, pf, nonconst);
    for (i=0; i<n; i++) {
      if (fabs(r[i]) < 1e-10) {
        d[i] = c;
      } else {
        d[i] = sign(r[i])+c;
      }
    }
    tmp = maxprod(x, d, n, p, pf, nonconst);
    if (tmp > lambda[0]) lambda[0] = tmp;
    lambda[0] = lambda[0]/(2*n*alpha);
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
    if (message) Rprintf("Lambda 1: Gamma = %f\n# iterations = %d\n", gamma, iter[0]);
    lstart = 1;
  } else {
    lstart = 0;
  }

  for (j=0; j<p; j++) if (pf[j] && nonconst[j]) z[j] = crossprod(x, d1, n, j)/(2.0*n);
  
  // Solution path
  for (l=lstart; l<nlam; l++) {
    if (gamma > lo) {
      tmp = ksav(r, n, m);
      if (tmp < gamma) gamma = tmp;
    }
    if (gamma < lo) gamma = lo;
    gi = 1.0/gamma;
    if (message) Rprintf("Lambda %d: Gamma = %f\n", l+1, gamma);
    lp = l*p;
    l1 = lambda[l]*alpha;
    l2 = lambda[l]*(1.0-alpha);
    // Variable screening
    if (scrflag != 0) {
      if (scrfactor > 3.0) scrfactor = 3.0;
      if (l == 0) {
      	cutoff = alpha*lambda[0];
      } else {
      	cutoff = alpha*((1.0+scrfactor)*lambda[l] - scrfactor*lambda[l-1]);
        ldiff = lambda[l-1] - lambda[l];
      }
      for (j=0; j<p; j++) {
        if (include[j] == 0 && nonconst[j] && fabs(z[j]) > cutoff * pf[j]) include[j] = 1;
      }
      if (scrflag == 1) scrfactor = 0.0; //reset scrfactor for ASR
    }
    while (iter[l] < max_iter) {
      // Check dfmax
      if (nnzero > dfmax) {
        for (ll = l; ll<nlam; ll++) iter[ll] = NA_INTEGER;
        saturated[0] = 1;
        break;
      }
      // Solve KKT equations on eligible ones
      while (iter[l] < max_iter) {
        iter[l]++; max_update = 0.0;
        for (j=0; j<p; j++) {
          if (include[j]) {
            for (k=0; k<5; k++) {
              update = 0.0; mismatch = 0;
              // Calculate v1, v2
	      jn = j*n; v1 = 0.0; v2 = 0.0; pct = 0.0;
              for (i=0; i<n; i++) {
                v1 += x[jn+i]*d1[i];
                v2 += x2[jn+i]*d2[i];
                pct += d2[i];
              }
	      pct *= gamma/n; // percentage of residuals with absolute values below gamma
	      if (pct < 0.07 || pct < 1.0/n || v2 == 0.0) {
	        // approximate v2 with a continuation technique
	        for (i=0; i<n; i++) {
	      	  tmp = fabs(r[i]);
		  if (tmp > gamma) v2 += x2[jn+i]/tmp;
                }
	      }
	      v1 /= 2.0*n; v2 /= 2.0*n;
              // Update beta_j
              if (pf[j] == 0.0) { // unpenalized
	        beta[lp+j] = beta_old[j] + v1/v2;
              } else if (fabs(beta_old[j]+s[j]) > 1.0) { // active
                s[j] = sign(beta_old[j]+s[j]);
                beta[lp+j] = beta_old[j] + (v1-l1*pf[j]*s[j]-l2*pf[j]*beta_old[j])/(v2+l2*pf[j]);
              } else { // inactive
                s[j] = (v1+v2*beta_old[j])/(l1*pf[j]);
                beta[lp+j] = 0.0;
              }
              // mismatch between beta and s
	      if (pf[j] > 0) {
                if (fabs(s[j]) > 1 || (beta[lp+j] != 0 && s[j] != sign(beta[lp+j]))) mismatch = 1;
              }
	      // Update r, d1, d2 and compute candidate of max_update
              change = beta[lp+j]-beta_old[j];
              if (fabs(change) > 1e-6) {
                for (i=0; i<n; i++) {
		  r[i] -= x[jn+i]*change;
                  if (fabs(r[i]) > gamma) {
                    d1[i] = sign(r[i])+c;
                    d2[i] = 0.0;
                  } else {
		    d1[i] = r[i]*gi+c;
		    d2[i] = gi;
	          }
	        }
                update = (v2+l2*pf[j])*change*change*n*4.0;
                if (update > max_update) max_update = update;
                beta_old[j] = beta[lp+j];
              }
              if (!mismatch && update < thresh) break;
            }
          }
        }
        // Check for convergence
        if (max_update < thresh) break;
      }
      // Scan for violations of the screening rule and count nonzero variables
      violations = 0; nnzero = 0;
      if (scrflag != 0) {
        for (j=0; j<p; j++) {
	  if (include[j] == 0 && nonconst[j]) {
	    // pf[j] > 0, beta_old = beta = s = 0
            v1 = crossprod(x, d1, n, j)/(2*n);
	    // Check for KKT conditions
	    if (fabs(v1) > l1*pf[j]) { 
	      include[j]=1; 
	      violations++;
              s[j] = v1/(l1*pf[j]);
              if (message) Rprintf("+V%d", j);
	    } else if (scrflag == 1) {
	      v3 = fabs((v1-z[j]));
              if (v3 > scrfactor) scrfactor = v3;
	    }
	    z[j] = v1;
	  }
	  if (beta_old[j] != 0) nnzero++;
        }
        scrfactor /= alpha*ldiff;
        if (message && violations > 0) Rprintf("\n");
      } else {
        for (j=0; j<p; j++) {
          if (beta_old[j] != 0) nnzero++;
        }
      }
      if (message) Rprintf("# iterations = %d\n", iter[l]);
      if (violations == 0) break;
      nv += violations;
    }
  }
  if (scrflag != 0 && message) Rprintf("# KKT violations detected and fixed: %d\n", nv);
  numv[0] = nv;
  // Postprocessing
  if (ppflag) postprocess(beta, shift, scale, nonconst, nlam, p);

  Free(x2);
  Free(shift);
  Free(scale);
  Free(beta_old);
  Free(r);
  Free(s);
  Free(d);
  Free(d1);
  Free(d2);
  Free(z);
  Free(include);
  Free(nonconst);
}

static void sncd_squared(double *beta, int *iter, double *lambda, int *saturated, int *numv, double *x, double *y, 
			 double *pf, double *alpha_, double *eps_, double *lambda_min_, int *nlam_, int *n_, int *p_, 
			 int *ppflag_, int *scrflag_, int *intercept_, int *dfmax_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double alpha = alpha_[0]; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0]; int scrflag = scrflag_[0]; int intercept = intercept_[0];
  int dfmax = dfmax_[0]; int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, ll, lp, jn, lstart, mismatch; 
  double lstep, ldiff = 1.0, l1, l2, v1, v2, v3, tmp, change, nullDev, max_update, update, thresh;
  double scrfactor = 1.0; // variable screening factor
  //scrflag = 0: no screening; scrflag = 1: Adaptive Strong Rule(ASR); scrflag = 2: Strong Rule(SR)
  // ASR fits an appropriate scrfactor adaptively; SR always uses scrfactor = 1
  int nnzero = 0; // number of nonzero variables
  double *x2 = Calloc(n*p, double); // x^2
  double *x2m = Calloc(p, double); // Column means of x2
  double *shift = Calloc(p, double);
  double *scale = Calloc(p, double);
  double *beta_old = Calloc(p, double); 
  double *r = Calloc(n, double);
  double *s = Calloc(p, double);
  double *z = Calloc(p, double); // X^t * r/n
  double cutoff;
  int *include = Calloc(p, int);
  int *nonconst = Calloc(p, int);
  int violations = 0, nv = 0;

  // Preprocessing
  if (ppflag == 1) {
    standardize(x, x2, shift, scale, nonconst, n, p);
  } else if (ppflag == 2) {
    rescale(x, x2, shift, scale, nonconst, n, p);
  } else {
    simple_process(x, x2, nonconst, n, p, intercept);
  }
  
  if (scrflag == 0) {
    for (j=0; j<p; j++) if (nonconst[j]) include[j] = 1;
  } else {
    for (j=0; j<p; j++) if (pf[j] == 0.0 && nonconst[j]) include[j] = 1; // unpenalized coefficients
  }
  
  // Initialize r, z and assign x2m, nullDev
  nullDev = 0.0;
  for (i=0; i<n; i++) {
    r[i] = y[i];
    nullDev += pow(r[i],2); // without dividing by 2n
  }
  thresh = eps*nullDev;
  if (message) Rprintf("Threshold = %f\n", thresh);

  for (j=0; j<p; j++) {
    jn = j*n;
    tmp = 0.0;
    for (i=0; i<n; i++) tmp += x2[jn+i];
    x2m[j] = tmp/n;
  }
    
  // Initial solution
  init_squared(beta, beta_old, iter, x, x2m, y, r, pf, nonconst, thresh, n, p, ppflag, max_iter);
  
  // Set up lambda
  if (user == 0) {
    lambda[0] = maxprod(x, r, n, p, pf, nonconst)/(n*alpha);
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
    if (message) Rprintf("Lambda 1\n# iterations = %d\n", iter[0]);
    lstart = 1;
  } else {
    lstart = 0;
  }
  
  for (j=0; j<p; j++) if (pf[j] && nonconst[j]) z[j] = crossprod(x, r, n, j)/n;
  
  // Solution path
  for (l=lstart; l<nlam; l++) {
    if (message) Rprintf("Lambda %d\n", l+1);
    lp = l*p;
    l1 = lambda[l]*alpha;
    l2 = lambda[l]*(1.0-alpha);

    // Variable screening
    if (scrflag != 0) {
      if (scrfactor > 1.0) scrfactor = 1.0;
      if (l == 0) {
      	cutoff = alpha*lambda[0];
      } else {
      	cutoff = alpha*((1.0+scrfactor)*lambda[l] - scrfactor*lambda[l-1]);
        ldiff = lambda[l-1] - lambda[l];
      }
      for (j=0; j<p; j++) {
        if (include[j] == 0 && nonconst[j] && fabs(z[j]) > cutoff * pf[j]) include[j] = 1;
      }
      if (scrflag == 1) scrfactor = 0.0; //reset scrfactor for ASR
    }
    while (iter[l] < max_iter) {
      // Check dfmax
      if (nnzero > dfmax) {
        for (ll = l; ll<nlam; ll++) iter[ll] = NA_INTEGER;
        saturated[0] = 1;
        break;
      }
      // Solve KKT equations on eligible ones
      while (iter[l] < max_iter) {
        iter[l]++; max_update = 0.0;
        for (j=0; j<p; j++) {
          if (j == 0 && ppflag == 1) continue; // intercept is constant for standardized data
          if (include[j]) {
            for (k=0; k<5; k++) {
              update = 0.0; mismatch = 0;
	      // Update v1=z[j], v2=x2m[j]
              v1 = crossprod(x, r, n, j)/n; v2 = x2m[j];
              // Update beta_j
              if (pf[j] == 0.0) { // unpenalized
                beta[lp+j] = beta_old[j] + v1/v2;
              } else if (fabs(beta_old[j]+s[j]) > 1.0) { // active
                s[j] = sign(beta_old[j]+s[j]);
                beta[lp+j] = beta_old[j] + (v1-l2*pf[j]*beta_old[j]-l1*pf[j]*s[j])/(v2+l2*pf[j]);
              } else { // inactive
                s[j] = (v1+v2*beta_old[j])/(l1*pf[j]);
                beta[lp+j] = 0.0;
              }
              // mismatch between beta and s
              if (pf[j] > 0) {
                if (fabs(s[j]) > 1 || (beta[lp+j] != 0 && s[j] != sign(beta[lp+j]))) mismatch = 1;
              }
              // Update residuals
              change = beta[lp+j]-beta_old[j];       
              if (fabs(change) > 1e-6) {
                jn = j*n;
                for (i=0; i<n; i++) r[i] -= x[jn+i]*change;
                update = (v2+l2*pf[j])*change*change*n;
                if (update > max_update) max_update = update;
                beta_old[j] = beta[lp+j];
              }
              if (!mismatch && update < thresh) break;
            }
          }
        }             
        // Check for convergence
        if (max_update < thresh) break;
      }
      // Scan for violations of the screening rule and count nonzero variables
      violations = 0; nnzero = 0;
      if (scrflag != 0) {
        for (j=0; j<p; j++) {
	  if (include[j] == 0 && nonconst[j]) {
	    // pf[j] > 0, beta_old = beta = s = 0
            v1 = crossprod(x, r, n, j)/n;
	    // Check for KKT conditions
	    if (fabs(v1) > l1*pf[j]) {
	      include[j]=1; 
	      violations++;
              s[j] = v1/(l1*pf[j]);
              if (message) Rprintf("+V%d", j);
	    } else if (scrflag == 1) {
	      v3 = fabs((v1-z[j]));
              if (v3 > scrfactor) scrfactor = v3;
	    }
	    z[j] = v1;
	  }
	  if (beta_old[j] != 0) nnzero++;
        }
        scrfactor /= alpha*ldiff;
        if (message && violations > 0) Rprintf("\n");
      } else {
        for (j=0; j<p; j++) {
          if (beta_old[j] != 0) nnzero++;
        }
      }
      if (message) Rprintf("# iterations = %d\n", iter[l]);
      if (violations == 0) break;
      nv += violations;
    }
  }
  if (scrflag != 0 && message) Rprintf("# KKT violations detected and fixed: %d\n", nv);
  numv[0] = nv;
  // Postprocessing
  if (ppflag) postprocess(beta, shift, scale, nonconst, nlam, p);
  
  Free(x2);
  Free(x2m);
  Free(shift);
  Free(scale);
  Free(beta_old);
  Free(r);
  Free(s);
  Free(include);
  Free(nonconst);
}

// alpha = 0, pure l2 penalty
static void sncd_huber_l2(double *beta, int *iter, double *lambda, double *x, double *y, double *pf, double *gamma_, double *eps_, 
			  double *lambda_min_, int *nlam_, int *n_, int *p_, int *ppflag_, int *intercept_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double gamma = gamma_[0]; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0]; int intercept = intercept_[0];
  int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lp, jn; 
  double gi = 1.0/gamma, pct, lstep, v1, v2, tmp, change, nullDev, max_update, update, thresh;
  double *x2 = Calloc(n*p, double); // x^2
  double *shift = Calloc(p, double);
  double *scale = Calloc(p, double);
  double *beta_old = Calloc(p, double); 
  double *r = Calloc(n, double);
  double *d1 = Calloc(n, double);
  double *d2 = Calloc(n, double);
  int *nonconst = Calloc(p, int);
  
  // Preprocessing
  if (ppflag == 1) {
    standardize(x, x2, shift, scale, nonconst, n, p);
  } else if (ppflag == 2) {
    rescale(x, x2, shift, scale, nonconst, n, p);
  } else {
    simple_process(x, x2, nonconst, n, p, intercept);
  }

  // Initialization
  nullDev = 0.0; // not divided by n
  for (i=0;i<n;i++) {
    r[i] = y[i];
    tmp = fabs(r[i]);
    if (tmp > gamma) {
      nullDev += tmp - gamma/2;
    } else {
      nullDev += tmp*tmp/(2*gamma);
    }
  }
  thresh = eps*nullDev;
  derivative_huber(d1, d2, r, gamma, n); 
  if (message) Rprintf("Threshold = %f\nGamma = %f\n", thresh, gamma);
  
  // Set up lambda
  if (user == 0) {
    lambda[0] = maxprod(x, d1, n, p, pf, nonconst)/n*10;
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
  }

  // Solution path
  for (l=0; l<nlam; l++) {
    lp = l*p;
    while (iter[l] < max_iter) {
      iter[l]++;
      max_update = 0.0; 
      for (j=0; j<p; j++) {
      	if (nonconst[j]) {
      	  for (k=0; k<5; k++) {
       	    update = 0.0;
      	    // Calculate v1, v2
            jn = j*n; v1 = 0.0; v2 = 0.0; pct = 0.0;
            for (i=0; i<n; i++) {
              v1 += x[jn+i]*d1[i];
              v2 += x2[jn+i]*d2[i];
              pct += d2[i];
            }
            pct *= gamma/n; // percentage of residuals with absolute values below gamma
            if (pct < 0.05 || pct < 1.0/n || v2 == 0.0) {
              // approximate v2 with a continuation technique
              for (i=0; i<n; i++) {
                tmp = fabs(r[i]);
                if (tmp > gamma) v2 += x2[jn+i]/tmp;
              }
            }
            v1 /= n; v2 /= n;
            // Update beta_j
            if (pf[j] == 0.0) { // unpenalized
              beta[lp+j] = beta_old[j] + v1/v2; 
            } else {
              beta[lp+j] = beta_old[j] + (v1-lambda[l]*pf[j]*beta_old[j])/(v2+lambda[l]*pf[j]); 
            }
            // Update r, d1, d2 and compute candidate of max_update
            change = beta[lp+j]-beta_old[j];
            if (fabs(change) > 1e-6) {
              for (i=0; i<n; i++) {
                r[i] -= x[jn+i]*change;
                if (fabs(r[i]) > gamma) {
                  d1[i] = sign(r[i]);
                  d2[i] = 0.0;
                } else {
                  d1[i] = r[i]*gi;
                  d2[i] = gi;
                }
              }
              update = (v2+lambda[l]*pf[j])*change*change*n;
              if (update > max_update) max_update = update;
              beta_old[j] = beta[lp+j];
            }
            if (update < thresh) break;
          }
        }
      }
      // Check for convergence
      if (max_update < thresh) break;
    }
    if (message) Rprintf("Lambda %d: # iterations = %d\n", l+1, iter[l]);
  }
  // Postprocessing
  if (ppflag) postprocess(beta, shift, scale, nonconst, nlam, p);

  Free(x2);
  Free(shift);
  Free(scale);
  Free(beta_old);
  Free(r);
  Free(d1);
  Free(d2);
  Free(nonconst);
}

static void sncd_quantile_l2(double *beta, int *iter, double *lambda, double *x, double *y, double *pf, 
			     double *tau_, double *eps_, double *lambda_min_, int *nlam_, int *n_, int *p_, 
			     int *ppflag_, int *intercept_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double tau = tau_[0]; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0]; int intercept = intercept_[0];
  int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int m, i, j, k, l, lp, jn; 
  double lo, gamma, gi, pct, lstep, v1, v2, tmp, change, nullDev, max_update, update, thresh;
  double c = 2*tau-1.0; // coefficient for the linear term in quantile loss
  double *x2 = Calloc(n*p, double); // x^2
  double *shift = Calloc(p, double);
  double *scale = Calloc(p, double);
  double *beta_old = Calloc(p, double); 
  double *r = Calloc(n, double);
  double *d = Calloc(n, double);
  double *d1 = Calloc(n, double);
  double *d2 = Calloc(n, double);
  int *nonconst = Calloc(p, int);
  if (tau >= 0.05 && tau <= 0.95) {
      m = n/10 + 1;
      lo = 0.001;
  } else {
      m = n/100 + 1;
      lo = 0.0001;
  }
  
  // Preprocessing
  if (ppflag == 1) {
    standardize(x, x2, shift, scale, nonconst, n, p);
  } else if (ppflag == 2) {
    rescale(x, x2, shift, scale, nonconst, n, p);
  } else {
    simple_process(x, x2, nonconst, n, p, intercept);
  }

  // Initialization
  nullDev = 0.0; // not divided by n
  for (i=0;i<n;i++) {
    r[i] = y[i];
    nullDev += fabs(r[i]) + c*r[i];
  }
  thresh = eps*nullDev;
  gamma = ksav(r, n, m);
  if (gamma < lo) gamma = lo;
  derivative_quantapprox(d1, d2, r, gamma, c, n);
  if (message) Rprintf("Threshold = %f\n", thresh);
  
  // Set up lambda
  if (user == 0) {
    lambda[0] = maxprod(x, d1, n, p, pf, nonconst);
    for (i=0; i<n; i++) {
      if (fabs(r[i]) < 1e-10) {
        d[i] = c;
      } else {
        d[i] = sign(r[i])+c;
      } 
    }
    tmp = maxprod(x, d, n, p, pf, nonconst);
    if (tmp > lambda[0]) lambda[0] = tmp;
    lambda[0] = lambda[0]/(2*n)*10;
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
  }

  // Solution path
  for (l=0; l<nlam; l++) {
    if (gamma > lo && l > 0) {
      tmp = ksav(r, n, m);
      if (tmp < gamma) gamma = tmp;
    }
    if (gamma < lo) gamma = lo;
    gi = 1.0/gamma;
    lp = l*p;
    while (iter[l] < max_iter) {
      iter[l]++;
      max_update = 0.0; 
      for (j=0; j<p; j++) {
      	if (nonconst[j]) {
          for (k=0; k<5; k++) {
            update = 0.0;
            // Calculate v1, v2
            jn = j*n; v1 = 0.0; v2 = 0.0; pct = 0.0;
            for (i=0; i<n; i++) {
              v1 += x[jn+i]*d1[i];
              v2 += x2[jn+i]*d2[i];
              pct += d2[i];
            }
            pct *= gamma/n; // percentage of residuals with absolute values below gamma
            if (pct < 0.07 || pct < 1.0/n || v2 == 0.0) {
              // approximate v2 with a continuation technique
              for (i=0; i<n; i++) {
                tmp = fabs(r[i]);
	        if (tmp > gamma) v2 += x2[jn+i]/tmp;
              }
            }
            v1 /= 2.0*n; v2 /= 2.0*n;
            // Update beta_j
            if (pf[j] == 0.0) { // unpenalized
              beta[lp+j] = beta_old[j] + v1/v2; 
            } else {
              beta[lp+j] = beta_old[j] + (v1-lambda[l]*pf[j]*beta_old[j])/(v2+lambda[l]*pf[j]); 
            }
            // Update r, d1, d2 and compute candidate of max_update
            change = beta[lp+j]-beta_old[j];
            if (fabs(change) > 1e-6) {
              for (i=0; i<n; i++) {
                r[i] -= x[jn+i]*change;
                if (fabs(r[i]) > gamma) {
                  d1[i] = sign(r[i])+c;
                  d2[i] = 0.0;
                } else {
                  d1[i] = r[i]*gi+c;
                  d2[i] = gi;
                }
              }
              update = (v2+lambda[l]*pf[j])*change*change*n*4;
              if (update > max_update) max_update = update;
              beta_old[j] = beta[lp+j];
            }
            if (update < thresh) break;
          }
      	}
      }
      // Check for convergence
      if (max_update < thresh) break;
    }
    if (message) Rprintf("Lambda %d: Gamma = %f, # iterations = %d\n", l+1, gamma, iter[l]);
  }
  // Postprocessing
  if (ppflag) postprocess(beta, shift, scale, nonconst, nlam, p);
  
  Free(x2);
  Free(shift);
  Free(scale);
  Free(beta_old);
  Free(r);
  Free(d);
  Free(d1);
  Free(d2);
  Free(nonconst);
}

static void sncd_squared_l2(double *beta, int *iter, double *lambda, double *x, double *y, double *pf, 
			    double *eps_, double *lambda_min_, int *nlam_, int *n_, int *p_, int *ppflag_, 
                            int *intercept_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0]; int intercept = intercept_[0];
  int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lp, jn; 
  double lstep, v1, v2, tmp, change, nullDev, max_update, update, thresh;
  double *x2 = Calloc(n*p, double); // x^2
  double *x2m = Calloc(p, double); // Column means of x2
  double *shift = Calloc(p, double);
  double *scale = Calloc(p, double);
  double *beta_old = Calloc(p, double); 
  double *r = Calloc(n, double);
  int *nonconst = Calloc(p, int);
  
  // Preprocessing
  if (ppflag == 1) {
    standardize(x, x2, shift, scale, nonconst, n, p);
  } else if (ppflag == 2) {
    rescale(x, x2, shift, scale, nonconst, n, p);
  } else {
    simple_process(x, x2, nonconst, n, p, intercept);
  }
  
  // Initialization
  nullDev = 0.0;
  for (i=0; i<n; i++) {
    r[i] = y[i];
    nullDev += pow(r[i],2); // without dividing by 2n
  }
  thresh = eps*nullDev;
  if (message) Rprintf("Threshold = %f\n", thresh);
  for (j=0; j<p; j++) {
    jn = j*n; tmp = 0.0;
    for (i=0; i<n; i++) tmp += x2[jn+i];
    x2m[j] = tmp/n;
  }

  // Set up lambda
  if (user == 0) {
    lambda[0] = maxprod(x, r, n, p, pf, nonconst)/n*10;
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
  }

  // Solution path
  for (l=0; l<nlam; l++) {
    lp = l*p;
    while (iter[l] < max_iter) {
      iter[l]++;
      max_update = 0.0; 
      for (j=0; j<p; j++) {
        if (j == 0 && ppflag == 1) continue; // intercept is constant for standardized data
        if (nonconst[j]) {
      	  for (k=0; k<5; k++) {
            update = 0.0;
            // Update v1, v2=x2m[j]
            v1 = crossprod(x, r, n, j)/n; v2 = x2m[j];
            // Update beta_j
            if (pf[j] == 0.0) { // unpenalized
              beta[lp+j] = beta_old[j] + v1/v2;
            } else {
              beta[lp+j] = beta_old[j] + (v1-lambda[l]*pf[j]*beta_old[j])/(v2+lambda[l]*pf[j]);
            }
            // Update r
            change = beta[lp+j]-beta_old[j];       
            if (fabs(change) > 1e-6) {
              jn = j*n;              
              for (i=0; i<n; i++) r[i] -= x[jn+i]*change;
              update = (v2+lambda[l]*pf[j])*change*change*n;
              if (update > max_update) max_update = update;
              beta_old[j] = beta[lp+j];
            }
            if (update < thresh) break;
          }
        }
      }
      // Check for convergence
      if (max_update < thresh) break;
    }
    if (message) Rprintf("Lambda %d: # iterations = %d\n", l+1, iter[l]);
  }
  // Postprocessing
  if (ppflag) postprocess(beta, shift, scale, nonconst, nlam, p);

  Free(x2);
  Free(x2m);
  Free(shift);
  Free(scale);
  Free(beta_old);
  Free(r);
  Free(nonconst);
}


static const R_CMethodDef cMethods[] = {
  {"huber", (DL_FUNC) &sncd_huber, 22},
  {"quant", (DL_FUNC) &sncd_quantile, 22},
  {"squared", (DL_FUNC) &sncd_squared, 21},
  {"huber_l2", (DL_FUNC) &sncd_huber_l2, 17},
  {"quant_l2", (DL_FUNC) &sncd_quantile_l2, 17},
  {"squared_l2", (DL_FUNC) &sncd_squared_l2, 16},
  {NULL}
};

void R_init_hqreg(DllInfo *info)
{
  R_registerRoutines(info,cMethods,NULL,NULL,NULL);
  R_useDynamicSymbols(info, FALSE);
  R_forceSymbols(info, TRUE);
}
