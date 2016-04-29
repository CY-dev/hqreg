#include <math.h>
#include <time.h>
#include <string.h>
#include <R.h>
#include <R_ext/Applic.h>
#include "Rinternals.h"
#include "R_ext/Rdynload.h"

double sign(double x);
double crossprod(double *x, double *v, int n, int j);
double maxprod(double *x, double *v, int n, int p, double *pf);
double ksav(double *a, int size, int K);
void standardize(double *x, double *x2, double *shift, double *scale, int n, int p);
void rescale(double *x, double *x2, double *shift, double *scale, int n, int p);
void postprocess(double *beta, double *shift, double *scale, int nlam, int p);
void init_huber(double *beta, double *beta_old, int *iter, double *x, double *x2, 
		double *y, double *r, double *pf, double *d1, double *d2, double gamma,
		double thresh, int n, int p, int max_iter);
void init_quantile(double *beta, double *beta_old, int *iter, double *x, double *x2, 
		   double *y, double *r, double *pf, double *d1, double *d2, double gamma,
		   double c, double thresh, int n, int p, int max_iter);
void init_squared(double *beta, double *beta_old, int *iter, double *x, double *x2bar, double *y, 
		  double *r, double *pf, double thresh, int n, int p, int ppflag, int max_iter);

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
		       int *n_, int *p_, int *ppflag_, int *scrflag_, int *dfmax_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double gamma = gamma_[0]; double alpha = alpha_[0]; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0]; int scrflag = scrflag_[0];
  int dfmax = dfmax_[0]; int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lp, jn, converged, mismatch; 
  double pct, lstep, ldiff, lmax, l1, l2, v1, v2, v3, temp, change, nullDev, max_update, update, thresh;
  double gi = 1.0/gamma; // 1/gamma as a multiplier
  double scrfactor = 1.0; // scaling factor used for screening rule
  int nnzero = 0; // number of nonzero variables
  double *x2 = Calloc(n*p, double); // x^2
  for (i=0; i<n; i++) x2[i] = 1.0; // column of 1's for intercept
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
  //scrflag = 0: no screening; scrflag = 1: Adaptive Strong Rule(ASR); scrflag = 2: Strong Rule(SR)
  // ASR fits an appropriate scrfactor adaptively; SR always uses scrfactor = 1
  if (scrflag == 0) {
    for (j=0; j<p; j++) include[j] = 1;
  } else {
    for (j=0; j<p; j++) if(pf[j] == 0.0) include[j] = 1; // include unpenalized coefficients
  }
  int violations = 0, nv = 0; 
  
  // Preprocessing
  if (ppflag == 1) {
    standardize(x, x2, shift, scale, n, p);
  } else if (ppflag == 2) {
    rescale(x, x2, shift, scale, n, p);
  } else {
    for (j=1; j<p; j++) {
      jn = j*n;
      for (i=0; i<n; i++) x2[jn+i]=pow(x[jn+i],2);
    }
  }

  // Initialization
  nullDev = 0.0; // not divided by n
  for (i=0;i<n;i++) {
    r[i] = y[i];
    temp = fabs(r[i]);
    if (temp>gamma) {
      nullDev += temp - gamma/2;
    } else {
      nullDev += temp*temp/(2*gamma);
    }
  }
  thresh = eps*nullDev;
  if (message) Rprintf("threshold = %f\n", thresh);
  derivative_huber(d1, d2, r, gamma, n);

  // Find initial solutions for lambda[0]
  init_huber(beta, beta_old, iter, x, x2, y, r, pf, d1, d2, gamma, thresh, n, p, max_iter);

  // Set up lambda
  if (user==0) {
    lambda[0] = maxprod(x, d1, n, p, pf)/(n*alpha);
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
  }

  for (j=0; j<p; j++) if (pf[j] > 0) z[j] = crossprod(x, d1, n, j)/n;

  // Solution path
  for (l=1; l<nlam; l++) {
    converged = 0; lp = l*p;
    l1 = lambda[l]*alpha;
    l2 = lambda[l]*(1.0-alpha);
    // Variable screening
    if (scrflag != 0) {
      if (scrfactor>5.0) scrfactor = 5.0;
      cutoff = alpha*((1.0+scrfactor)*lambda[l] - scrfactor*lambda[l-1]);
      ldiff = lambda[l-1] - lambda[l];
      for (j=1; j<p; j++) {
        if(include[j] == 0 && fabs(z[j]) > (cutoff * pf[j])) include[j] = 1;
      }
      scrfactor = 1.0; //reset scrfactor
    }
    while(iter[l] < max_iter) {
      converged = 0;
      // Check dfmax
      if (nnzero > dfmax) {
        for (int ll = l; ll<nlam; ll++) iter[ll] = NA_INTEGER;
        saturated[0] = 1;
        break;
      }

      // Solve KKT equations on eligible predictors
      while(iter[l]<max_iter) {
        iter[l]++;
        mismatch = 0; max_update = 0.0;
        for (j=0; j<p; j++) {
          if (include[j]) {
            // Calculate v1, v2
	    jn = j*n; v1 = 0.0; v2 = 0.0; pct = 0.0;
            for (i=0; i<n; i++) {
              v1 += x[jn+i]*d1[i];
              v2 += x2[jn+i]*d2[i];
              pct += d2[i];
            }
	    pct = pct*gamma/n; // percentage of residuals with absolute values below gamma
	    if (pct < 0.05 || pct < 1.0/n) {
	      // approximate v2 with a continuation technique
	      for (i=0; i<n; i++) {
	      	temp = fabs(r[i]);
		if (temp > gamma) v2 += x2[jn+i]/temp;
              }
	    }
	    v1 = v1/n; v2 = v2/n;
            // Update beta_j
            if (pf[j]==0.0) { // unpenalized
	      beta[lp+j] = beta_old[j] + v1/v2; 
            } else if (fabs(beta_old[j]+s[j])>1.0) { // active
              s[j] = sign(beta_old[j]+s[j]);
              beta[lp+j] = beta_old[j] + (v1-l1*pf[j]*s[j]-l2*pf[j]*beta_old[j])/(v2+l2*pf[j]); 
            } else {
              s[j] = (v1+v2*beta_old[j])/(l1*pf[j]);
              beta[lp+j] = 0.0;
            }
            // mark the first mismatch between beta and s
	    if (!mismatch && pf[j] > 0) {
              if (fabs(s[j]) > 1 || (beta[lp+j] != 0 && s[j] != sign(beta[lp+j])))
		 mismatch = 1;
            }
	    // Update r, d1, d2 and compute candidate of max_update
            change = beta[lp+j]-beta_old[j];
            if (fabs(change) > 1e-6) {
	      //v2 = 0.0;
              for (i=0; i<n; i++) {
		r[i] -= x[jn+i]*change;
                if (fabs(r[i])>gamma) {
                  d1[i] = sign(r[i]);
                  d2[i] = 0.0;
                } else {
		  d1[i] = r[i]*gi;
		  d2[i] = gi;
	          //v2 += x2[jn+i]*d2[i];
	        }
	      }
	      //v2 += n*l2*pf[j];
	      update = (v2+l2*pf[j])*change*change*n;
              if (update>max_update) max_update = update;
              beta_old[j] = beta[lp+j];
            }
          }
        }
        // Check for convergence
        if (iter[l] > 1) {
          if (!mismatch && max_update < thresh) {
            converged = 1;
	    break;
	  }
        }
      }
      // Scan for violations of the screening rule and count nonzero variables
      violations = 0; nnzero = 0;
      if (scrflag != 0) {
        for (j=0; j<p; j++) {
	  if (include[j]==0) {
            v1 = crossprod(x, d1, n, j)/n;
	    // Check for KKT conditions
	    if (fabs(v1)>l1*pf[j]) { 
	      include[j]=1; 
	      violations++;
	      // pf[j] > 0
	      // beta_old = beta = d = 0, no need for judgement
              s[j] = v1/(l1*pf[j]);
              if (violations == 1 & message) Rprintf("Lambda %d\n", l+1);
              if (message) Rprintf("+V%d", j);
	    } else if (scrflag == 1 && ldiff != 0) {
	      v3 = fabs((v1-z[j])/(pf[j]*ldiff*alpha));
              if (v3 > scrfactor) scrfactor = v3;
	    }
	    z[j] = v1;
	  } else if (beta_old[j] != 0) nnzero++;
        }
        if (violations>0 && message) Rprintf("\n");
      } else {
        for (j=0; j<p; j++) {
          if (beta_old[j] != 0) nnzero++;
        }
      }
      if (violations==0) break;
      nv += violations;
    }
    //Rprintf("iter[%d] = %d, beta[0] = %f\n", l+1, iter[l], beta[l*p]);
  }
  if (scrflag != 0 && message) Rprintf("# violations detected and fixed: %d\n", nv);
  numv[0] = nv;
  // Postprocessing
  if (ppflag) postprocess(beta, shift, scale, nlam, p);

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
}

static void sncd_quantile(double *beta, int *iter, double *lambda, int *saturated, int *numv, double *x, double *y, 
			  double *pf, double *tau_, double *alpha_, double *eps_, double *lambda_min_, int *nlam_, 
			  int *n_, int *p_, int *ppflag_, int *scrflag_, int *dfmax_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double tau = tau_[0]; double alpha = alpha_[0]; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0]; int scrflag = scrflag_[0];
  int dfmax = dfmax_[0]; int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lp, jn, converged, mismatch; 
  double gamma, gi, pct, lstep, ldiff, lmax, l1, l2, v1, v2, v3, temp, change, nullDev, max_update, update, thresh; 
  double c = 2*tau-1.0; // coefficient for the linear term in quantile loss
  double scrfactor = 1.0; // variable screening factor
  int nnzero = 0; // number of nonzero variables
  double *x2 = Calloc(n*p, double); // x^2
  for (i=0; i<n; i++) x2[i] = 1.0; // column of 1's for intercept
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
  //scrflag = 0: no screening; scrflag = 1: Adaptive Strong Rule(ASR); scrflag = 2: Strong Rule(SR)
  // ASR fits an appropriate scrfactor adaptively; SR always uses scrfactor = 1
  if (scrflag == 0) {
    for (j=0; j<p; j++) include[j] = 1;
  } else {
    for (j=0; j<p; j++) if (pf[j] == 0.0) include[j] = 1; // include unpenalized coefficients
  }
  int violations = 0, nv = 0;
  int m = n/10;

  // Preprocessing
  if (ppflag == 1) {
    standardize(x, x2, shift, scale, n, p);
  } else if (ppflag == 2) {
    rescale(x, x2, shift, scale, n, p);
  } else {
    for (j=1; j<p; j++) {
      jn = j*n;
      for (i=0; i<n; i++) x2[jn+i]=pow(x[jn+i],2);
    }
  }

  // Initialization
  nullDev = 0.0; // not divided by n
  for (i=0;i<n;i++) {
    r[i] = y[i];
    nullDev += fabs(r[i]) + c*r[i];
  }
  thresh = eps*nullDev;
  gamma = ksav(r, n, m);
  if (gamma<0.001) gamma = 0.001;
  derivative_quantapprox(d1, d2, r, gamma, c, n);

  // Find initial solutions for lambda[0]
  init_quantile(beta, beta_old, iter, x, x2, y, r, pf, d1, d2, gamma, c, thresh, n, p, max_iter);

  // Set up lambda
  if (user==0) {
    lambda[0] = maxprod(x, d1, n, p, pf);
    for (i=0; i<n; i++) {
      if (fabs(r[i]) < 1e-10) {
        d[i] = c;
      } else {
        d[i] = sign(r[i])+c;
      }
    }
    temp = maxprod(x, d, n, p, pf);
    if (temp>lambda[0]) lambda[0] = temp;
    lambda[0] = lambda[0]/(2*n*alpha);
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
  }

  for (j=0; j<p; j++) if (pf[j] > 0) z[j] = crossprod(x, d1, n, j)/(2*n);
  
  // Solution path
  for (l=1; l<nlam; l++) {
    if (gamma>0.001) {
      temp = ksav(r, n, m);
      if (temp < gamma) gamma = temp;
    }
    if (gamma<0.001) gamma = 0.001;
    gi = 1.0/gamma;
    if (message) Rprintf("Lambda %d: Gamma = %f\n", l+1, gamma);
    converged = 0; lp = l*p;
    l1 = lambda[l]*alpha;
    l2 = lambda[l]*(1.0-alpha);
    // Variable screening
    if (scrflag != 0) {
      if (scrfactor>5.0) scrfactor = 5.0;
      cutoff = alpha*((1.0+scrfactor)*lambda[l] - scrfactor*lambda[l-1]);
      ldiff = lambda[l-1] - lambda[l];
      for (j=1; j<p; j++) {
        if (include[j] == 0 && fabs(z[j]) > (cutoff * pf[j])) include[j] = 1;
      }
      scrfactor = 1.0; //reset scrfactor
    }
    
    while(iter[l] < max_iter) {
      converged = 0;
      // Check dfmax
      if (nnzero > dfmax) {
        for (int ll = l; ll<nlam; ll++) iter[ll] = NA_INTEGER;
        saturated[0] = 1;
        break;
      }

      // Solve KKT equations on eligible ones
      while(iter[l]<max_iter) {
        iter[l]++;
        mismatch = 0; max_update = 0.0;
        for (j=0; j<p; j++) {
          if (include[j]) {
            int it = 0;
            update = 2*thresh;
            while (update > thresh && it < 5) {
            it++;
            // Calculate v1, v2
	    jn = j*n; v1 = 0.0; v2 = 0.0; pct = 0.0;
            for (i=0; i<n; i++) {
              v1 += x[jn+i]*d1[i];
              v2 += x2[jn+i]*d2[i];
              pct += d2[i];
            }
	    pct = pct*gamma/n; // percentage of residuals with absolute values below gamma
	    if (pct < 0.08 || pct < 1.0/n) {
	      // approximate v2 with a continuation technique
	      for (i=0; i<n; i++) {
	      	temp = fabs(r[i]);
		if (temp > gamma) v2 += x2[jn+i]/temp;
              }
	    }
	    v1 = v1/(2.0*n); v2 = v2/(2.0*n);
            // Update beta_j
            if (pf[j]==0.0) { // unpenalized
	      beta[lp+j] = beta_old[j] + v1/v2;
            } else if (fabs(beta_old[j]+s[j])>1.0) { // active
              s[j] = sign(beta_old[j]+s[j]);
              beta[lp+j] = beta_old[j] + (v1-l1*pf[j]*s[j]-l2*pf[j]*beta_old[j])/(v2+l2*pf[j]);
            } else {
              s[j] = (v1+v2*beta_old[j])/(l1*pf[j]);
              beta[lp+j] = 0.0;
            }
            // mark the first mismatch between beta and s
	    if (!mismatch && pf[j] > 0) {
              if (fabs(s[j]) > 1 || (beta[lp+j] != 0 && s[j] != sign(beta[lp+j])))
		 mismatch = 1;
            }
	    // Update r, d1, d2 and compute candidate of max_update
            change = beta[lp+j]-beta_old[j];
            if (fabs(change) > 1e-6) {
	      //v2 = 0.0;
              for (i=0; i<n; i++) {
		r[i] -= x[jn+i]*change;
                if (fabs(r[i])>gamma) {
                  d1[i] = sign(r[i])+c;
                  d2[i] = 0.0;
                } else {
		  d1[i] = r[i]*gi+c;
		  d2[i] = gi;
	          //v2 += x2[jn+i]*d2[i];
	        }
	      }
	      //v2 += 2*n*l2*pf[j];
	      //update = v2*change*change;
              update = (v2+l2*pf[j])*change*change*n;
              if (update>max_update) max_update = update;
              beta_old[j] = beta[lp+j];
            } else {update = 0;}
            if(l == 99) Rprintf("beta[%d] = %f, update = %f, thresh = %f\n", j, beta[lp+j], update, thresh);
            }
          }
        }
        // Check for convergence
        if (iter[l] > 1) {
          if (!mismatch && max_update < thresh) {
            converged = 1;
	    break;
	  }
        }
      }
      // Scan for violations of the screening rule and count nonzero variables
      violations = 0; nnzero = 0;
      if (scrflag != 0) {
        for (j=0; j<p; j++) {
	  if (include[j]==0) {
            v1 = crossprod(x, d1, n, j)/(2*n);
	    // Check for KKT conditions
	    if (fabs(v1)>l1*pf[j]) { 
	      include[j]=1; 
	      violations++;
	      // pf[j] > 0
	      // beta_old = beta = d = 0, no need for judgement
              s[j] = v1/(l1*pf[j]);
              if (message) Rprintf("+V%d", j);
	    } else if (scrflag == 1 && ldiff != 0) {
	      v3 = fabs((v1-z[j])/(pf[j]*ldiff*alpha));
              if (v3 > scrfactor) scrfactor = v3;
	    }
	    z[j] = v1;
	  }
          if (beta_old[j] != 0) nnzero++;
        }
        if (violations>0 && message) Rprintf("\n");
      } else {
        for (j=0; j<p; j++) {
          if (beta_old[j] != 0) nnzero++;
        }
      }
      if (violations==0) break;
      nv += violations;
    }
    //if (message) Rprintf("# iterations = %d\n", iter[l]);
  }
  if (scrflag != 0 && message) Rprintf("# KKT violations detected and fixed: %d\n", nv);
  numv[0] = nv;
  // Postprocessing
  if (ppflag) postprocess(beta, shift, scale, nlam, p);

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
}

static void sncd_squared(double *beta, int *iter, double *lambda, int *saturated, int *numv, double *x, double *y, 
			 double *pf, double *alpha_, double *eps_, double *lambda_min_, int *nlam_, int *n_, int *p_, 
			 int *ppflag_, int *scrflag_, int *dfmax_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double alpha = alpha_[0]; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0]; int scrflag = scrflag_[0];
  int dfmax = dfmax_[0]; int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lp, jn, converged, mismatch; 
  double lstep, ldiff, lmax, l1, l2, v1, v2, v3, temp, change, nullDev, max_update, update, thresh, scrfactor = 1.0;
  int nnzero = 0; // number of nonzero variables
  double *x2 = Calloc(n*p, double); // x^2
  for (i=0; i<n; i++) x2[i] = 1.0; // column of 1's for intercept
  double *x2bar = Calloc(p, double); // Col Mean of x2
  x2bar[0] = 1.0;
  double *shift = Calloc(p, double);
  double *scale = Calloc(p, double);
  double *beta_old = Calloc(p, double); 
  double *r = Calloc(n, double);
  double *s = Calloc(p, double);
  double *z = Calloc(p, double); // X^t * r/n
  double cutoff;
  int *include = Calloc(p, int);
  //scrflag = 0: no screening; scrflag = 1: Adaptive Strong Rule(ASR); scrflag = 2: Strong Rule(SR)
  // ASR fits an appropriate scrfactor adaptively; SR always uses scrfactor = 1
  if (scrflag == 0) {
    for (j=0; j<p; j++) include[j] = 1;
  } else {
    for (j=0; j<p; j++) if (pf[j] == 0.0) include[j] = 1; // include unpenalized coefficients
  }
  int violations = 0, nv = 0;

  // Preprocessing
  if (ppflag == 1) {
    standardize(x, x2, shift, scale, n, p);
  } else if (ppflag == 2) {
    rescale(x, x2, shift, scale, n, p);
  } else {
    for (j=1; j<p; j++) {
      jn = j*n;
      for (i=0; i<n; i++) x2[jn+i]=pow(x[jn+i],2);
    }
  }
  
  // Initialize r, z and assign x2bar, nullDev
  nullDev = 0.0;
  for (i=0; i<n; i++) {
    r[i] = y[i];
    nullDev += pow(r[i],2); // without dividing by 2n
  }
  thresh = eps*nullDev;

  for (j=0; j<p; j++) {
    jn = j*n;
    temp = 0.0;
    for (i=0; i<n; i++) temp += x2[jn+i];
    x2bar[j] = temp/n;
  }

  // Find initial solutions for lambda[0]
  init_squared(beta, beta_old, iter, x, x2bar, y, r, pf, thresh, n, p, ppflag, max_iter);
  
  // Set up lambda
  if (user==0) {
    lambda[0] = maxprod(x, r, n, p, pf)/(n*alpha);
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
  }
  
  for (j=0; j<p; j++) if (pf[j] > 0) z[j] = crossprod(x, r, n, j)/n;
  
  // Solution path
  for (l=1; l<nlam; l++) {
    converged = 0; lp = l*p;
    l1 = lambda[l]*alpha;
    l2 = lambda[l]*(1.0-alpha);

    // Variable screening
    if (scrflag != 0) {
      if (scrfactor>5.0) scrfactor = 5.0;
      cutoff = alpha*((1.0+scrfactor)*lambda[l] - scrfactor*lambda[l-1]);
      ldiff = lambda[l-1] - lambda[l];
      for (j=1; j<p; j++) {
        if (include[j] == 0 && fabs(z[j]) > (cutoff * pf[j])) include[j] = 1;
      }
      scrfactor = 1.0; //reset scrfactor
    }
    while(iter[l] < max_iter) {
      converged = 0;
      // Check dfmax
      if (nnzero > dfmax) {
        for (int ll = l; ll<nlam; ll++) iter[ll] = NA_INTEGER;
        saturated[0] = 1;
        break;
      }
      // Solve KKT equations on eligible ones
      while(iter[l]<max_iter) {
        iter[l]++;
	mismatch=0; max_update = 0.0;
        for (j=0; j<p; j++) {
          if (j == 0 && ppflag == 1) continue; // intercept is constant for standardized data
          if (include[j]) {
            // Update v1=z[j], v2=x2bar[j]
      	    v1 = crossprod(x, r, n, j)/n; v2 = x2bar[j];
            // Update beta_j
            if (pf[j]==0.0) { // unpenalized
	      beta[lp+j] = beta_old[j] + v1/v2;
	    } else if (fabs(beta_old[j]+s[j])>1.0) { // active
              s[j] = sign(beta_old[j]+s[j]);
              beta[lp+j] = beta_old[j] + (v1-l2*pf[j]*beta_old[j]-l1*pf[j]*s[j])/(v2+l2*pf[j]);
            } else {
              s[j] = (v1+v2*beta_old[j])/(l1*pf[j]);
              beta[lp+j] = 0.0;
            }
            // mark the first mismatch between beta and s
	    if (!mismatch && pf[j] > 0) {
              if (fabs(s[j]) > 1 || (beta[lp+j] != 0 && s[j] != sign(beta[lp+j])))
		 mismatch = 1;
            }
	    // Update residuals
            change = beta[lp+j]-beta_old[j];       
            if (fabs(change) > 1e-6) {
	      jn = j*n;
              for (i=0; i<n; i++) r[i] -= x[jn+i]*change;
	      update = (v2+l2*pf[j])*change*change*n;
	      if (update>max_update) max_update = update;
	      beta_old[j] = beta[lp+j];
            }
          }
        }             
        // Check for convergence
        if (iter[l] > 1) {
          if (!mismatch && max_update < thresh) {
            converged = 1;
	    break;
	  }
        }
      }
      // Scan for violations of the screening rule and count nonzero variables
      violations = 0; nnzero = 0;
      if (scrflag != 0) {
        for (j=0; j<p; j++) {
	  if (include[j]==0) {
            v1 = crossprod(x, r, n, j)/n;
	    // Check for KKT conditions
	    if (fabs(v1)>l1*pf[j]) {
	      include[j]=1; 
	      violations++;
	      // pf[j] > 0
	      // beta_old = beta = d = 0, no need for judgement
              s[j] = v1/(l1*pf[j]);
              if (violations == 1 & message) Rprintf("Lambda %d\n", l+1);
              if (message) Rprintf("+V%d", j);
	    } else if (scrflag == 1 && ldiff != 0) {
	      v3 = fabs((v1-z[j])/(pf[j]*ldiff*alpha));
              if (v3 > scrfactor) scrfactor = v3;
	    }
	    z[j] = v1;
	  }
          if (beta_old[j] != 0) nnzero++;
        }
        if (violations>0 && message) Rprintf("\n");
      } else {
        for (j=0; j<p; j++) {
          if (beta_old[j] != 0) nnzero++;
        }
      }
      if (violations==0) break;
      nv += violations;
    }
  }
  if (scrflag != 0 && message) Rprintf("# KKT violations detected and fixed: %d\n", nv);
  numv[0] = nv;
  // Postprocessing
  if (ppflag) postprocess(beta, shift, scale, nlam, p);
  
  Free(x2);
  Free(x2bar);
  Free(shift);
  Free(scale);
  Free(beta_old);
  Free(r);
  Free(s);
  Free(include);
}

// alpha = 0, pure l2 penalty
static void sncd_huber_l2(double *beta, int *iter, double *lambda, double *x, double *y, double *pf, double *gamma_, double *eps_, 
			  double *lambda_min_, int *nlam_, int *n_, int *p_, int *ppflag_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double gamma = gamma_[0]; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0];
  int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lp, jn, converged; 
  double gi = 1.0/gamma, pct, lstep, ldiff, lmax, v1, v2, v3, temp, change, nullDev, max_update, update, thresh;
  double *x2 = Calloc(n*p, double); // x^2
  for (i=0; i<n; i++) x2[i] = 1.0; // column of 1's for intercept
  double *shift = Calloc(p, double);
  double *scale = Calloc(p, double);
  double *beta_old = Calloc(p, double); 
  double *r = Calloc(n, double);
  double *d1 = Calloc(n, double);
  double *d2 = Calloc(n, double);
  
  // Preprocessing
  if (ppflag == 1) {
    standardize(x, x2, shift, scale, n, p);
  } else if (ppflag == 2) {
    rescale(x, x2, shift, scale, n, p);
  } else {
    for (j=1; j<p; j++) {
      jn = j*n;
      for (i=0; i<n; i++) x2[jn+i]=pow(x[jn+i],2);
    }
  }

  // Initialization
  nullDev = 0.0; // not divided by n
  for (i=0;i<n;i++) {
    r[i] = y[i];
    temp = fabs(r[i]);
    if (temp>gamma) {
      nullDev += temp - gamma/2;
    } else {
      nullDev += temp*temp/(2*gamma);
    }
  }
  thresh = eps*nullDev;
  //if (message) Rprintf("threshold = %f\n", thresh);
  derivative_huber(d1, d2, r, gamma, n); 

  // Set up lambda
  if (user==0) {
    lambda[0] = maxprod(x, d1, n, p, pf)/n*10;
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
  }

  // Solution path
  for (l=0; l<nlam; l++) {
    converged = 0; lp = l*p;
    while(iter[l] < max_iter) {
      iter[l]++;
      max_update = 0.0; 
      for (j=0; j<p; j++) {
        // Calculate v1, v2
	jn = j*n; v1 = 0.0; v2 = 0.0; pct = 0.0;
        for (i=0; i<n; i++) {
          v1 += x[jn+i]*d1[i];
          v2 += x2[jn+i]*d2[i];
          pct += d2[i];
        }
	pct = pct*gamma/n; // percentage of residuals with absolute values below gamma
	if (pct < 0.05 || pct < 1.0/n) {
	  // approximate v2 with a continuation technique
	  for (i=0; i<n; i++) {
	    temp = fabs(r[i]);
	    if (temp > gamma) v2 += x2[jn+i]/temp;
	  }
	}
	v1 = v1/n; v2 = v2/n;
        // Update beta_j
        if (pf[j]==0.0) { // unpenalized
	  beta[lp+j] = beta_old[j] + v1/v2; 
        } else {
          beta[lp+j] = beta_old[j] + (v1-lambda[l]*pf[j]*beta_old[j])/(v2+lambda[l]*pf[j]); 
        }
	// Update r, d1, d2 and compute candidate of max_update
        change = beta[lp+j]-beta_old[j];
        if (fabs(change) > 1e-6) {
	  //v2 = 0.0;
          for (i=0; i<n; i++) {
	    r[i] -= x[jn+i]*change;
            if (fabs(r[i])>gamma) {
              d1[i] = sign(r[i]);
              d2[i] = 0.0;
            } else {
	      d1[i] = r[i]*gi;
              d2[i] = gi;
	      //v2 += x2[jn+i]*d2[i];
	    }
	  }
	  //v2 += n*lambda[l]*pf[j];
	  update = (v2+lambda[l]*pf[j])*change*change*n;
          if (update>max_update) max_update = update;
          beta_old[j] = beta[lp+j];
        }
      }
      // Check for convergence
      if (iter[l] > 1) {
        if (max_update < thresh) {
          converged = 1;
	  break;
	}
      }
    }
  }
  // Postprocessing
  if (ppflag) postprocess(beta, shift, scale, nlam, p);

  Free(x2);
  Free(shift);
  Free(scale);
  Free(beta_old);
  Free(r);
  Free(d1);
  Free(d2);
}

static void sncd_quantile_l2(double *beta, int *iter, double *lambda, double *x, double *y, double *pf, 
			     double *tau_, double *eps_, double *lambda_min_, int *nlam_, int *n_, int *p_, 
			     int *ppflag_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double tau = tau_[0]; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0];
  int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lp, jn, converged; 
  double gamma, gi, pct, lstep, ldiff, lmax, v1, v2, v3, temp, change, nullDev, max_update, update, thresh;
  double c = 2*tau-1.0; // coefficient for the linear term in quantile loss
  double *x2 = Calloc(n*p, double); // x^2
  for (i=0; i<n; i++) x2[i] = 1.0; // column of 1's for intercept
  double *shift = Calloc(p, double);
  double *scale = Calloc(p, double);
  double *beta_old = Calloc(p, double); 
  double *r = Calloc(n, double);
  double *d = Calloc(n, double);
  double *d1 = Calloc(n, double);
  double *d2 = Calloc(n, double);
  int m = (int) (n*0.10);

  // Preprocessing
  if (ppflag == 1) {
    standardize(x, x2, shift, scale, n, p);
  } else if (ppflag == 2) {
    rescale(x, x2, shift, scale, n, p);
  } else {
    for (j=1; j<p; j++) {
      jn = j*n;
      for (i=0; i<n; i++) x2[jn+i]=pow(x[jn+i],2);
    }
  }

  // Initialization
  nullDev = 0.0; // not divided by n
  for (i=0;i<n;i++) {
    r[i] = y[i];
    nullDev += fabs(r[i]) + c*r[i];
  }
  thresh = eps*nullDev;
  derivative_quantapprox(d1, d2, r, gamma, c, n);

  // Set up lambda
  if (user==0) {
    lambda[0] = maxprod(x, d1, n, p, pf);
    for (i=0; i<n; i++) {
      if (fabs(r[i]) < 1e-10) {
        d[i] = c;
      } else {
        d[i] = sign(r[i])+c;
      } 
    }
    temp = maxprod(x, d, n, p, pf);
    if (temp>lambda[0]) lambda[0] = temp;
    lambda[0] = lambda[0]/(2*n)*10;
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
  }

  // Solution path
  for (l=0; l<nlam; l++) {
    if (gamma>0.001 && l>0) {
      temp = ksav(r, n, m);
      if (temp < gamma) gamma = temp;
    }
    if (gamma<0.001) gamma = 0.001;
    gi = 1.0/gamma;
    if (message) Rprintf("Lambda %d: Gamma = %f\n", l+1, gamma);
    converged = 0; lp = l*p;
    while(iter[l] < max_iter) {
      iter[l]++;
      max_update = 0.0; 
      for (j=0; j<p; j++) {
        // Calculate v1, v2
	jn = j*n; v1 = 0.0; v2 = 0.0; pct = 0.0;
        for (i=0; i<n; i++) {
          v1 += x[jn+i]*d1[i];
          v2 += x2[jn+i]*d2[i];
          pct += d2[i];
        }
	pct = pct*gamma/n; // percentage of residuals with absolute values below gamma
	if (pct < 0.08 || pct < 1.0/n) {
	  // approximate v2 with a continuation technique
	  for (i=0; i<n; i++) {
	    temp = fabs(r[i]);
	    if (temp > gamma) v2 += x2[jn+i]/temp;
	  }
	}
	v1 = v1/(2.0*n); v2 = v2/(2.0*n);
        // Update beta_j
        if (pf[j]==0.0) { // unpenalized
	  beta[lp+j] = beta_old[j] + v1/v2; 
        } else {
          beta[lp+j] = beta_old[j] + (v1-lambda[l]*pf[j]*beta_old[j])/(v2+lambda[l]*pf[j]); 
        }
	// Update r, d1, d2 and compute candidate of max_update
        change = beta[lp+j]-beta_old[j];
        if (fabs(change) > 1e-6) {
	  //v2 = 0.0;
          for (i=0; i<n; i++) {
	    r[i] -= x[jn+i]*change;
            if (fabs(r[i])>gamma) {
              d1[i] = sign(r[i])+c;
              d2[i] = 0.0;
            } else {
	      d1[i] = r[i]*gi+c;
              d2[i] = gi;
	    }
	  }
          update = (v2+lambda[l]*pf[j])*change*change*n;
          if (update>max_update) max_update = update;
          beta_old[j] = beta[lp+j];
        }
      }
      // Check for convergence
      if (iter[l] > 5) {
        if (max_update < thresh) {
          converged = 1;
	  break;
	}
      }
    }
  }
  // Postprocessing
  if (ppflag) postprocess(beta, shift, scale, nlam, p);
  
  Free(x2);
  Free(shift);
  Free(scale);
  Free(beta_old);
  Free(r);
  Free(d);
  Free(d1);
  Free(d2);
}

static void sncd_squared_l2(double *beta, int *iter, double *lambda, double *x, double *y, double *pf, 
			    double *eps_, double *lambda_min_, int *nlam_, int *n_, int *p_, int *ppflag_, 
                            int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0];
  int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lp, jn, converged; 
  double pct, lstep, ldiff, lmax, v1, v2, v3, temp, change, nullDev, max_update, update, thresh;
  double *x2 = Calloc(n*p, double); // x^2
  for (i=0; i<n; i++) x2[i] = 1.0; // column of 1's for intercept
  double *x2bar = Calloc(p, double); // Col Mean of x2
  x2bar[0] = 1.0; 
  double *shift = Calloc(p, double);
  double *scale = Calloc(p, double);
  double *beta_old = Calloc(p, double); 
  double *r = Calloc(n, double);
  
  // Preprocessing
  if (ppflag == 1) {
    standardize(x, x2, shift, scale, n, p);
  } else if (ppflag == 2) {
    rescale(x, x2, shift, scale, n, p);
  } else {
    for (j=1; j<p; j++) {
      jn = j*n;
      for (i=0; i<n; i++) x2[jn+i]=pow(x[jn+i],2);
    }
  }

  // Initialization
  nullDev = 0.0;
  for (i=0; i<n; i++) {
    r[i] = y[i];
    nullDev += pow(r[i],2); // without dividing by 2n
  }
  thresh = eps*nullDev;
  for (j=0; j<p; j++) {
    jn = j*n; temp = 0.0;
    for (i=0; i<n; i++) temp += x2[jn+i];
    x2bar[j] = temp/n;
  }

  // Set up lambda
  if (user==0) {
    lambda[0] = maxprod(x, r, n, p, pf)/n*10;
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
  }

  // Solution path
  for (l=0; l<nlam; l++) {
    converged = 0; lp = l*p;
    while(iter[l] < max_iter) {
      iter[l]++;
      max_update = 0.0; 
      for (j=0; j<p; j++) {
        if (j == 0 && ppflag == 1) continue; // intercept is constant for standardized data
        // Update v1, v2=x2bar[j]
      	v1 = crossprod(x, r, n, j)/n; v2 = x2bar[j];
        // Update beta_j
        if (pf[j]==0.0) { // unpenalized
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
	  if (update>max_update) max_update = update;
	  beta_old[j] = beta[lp+j];
        }
      }
      // Check for convergence
      if (iter[l] > 1) {
        if (max_update < thresh) {
          converged = 1;
	  break;
	}
      }
    }
  }
  // Postprocessing
  if (ppflag) postprocess(beta, shift, scale, nlam, p);

  Free(x2);
  Free(x2bar);
  Free(shift);
  Free(scale);
  Free(beta_old);
  Free(r);
}


static const R_CMethodDef cMethods[] = {
  {"huber", (DL_FUNC) &sncd_huber, 21},
  {"quant", (DL_FUNC) &sncd_quantile, 21},
  {"squared", (DL_FUNC) &sncd_squared, 20},
  {"huber_l2", (DL_FUNC) &sncd_huber_l2, 16},
  {"quantile_l2", (DL_FUNC) &sncd_quantile_l2, 16},
  {"squared_l2", (DL_FUNC) &sncd_squared_l2, 15},
  {NULL}
};

void R_init_hqreg(DllInfo *info)
{
  R_registerRoutines(info,cMethods,NULL,NULL,NULL);
}
