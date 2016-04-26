#include <math.h>
#include <time.h>
#include <string.h>
#include <R.h>
#include <R_ext/Applic.h>
#include "Rinternals.h"
#include "R_ext/Rdynload.h"

double ksav(double *a, int size, int K);

static double sign(double x) {
  if (x > 0.0) return 1.0;
  else if (x < 0.0) return -1.0;
  else return 0.0;
}

static void derivative_huber(double *d1, double *d2, double *r, double gamma, int n) {
  for (int i=0; i<n; i++)
    if (fabs(r[i]) > gamma) {
      d1[i] = sign(r[i]);
      d2[i] = 0.0;
    } else {
      d1[i] = r[i]/gamma;
      d2[i] = 1.0/gamma;
    }
}

static void derivative_quantapprox(double *d1, double *d2, double *r, double gamma, double c, int n) {
  for (int i=0; i<n; i++) {
    if (fabs(r[i]) > gamma) {
      d1[i] = sign(r[i]);
      d2[i] = 0.0;
    } else {
      d1[i] = r[i]/gamma;
      d2[i] = 1.0/gamma;
    }
    d1[i] += c;
  }
}

static double crossprod(double *x, double *v, int n, int j) {
  int jn = j*n, i;
  double sum=0.0;
  for (i=0;i<n;i++) sum += x[jn+i]*v[i];
  return(sum);
}

static double maxprod(double *x, double *v, int n, int p, double *pf) {
  int j;
  double z, max=0.0;
  for (j=1; j<p; j++) {
    if (pf[j]) {
      z = fabs(crossprod(x, v, n, j))/pf[j];
      if (z>max) max = z;
    }
  }
  return(max);
}

// standardization for feature matrix
static void standardize(double *x, double *x2, double *shift, double *scale, int n, int p) 
{
  int i, j, jn; double xm, xsd, xvar;
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
    for (i=0; i<n; i++) {
      x[jn+i] = x[jn+i]/xsd;
      x2[jn+i] = x2[jn+i]/xvar;
    }
    shift[j] = xm;
    scale[j] = xsd;
  }
} 

// rescaling for feature matrix
static void rescale(double *x, double *x2, double *shift, double *scale, int n, int p) 
{
  int i, j, jn; double cmin, cmax, crange;
  for (j=1; j<p; j++) {
    jn = j*n; cmin = x[jn]; cmax = x[jn];
    for (i=1; i<n; i++) {
      if (x[jn+i] < cmin) {
        cmin = x[jn+i];
      } else if (x[jn+i] > cmax) {
        cmax = x[jn+i];
      }
    }
    crange = cmax - cmin;
    for (i=0; i<n; i++) {
      x[jn+i] = (x[jn+i]-cmin)/crange;
      x2[jn+i] = pow(x[jn+i], 2);
    }
    shift[j] = cmin;
    scale[j] = crange;
  }
}

// postprocess feature coefficients
static void postprocess(double *beta, double *shift, double *scale, int nlam, int p) {
  int l, j, lp; double prod;
  for (l = 0; l<nlam; l++) {
    lp = l*p;
    prod = 0.0;
    for (j = 1; j<p; j++) {
      beta[lp+j] = beta[lp+j]/scale[j];
      prod += shift[j]*beta[lp+j];
    }
    beta[lp] -= prod;
  }
}

// Semismooth Newton Coordinate Descent (SNCD)
static void sncd_huber(double *beta, int *iter, double *lambda, int *saturated, int *numv, double *x, double *y, double *d, double *pf, double *gamma_, double *alpha_, double *eps_, double *lambda_min_, 
	int *nlam_, int *n_, int *p_, int *ppflag_, int *scrflag_, int *dfmax_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double gamma = gamma_[0]; double alpha = alpha_[0]; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0]; int scrflag = scrflag_[0];
  int dfmax = dfmax_[0]; int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lp, jn, converged, mismatch; double pct, lstep, ldiff, lmax, l1, l2, v1, v2, v3, temp, change, nullDev, max_update, update, thresh, strfactor = 1.0; 
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
  include[0] = 1; // intercept is unpenalized so always included
  //scrflag = 0: no screening; scrflag = 1: Adaptive Strong Rule(ASR); scrflag = 2: Strong Rule(SR)
  // ASR fits an appropriate strfactor adaptively; SR always uses strfactor = 1
  if (scrflag == 0) {
    for (j=1; j<p; j++) include[j] = 1;
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
 
  for (j=0; j<p; j++) {
    z[j] = crossprod(x, d1, n, j)/n;
  }

  // Setup lambda
  if (user==0) {
    lambda[0] = maxprod(x, d, n, p, pf)/(n*alpha);
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
  }

  // Solution path
  for (l=0; l<nlam; l++) {
    converged = 0; lp = l*p;
    l1 = lambda[l]*alpha;
    l2 = lambda[l]*(1.0-alpha);
    // Variable screening
    if (scrflag != 0) {
      if (strfactor>5.0) strfactor = 5.0;
      if (l!=0) {
        cutoff = alpha*((1.0+strfactor)*lambda[l] - strfactor*lambda[l-1]);
        ldiff = lambda[l-1] - lambda[l];
      } else {
        lmax = 0.0;
        for (j=0; j<p; j++) if (fabs(z[j])>lmax) lmax = fabs(z[j]);
        lmax = lmax/alpha;
        cutoff = alpha*((1+strfactor)*lambda[0] - strfactor*lmax);
        ldiff = lmax - lambda[0];
      }
      for (j=1; j<p; j++) {
        if(fabs(z[j]) > (cutoff * pf[j])) {
          include[j] = 1;
        } else {
          include[j] = 0;
        }
      }
      strfactor = 1.0; //reset strfactor for ASR
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
	    jn = j*n; v1 = 0.0; v2 = 0.0; //pct = 0.0;
            for (i=0; i<n; i++) {
              v1 += x[jn+i]*d1[i];
              v2 += x2[jn+i]*d2[i];
              //pct += d2[i];
            }
	    v1 = v1/n; v2 = v2/n; //pct = pct*gamma/n;
	    if (v2 < 0.01 && v2 < 1.0/n) {
	    //if (pct < 0.05 || pct < 1.0/n) {
	      // approximate v2 with a continuation technique
              v2 = 0.0; 
	      for (i=0; i<n; i++) {
		if (d2[i]) {
		  v2 += x2[jn+i]*d2[i];
		} else { // |r_i|>gamma
                  v2 += x2[jn+i]*d1[i]/r[i];
                }
              }
              v2 = v2/n;
              // Rprintf("After: v2=%f\n", v2);              
	    }
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
	    //if (!mismatch && j>0) {
              //if (fabs(s[j]) > 1 || (beta[lp+j] != 0 && s[j] != sign(beta[lp+j])))
		 //mismatch = 1;
            //}
	    // Update r, d1, d2 and compute candidate of max_update
            change = beta[lp+j]-beta_old[j];
            if (change!=0.0) {
	      v2 = 0.0;
              for (i=0; i<n; i++) {
		r[i] -= x[jn+i]*change;
                if (fabs(r[i])>gamma) {
                  d1[i] = sign(r[i]);
                  d2[i] = 0.0;
                } else {
		  d1[i] = r[i]/gamma;
		  d2[i] = 1.0/gamma;
	          v2 += x2[jn+i]*d2[i];
	        }
	      }
	      v2 += n*l2*pf[j];
	      update = v2*change*change;
              if (update>max_update) max_update = update;
              beta_old[j] = beta[lp+j];
            }
          }
        }
        // Check for convergence
        if (iter[l]>1) {
          if (max_update < thresh) {
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
	    } else if (scrflag == 1 && ldiff != 0.0) {
	      v3 = fabs((v1-z[j])/(pf[j]*ldiff*alpha));
              if (v3 > strfactor) strfactor = v3;
	    }
	    z[j] = v1;
	  }
          if (beta_old[j] != 0.0) nnzero++;
        }
        if (violations>0 && message) Rprintf("\n");
      } else {
        for (j=0; j<p; j++) {
          if (beta_old[j] != 0.0) nnzero++;
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

static void sncd_quantile(double *beta, int *iter, double *lambda, int *saturated, int *numv, double *x, double *y, double *d, double *pf, double *gamma_, double *tau_, double *alpha_, double *eps_, 
       double *lambda_min_, int *nlam_, int *n_, int *p_, int *ppflag_, int *scrflag_, int *dfmax_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double gamma = gamma_[0]; double tau = tau_[0]; double c = 2*tau-1.0; double alpha = alpha_[0]; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0]; int scrflag = scrflag_[0];
  int dfmax = dfmax_[0]; int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lp, jn, converged, mismatch; double pct, lstep, ldiff, lmax, l1, l2, v1, v2, v3, temp, change, nullDev, max_update, update, thresh, strfactor = 1.0; 
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
  include[0] = 1; // intercept is unpenalized so always included
  //scrflag = 0: no screening; scrflag = 1: Adaptive Strong Rule(ASR); scrflag = 2: Strong Rule(SR)
  // ASR fits an appropriate strfactor adaptively; SR always uses strfactor = 1
  if (scrflag == 0) {
    for (j=1; j<p; j++) include[j] = 1;
  }
  int violations = 0, nv = 0;
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
    temp = r[i];
    nullDev += fabs(temp) + c*temp;
  }
  thresh = eps*nullDev;
  derivative_quantapprox(d1, d2, r, gamma, c, n);
  for (j=0; j<p; j++) z[j] = crossprod(x, d1, n, j)/(2*n);

  // Setup lambda
  if (user==0) {
    lambda[0] = maxprod(x, d, n, p, pf);
    // compute lambda[0] for original quantile loss
    for (i=0; i<n; i++) {
      if (fabs(r[i]) < 1e-10) {
        d[i] = 1.0+c;
      } else {
        d[i] = sign(r[i])+c;
      } 
    }
    temp = maxprod(x, d, n, p, pf);
    if (temp>lambda[0]) lambda[0] = temp; // pick the larger one
    lambda[0] = lambda[0]/(2*n*alpha);
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
    if (message) Rprintf("Lambda %d: Gamma = %f\n", l+1, gamma);
    converged = 0; lp = l*p;
    l1 = lambda[l]*alpha;
    l2 = lambda[l]*(1.0-alpha);
    // Variable screening
    if (scrflag != 0) {
      if (strfactor>5.0) strfactor = 5.0;
      if (l!=0) {
        cutoff = alpha*((1.0+strfactor)*lambda[l] - strfactor*lambda[l-1]);
        ldiff = lambda[l-1] - lambda[l];
      } else {
        lmax = 0.0;
        for (j=0; j<p; j++) if (fabs(z[j])>lmax) lmax = fabs(z[j]);
        lmax = lmax/alpha;
        cutoff = alpha*((1+strfactor)*lambda[0] - strfactor*lmax);
        ldiff = lmax - lambda[0];
      }
      for (j=1; j<p; j++) {
        if (include[j] == 0 && fabs(z[j]) > (cutoff * pf[j])) include[j] = 1;
      }
      strfactor = 1.0; //reset strfactor for ASR
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
            // Calculate v1, v2
	    jn = j*n; v1 = 0.0; v2 = 0.0; pct = 0.0;
            for (i=0; i<n; i++) {
              v1 += x[jn+i]*d1[i];
              v2 += x2[jn+i]*d2[i];
              pct += d2[i];
            }
	    v1 = v1/(2*n); v2 = v2/(2*n); pct = pct*gamma/n;
	    if (pct < 0.05 || pct < 1.0/n) {
	      // Rprintf("j=%d, pct=%lf\n",j,pct);
	      // approximate v2 with a continuation technique
              v2 = 0.0;
	      for (i=0; i<n; i++) {
                if (d2[i]) {
                  v2 += x2[jn+i]*d2[i];
                } else { // |r_i| > gamma
                  v2 += x2[jn+i]*(d1[i]-c)/r[i];
                }
              }
              v2 = v2/(2*n);
              //Rprintf("After: v2=%f\n", v2);
	    }
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
	    if (!mismatch && j>0) {
              if (fabs(s[j]) > 1 || (beta[lp+j] != 0 && s[j] != sign(beta[lp+j])))
		 mismatch = 1;
            }
	    // Update r, d1, d2 and compute candidate of max_update
            change = beta[lp+j]-beta_old[j];
            if (change!=0) {
	      v2 = 0.0;
              for (i=0; i<n; i++) {
		r[i] -= x[jn+i]*change;
                if (fabs(r[i])>gamma) {
                  d1[i] = sign(r[i])+c;
                  d2[i] = 0.0;
                } else {
		  d1[i] = r[i]/gamma+c;
		  d2[i] = 1.0/gamma;
	          //v2 += x2[jn+i]*d2[i];
	        }
	      }
	      //v2 += 2*n*l2*pf[j];
	      //update = v2*change*change;
              //Rprintf("loss decrease = %f, penalty decrease = %f\n", v2*change*change, 2*n*l1*pf[j]*fabs(fabs(beta[lp+j])-fabs(beta_old[j])));
              update = n*(v2*change*change + 2*fabs(v1*change));
              if (update>max_update) max_update = update;
              beta_old[j] = beta[lp+j];
            }
          }
        }
        // Check for convergence
        if (iter[l]>10) {
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
	    } else if (scrflag == 1 && ldiff != 0.0) {
	      v3 = fabs((v1-z[j])/(pf[j]*ldiff*alpha));
              if (v3 > strfactor) strfactor = v3;
	    }
	    z[j] = v1;
	  }
          if (beta_old[j] != 0.0) nnzero++;
        }
        if (violations>0 && message) Rprintf("\n");
      } else {
        for (j=0; j<p; j++) {
          if (beta_old[j] != 0.0) nnzero++;
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
  Free(d1);
  Free(d2);
  Free(z);
  Free(include);
}

static void sncd_squared(double *beta, int *iter, double *lambda, int *saturated, int *numv, double *x, double *y, double *d, double *pf, double *alpha_, double *eps_, double *lambda_min_, 
	int *nlam_, int *n_, int *p_, int *ppflag_, int *scrflag_, int *dfmax_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double alpha = alpha_[0]; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0]; int scrflag = scrflag_[0];
  int dfmax = dfmax_[0]; int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lp, jn, converged, mismatch; double lstep, ldiff, lmax, l1, l2, v1, v2, v3, temp, change, nullDev, max_update, update, thresh, strfactor = 1.0;
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
  include[0] = 1; // intercept is unpenalized so always included
  //scrflag = 0: no screening; scrflag = 1: Adaptive Strong Rule(ASR); scrflag = 2: Strong Rule(SR)
  // ASR fits an appropriate strfactor adaptively; SR always uses strfactor = 1
  if (scrflag == 0) {
    for (j=1; j<p; j++) include[j] = 1;
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
    z[j] = crossprod(x, r, n, j)/n;
    jn = j*n;
    for (i=0; i<n; i++) x2bar[j] += x2[jn+i];
    x2bar[j] = x2bar[j]/n;
  }
  
  // Setup lambda
  if (user==0) {
    lambda[0] = maxprod(x, d, n, p, pf)/(n*alpha);
    if (lambda_min == 0.0) lambda_min = 0.001;
    lstep = log(lambda_min)/(nlam - 1);
    for (l=1; l<nlam; l++) lambda[l] = lambda[l-1]*exp(lstep);
  }

  // Solution path
  for (l=0; l<nlam; l++) {
    converged = 0; lp = l*p;
    l1 = lambda[l]*alpha;
    l2 = lambda[l]*(1.0-alpha);

    // Variable screening
    if (scrflag != 0) {
      if (strfactor>5.0) strfactor = 5.0;
      if (l!=0) {
        cutoff = alpha*((1.0+strfactor)*lambda[l] - strfactor*lambda[l-1]);
        ldiff = lambda[l-1] - lambda[l];
      } else {
        lmax = 0.0;
        for (j=0; j<p; j++) if (fabs(z[j])>lmax) lmax = fabs(z[j]);
        lmax = lmax/alpha;
        cutoff = alpha*((1+strfactor)*lambda[0] - strfactor*lmax);
        ldiff = lmax - lambda[0];
      }
      for (j=1; j<p; j++) {
        if (include[j] == 0 && fabs(z[j]) > (cutoff * pf[j])) include[j] = 1;
      }
      strfactor = 1.0; //reset strfactor for ASR
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
	    if (!mismatch && j>0) {
              if (fabs(s[j]) > 1 || (beta[lp+j] != 0 && s[j] != sign(beta[lp+j])))
		 mismatch = 1;
            }
	    // Update residuals
            change = beta[lp+j]-beta_old[j];       
            if (change!=0.0) {
	      jn = j*n;              
              for (i=0; i<n; i++) r[i] -= x[jn+i]*change;
	      update = n*(1.0+l2*pf[j])*change*change;
	      if (update>max_update) max_update = update;
	      beta_old[j] = beta[lp+j];
            }
          }
        }             
        // Check for convergence
        if (iter[l]>1) {
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
	    } else if (scrflag == 1 && ldiff != 0.0) {
	      v3 = fabs((v1-z[j])/(pf[j]*ldiff*alpha));
              if (v3 > strfactor) strfactor = v3;
	    }
	    z[j] = v1;
	  }
          if (beta_old[j] != 0.0) nnzero++;
        }
        if (violations>0 && message) Rprintf("\n");
      } else {
        for (j=0; j<p; j++) {
          if (beta_old[j] != 0.0) nnzero++;
        }
      }
      if (violations==0) break;
      nv += violations;
    }
  }
  if (scrflag!=0 && message) Rprintf("# KKT violations detected and fixed: %d\n", nv);
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
static void sncd_huber_l2(double *beta, int *iter, double *lambda, double *x, double *y, double *d, double *pf, double *gamma_, double *eps_, double *lambda_min_, 
	int *nlam_, int *n_, int *p_, int *ppflag_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double gamma = gamma_[0]; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0];
  int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lp, jn, converged; double pct, lstep, ldiff, lmax, v1, v2, v3, temp, change, nullDev, max_update, update, thresh;
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

  // Setup lambda
  if (user==0) {
    lambda[0] = maxprod(x, d, n, p, pf)/(n*0.01);
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
	v1 = v1/n; v2 = v2/n; pct = pct*gamma/n;
	if (pct < 0.05 || pct < 1.0/n) {
          //Rprintf("j=%d, pct=%lf\n",j,pct);
	  // approximate v2 with a continuation technique
          v2 = 0.0; 
	  for (i=0; i<n; i++) {
	    if (d2[i]) {
	      v2 += x2[jn+i]*d2[i];
	    } else { // |r_i|>gamma
              v2 += x2[jn+i]*d1[i]/r[i];
            }
          }
          v2 = v2/n;
          // Rprintf("After: v2=%f\n", v2);              
	}
        // Update beta_j
        if (pf[j]==0.0) { // unpenalized
	  beta[lp+j] = beta_old[j] + v1/v2; 
        } else {
          beta[lp+j] = beta_old[j] + (v1-lambda[l]*pf[j]*beta_old[j])/(v2+lambda[l]*pf[j]); 
        }
	// Update r, d1, d2 and compute candidate of max_update
        change = beta[lp+j]-beta_old[j];
        if (change!=0.0) {
	  v2 = 0.0;
          for (i=0; i<n; i++) {
	    r[i] -= x[jn+i]*change;
            if (fabs(r[i])>gamma) {
              d1[i] = sign(r[i]);
              d2[i] = 0.0;
            } else {
	      d1[i] = r[i]/gamma;
              d2[i] = 1.0/gamma;
	      v2 += x2[jn+i]*d2[i];
	    }
	  }
	  v2 += n*lambda[l]*pf[j];
	  update = v2*change*change;
          if (update>max_update) max_update = update;
          beta_old[j] = beta[lp+j];
        }
      }
      // Check for convergence
      if (iter[l]>1) {
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

static void sncd_quantile_l2(double *beta, int *iter, double *lambda, double *x, double *y, double *d, double *pf, double *gamma_, double *tau_, double *eps_, 
       double *lambda_min_, int *nlam_, int *n_, int *p_, int *ppflag_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double gamma = gamma_[0]; double tau = tau_[0]; double c = 2*tau-1.0; double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0];
  int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lp, jn, converged; double pct, lstep, ldiff, lmax, v1, v2, v3, temp, change, nullDev, max_update, update, thresh;
  double *x2 = Calloc(n*p, double); // x^2
  for (i=0; i<n; i++) x2[i] = 1.0; // column of 1's for intercept
  double *shift = Calloc(p, double);
  double *scale = Calloc(p, double);
  double *beta_old = Calloc(p, double); 
  double *r = Calloc(n, double);
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
    temp = r[i];
    nullDev += fabs(temp) + c*temp;
  }
  thresh = eps*nullDev;
  derivative_quantapprox(d1, d2, r, gamma, c, n);

  // Setup lambda
  if (user==0) {
    lambda[0] = maxprod(x, d, n, p, pf);
    // compute lambda[0] for original quantile loss
    for (i=0; i<n; i++) {
      if (fabs(r[i]) < 1e-10) {
        d[i] = 1.0+c;
      } else {
        d[i] = sign(r[i])+c;
      } 
    }
    temp = maxprod(x, d, n, p, pf);
    if (temp>lambda[0]) lambda[0] = temp; // pick the larger one
    lambda[0] = lambda[0]/(2*n*0.01);
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
	v1 = v1/(2*n); v2 = v2/(2*n); pct = pct*gamma/n;
	if (pct < 0.05 || pct < 1.0/n) {
	  // Rprintf("j=%d, pct=%lf\n",j,pct);
	  // approximate v2 with a continuation technique
          v2 = 0.0;
	  for (i=0; i<n; i++) {
            if (d2[i]) {
              v2 += x2[jn+i]*d2[i];
            } else { // |r_i| > gamma
              v2 += x2[jn+i]*(d1[i]-c)/r[i];
            }
          }
          v2 = v2/(2*n);
          //Rprintf("After: v2=%f\n", v2);
	}
        // Update beta_j
        if (pf[j]==0.0) { // unpenalized
	  beta[lp+j] = beta_old[j] + v1/v2; 
        } else {
          beta[lp+j] = beta_old[j] + (v1-lambda[l]*pf[j]*beta_old[j])/(v2+lambda[l]*pf[j]); 
        }
	// Update r, d1, d2 and compute candidate of max_update
        change = beta[lp+j]-beta_old[j];
        if (change!=0) {
	  v2 = 0.0;
          for (i=0; i<n; i++) {
	    r[i] -= x[jn+i]*change;
            if (fabs(r[i])>gamma) {
              d1[i] = sign(r[i])+c;
              d2[i] = 0.0;
            } else {
	      d1[i] = r[i]/gamma+c;
              d2[i] = 1.0/gamma;
	    }
	  }
          update = n*(v2*change*change + 2*fabs(v1*change));
          if (update>max_update) max_update = update;
          beta_old[j] = beta[lp+j];
        }
      }
      // Check for convergence
      if (iter[l]>10) {
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

static void sncd_squared_l2(double *beta, int *iter, double *lambda, double *x, double *y, double *d, double *pf, double *eps_, double *lambda_min_, 
	int *nlam_, int *n_, int *p_, int *ppflag_, int *max_iter_, int *user_, int *message_)
{
  // Declarations
  double eps = eps_[0]; double lambda_min = lambda_min_[0]; 
  int nlam = nlam_[0]; int n = n_[0]; int p = p_[0]; int ppflag = ppflag_[0];
  int max_iter = max_iter_[0]; int user = user_[0]; int message = message_[0];
  int i, j, k, l, lp, jn, converged; double pct, lstep, ldiff, lmax, v1, v2, v3, temp, change, nullDev, max_update, update, thresh;
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
    jn = j*n;
    for (i=0; i<n; i++) x2bar[j] += x2[jn+i];
    x2bar[j] = x2bar[j]/n;
  }

  // Setup lambda
  if (user==0) {
    lambda[0] = maxprod(x, d, n, p, pf)/(n*0.01);
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
        if (change!=0.0) {
	  jn = j*n;              
          for (i=0; i<n; i++) r[i] -= x[jn+i]*change;
	  update = n*(1.0+lambda[l]*pf[j])*change*change;
	  if (update>max_update) max_update = update;
	  beta_old[j] = beta[lp+j];
        }
      }
      // Check for convergence
      if (iter[l]>1) {
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
  {"huber", (DL_FUNC) &sncd_huber, 22},
  {"quant", (DL_FUNC) &sncd_quantile, 23},
  {"squared", (DL_FUNC) &sncd_squared, 21},
  {"huber_l2", (DL_FUNC) &sncd_huber_l2, 17},
  {"quantile_l2", (DL_FUNC) &sncd_quantile_l2, 18},
  {"squared_l2", (DL_FUNC) &sncd_squared_l2, 16},
  {NULL}
};

void R_init_hqreg(DllInfo *info)
{
  R_registerRoutines(info,cMethods,NULL,NULL,NULL);
}

