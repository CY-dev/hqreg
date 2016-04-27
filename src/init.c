#include <math.h>
#include <time.h>
#include <string.h>
#include <R.h>
#include <R_ext/Applic.h>
#include "Rinternals.h"
#include "R_ext/Rdynload.h"

double sign(double x);
double crossprod(double *x, double *v, int n, int j);

// Fit the initial solutions for unpenalized features in elastic-net penalized models
void init_huber(double *beta, double *beta_old, int *iter, double *x, double *x2, 
		double *y, double *r, double *pf, double *d1, double *d2, double gamma, 
		double thresh, int n, int p, int max_iter)
{
  double gi = 1.0/gamma, v1, v2, pct, change, max_update, update; int i, j, jn;
  while(iter[0]<max_iter) {
    iter[0]++;
    max_update = 0.0;
    for (j=0; j<p; j++) {
      if (pf[j] == 0.0) { // unpenalized
        // Calculate v1, v2
        jn = j*n; v1 = 0.0; v2 = 0.0; pct = 0.0;
        for (i=0; i<n; i++) {
          v1 += x[jn+i]*d1[i];
          v2 += x2[jn+i]*d2[i];
          pct += d2[i];
        }
        v1 = v1/n; v2 = v2/n; pct = pct*gamma/n;
        if (pct < 0.05 || pct < 1.0/n) {
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
        }
        // Update beta_j
        beta[j] = beta_old[j] + v1/v2; 
        // Update r, d1, d2 and compute candidate of max_update
        change = beta[j]-beta_old[j];
        if (fabs(change) > 1e-6) {
          for (i=0; i<n; i++) {
	    r[i] -= x[jn+i]*change;
            if (fabs(r[i])>gamma) {
              d1[i] = sign(r[i]);
              d2[i] = 0.0;
            } else {
              d1[i] = r[i]*gi;
              d2[i] = gi;
            }
          }
          update = n*v2*change*change;
          if (update>max_update) max_update = update;
          beta_old[j] = beta[j];
        }
      }
    }
    // Check for convergence
    if (max_update < thresh) break;
  }
}

void init_quantile(double *beta, double *beta_old, int *iter, double *x, double *x2, 
		   double *y, double *r, double *pf, double *d1, double *d2, double gamma, 
		   double c, double thresh, int n, int p, int max_iter)
{
  double gi = 1.0/gamma, v1, v2, pct, change, max_update, update; int i, j, jn, num_unpenalized = 0;
  // return when only intercept is unpenalized since intercept = quantile(y, tau) = 0
  for (j=1; j<p; j++) if(pf[j] == 0.0) num_unpenalized++;
  if (num_unpenalized == 0) return;
  while(iter[0]<max_iter) {
    iter[0]++;
    max_update = 0.0;
    for (j=0; j<p; j++) {
      if (pf[j] == 0.0) { // unpenalized
        // Calculate v1, v2
        jn = j*n; v1 = 0.0; v2 = 0.0; pct = 0.0;
        for (i=0; i<n; i++) {
          v1 += x[jn+i]*d1[i];
          v2 += x2[jn+i]*d2[i];
          pct += d2[i];
        }
        v1 = v1/(2*n); v2 = v2/(2*n); pct = pct*gamma/n;
        if (pct < 0.05 || pct < 1.0/n) {
	  // approximate v2 with a continuation technique
          v2 = 0.0;
          for (i=0; i<n; i++) {
            if (d2[i]) {
              v2 += x2[jn+i]*d2[i];
            } else { // |r_i|>gamma
              v2 += x2[jn+i]*(d1[i]-c)/r[i];
            }
          }
          v2 = v2/(2*n);
        }
        // Update beta_j
        beta[j] = beta_old[j] + v1/v2; 
        // Update r, d1, d2 and compute candidate of max_update
        change = beta[j]-beta_old[j];
        if (fabs(change) > 1e-6) {
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
          update = n*(v2*change*change + 2*fabs(v1*change));
          if (update>max_update) max_update = update;
          beta_old[j] = beta[j];
        }
      }
    }
    // Check for convergence
    if (max_update < thresh) break;
  }
}

void init_squared(double *beta, double *beta_old, int *iter, double *x, double x2bar, double *y, 
		  double *r, double *pf, double thresh, int n, int p, int ppflag, int max_iter)
{
  double gi = 1.0/gamma, v1, v2, change, max_update, update; int i, j, jn;
  while(iter[0]<max_iter) {
    iter[0]++;
    max_update = 0.0;
    for (j=0; j<p; j++) {
      if (j == 0 && ppflag == 1) continue; // intercept is constant for standardized data
      if (pf[j] == 0.0) { // unpenalized
        // Calculate v1, v2
      	v1 = crossprod(x, r, n, j)/n; v2 = x2bar[j];
        // Update beta_j
        beta[j] = beta_old[j] + v1/v2; 
        // Update r and compute candidate of max_update
        change = beta[j]-beta_old[j];
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
    if (max_update < thresh) break;
  }
}