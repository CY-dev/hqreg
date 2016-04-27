#include <math.h>
#include <time.h>
#include <string.h>
#include <R.h>
#include <R_ext/Applic.h>
#include "Rinternals.h"
#include "R_ext/Rdynload.h"

// Fit initial solutions for unpenalized coefficients in elastic-net penalized models
static void init_huber(double *beta, double *beta_old, double *x, double *x2, double *y, double *r, double *pf, double *d1, double *d2,
                       double gamma, double gi, double thresh, int max_iter, int n, int p) {
  double v1, v2, pct; int i, j, jn;
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
        // Rprintf("After: v2=%f\n", v2);              
	    }
      // Update beta_j
	    beta[j] = beta_old[j] + v1/v2; 
	    // Update r, d1, d2 and compute candidate of max_update
      change = beta[j]-beta_old[j];
      if (fabs(change) > 1e-6) {
        beta_old[j] = beta[j];
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
      }
    }
  }
}
