
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <nlopt.h> 
#include "SweeD_BFGS.h"

// =========================================================================
// ORIGINAL NUMERICAL GRADIENT FUNCTIONS (Required by SweeD_SFS.c)
// =========================================================================
/* * Yanggradient: Calculates the numerical derivative (gradient) of the function.
 * Since we don't always have a perfect mathematical formula for the slope,
 * this function estimates it by taking tiny steps (eh) forward and backward 
 * (Finite Difference Method).
 */

void Yanggradient (int n, const double x[], const int need[], double f0, 
		   double g[], double (*fun)(const double x[]), 
		   double space[], int central,
		   const double* lowbound, const double* upbound)
{
  int i,j;
  double *x0=space, *x1=space+n, eh0, eh01, eh;
  eh0=eh01=1.e-8;

  if (central) {
    // Central difference: slope is calculated by looking slightly ahead AND slightly behind
    for (i=0;i<n;i++)  {
      if (need[i]) {    
	for (j=0;j<n;j++)  x0[j]=x1[j]=x[j];
	eh=pow(eh01*(fabs(x[i])+1), 0.67);
	x0[i]-=eh; x1[i]+=eh;
	if (x0[i]<lowbound[i])
	  {x1[i]+=eh; g[i] = ((*fun)(x1)-f0)/(eh*2.0);}
	else if (x1[i]>upbound[i])
	  {x0[i]-=eh; g[i] = (f0-(*fun)(x0))/(eh*2.0);}
	else
	  g[i] = ((*fun)(x1) - (*fun)(x0))/(eh*2.0);
      }
    }
  }
  else {
    for (i=0;i<n;i++)  {
      if (need[i]) {
	for (j=0;j<n;j++)  x1[j]=x[j];
	eh=2.0*pow(eh0*(fabs(x[i])+1), 0.67);
	if (x1[i]+eh>upbound[i])
	  {
	    x1[i]-=eh;
	    g[i] = (f0-(*fun)(x1))/eh;
	  }
	else
	  {
	    x1[i]+=eh;
	    g[i] = ((*fun)(x1)-f0)/eh;
	  }
      }
    }
  }
}

static double oldf0;
static int CENTRALMODE=1;

void getgradient(int npar, const double invec[], const int need_gradient[], 
		 double outvec[], double(*func)(const double []), 
		 const double* lowbound, const double* upbound)     
{
  double f0;
  int i;
  double *space = malloc((2*npar+10)*sizeof(double));

  CENTRALMODE=1;
  for (i=0; i<npar; i++)
    if (need_gradient[i]) break;
  if (i==npar) goto checkbounds;
  
  f0 = func(invec);
  
  if (CENTRALMODE > 0)
    Yanggradient(npar, invec, need_gradient, f0, outvec, func, space, 1, lowbound, upbound);
  else 
    Yanggradient(npar, invec, need_gradient, f0, outvec, func, space, 0, lowbound, upbound);
    
  if (CENTRALMODE == 0 && fabs(oldf0-f0) < 0.1)
    CENTRALMODE = 1;
  else if (CENTRALMODE > 0 && fabs(oldf0-f0) > 1.0)
    CENTRALMODE = 0;
  oldf0 = f0;

 checkbounds:
  for (i=0; i<npar; i++) {
    if (invec[i] <= lowbound[i] && outvec[i] > 0.0)
      outvec[i] = 0.0;
    if (invec[i] >=upbound[i] && outvec[i] < 0.0)
      outvec[i] = 0.0;
  }
  free(space);
}

// =========================================================================
//  NEW NLOPT IMPLEMENTATION 
// =========================================================================

typedef struct {
    double (*fun)(const double x[]);
    void (*dfun)(const double x[], double y[]);
    double *lowbound;
    double *upbound;
} SweeD_Funcs;

double nlopt_wrapper(unsigned n, const double *x, double *grad, void *my_func_data) {
    SweeD_Funcs *funcs = (SweeD_Funcs *)my_func_data;
    
    if (grad) {
        if (funcs->dfun != NULL) { 
            funcs->dfun(x, grad); 
        } else {
            unsigned i;
            int *need_gradient = malloc(n * sizeof(int));
            for (i = 0; i < n; i++) need_gradient[i] = 1;
            getgradient((int)n, x, need_gradient, grad, funcs->fun, funcs->lowbound, funcs->upbound);
            free(need_gradient);
        }
    }
    return funcs->fun(x); 
}

double findmax_bfgs(int numpars, double *invec, double (*fun)(const double x[]),
            void (*dfun)(const double x[], double y[]),
            double *lowbound, double *upbound, int *nbd, int noisy) 
{
    nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, numpars);

    if (lowbound != NULL) nlopt_set_lower_bounds(opt, lowbound);
    if (upbound != NULL) nlopt_set_upper_bounds(opt, upbound);

    SweeD_Funcs my_funcs = {fun, dfun, lowbound, upbound};
    nlopt_set_max_objective(opt, nlopt_wrapper, &my_funcs);

    nlopt_set_ftol_rel(opt, 1e-4); 

    double max_likelihood;
    
    nlopt_optimize(opt, invec, &max_likelihood);

    nlopt_destroy(opt);

    return -max_likelihood; 
}
