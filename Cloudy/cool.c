#include "../copyright.h"
/*============================================================================*/
/*! \file cool.c
 *  \brief Implements various optically thin cooling functions.  
 *
 *  These can be
 *  enrolled by setting CoolingFunc=NAME in the problem generator, where NAME
 *  is one of the functions in this file.
 *
 *  Each cooling function returns the cooling rate per volume.  The total 
 *  (or equivalently the internal) energy then evolves as
 *   -   dE/dt = de/dt = - CoolingFunc
 *
 *  Some of these cooling functions return the cooling rate per volume in
 *  cgs units [ergs/cm^{3}/s].  Thus, to use these functions, the entire
 *  calculation must be in cgs, or else the cooling rate has to scaled
 *  appropriately in the calling function. 
 *
 *  To add a new cooling function, implement it below and add the name to 
 *  src/microphysics/prototypes.h.  Note the argument list must be (d,P,dt).
 *
 * CONTAINS PUBLIC FUNCTIONS:
 * - KoyInut() - Koyama & Inutsuka cooling function */
/*============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include "../defs.h"
#include "../athena.h"
#include "../globals.h"
#include "prototypes.h"
#include "../prototypes.h"
#include "../table.h"

/* units for the SuthDop cooling function */
static const Real MIN_TEMPERATURE = 5.0e3;
static Real TempNorm;
static Real Ncool;
Real PressureFloor, COOL_SAFETY_FACTOR;

void SetCoolingFunction()
{
  /*Costanti di normalizzazione*/
  Real v_zero = par_getd("problem", "vel_units");
  Real rho_zero = par_getd("problem", "rho_units");
  Real T_zero = par_getd("problem", "temp_units");
  Real L_zero = par_getd("problem", "lenght_units");
  Real cool_units = par_getd("problem", "cool_units");
  Real min_T_fact = par_getd("problem", "min_T_fact");
  COOL_SAFETY_FACTOR = par_getd_def("problem", "cool_safety_fact", 10.0);
  Real Z1 = par_getd("problem","Z1");

  TempNorm = v_zero * v_zero * T_zero;
  Ncool = cool_units * L_zero * rho_zero / (v_zero * v_zero * v_zero);

  PressureFloor = min_T_fact * MIN_TEMPERATURE / (TempNorm); //(MolWeight(MIN_TEMPERATURE,Z1,) * TempNorm);
  
  CoolingFunc = CoolHeat;
}

double MolWeight(Real T, Real metallicity, Real dens)
{
  int i,j,k;
  Real logT, mw, log_rho, nh;
  Real interp1, interp2;
  
  logT = log10(T);

  if (logT>=6.95) logT=6.85;

  i = (int) ((logT - temper_zero) / T_delta);
  i = MAX(i, 0);
    
  j = (int) ((metallicity - met_zero) / Z_delta);
  j = MAX(j, 0);
    
  nh = ab_H[j] + (ab_H[j+1] - ab_H[j])/Z_delta * (metallicity - met_zero - j * Z_delta);
  nh *= dens * 1e-2 / 1.66;
    
  log_rho = log10(nh);
    
  k = (int) ((log_rho - rho_zero) / rho_delta);
  k = MAX(k, 0);

  interp1 = molw[i][j][k] + (molw[i+1][j][k]-molw[i][j][k])/T_delta * (logT - temper_zero - i * T_delta) + (molw[i][j+1][k] - molw[i][j][k] + (molw[i+1][j+1][k]-molw[i][j+1][k])/T_delta * (logT - temper_zero - i * T_delta) - (molw[i+1][j][k]-molw[i][j][k])/T_delta * (logT - temper_zero - i * T_delta))/Z_delta * (metallicity - met_zero - j * Z_delta);

  interp2 = molw[i][j][k+1] + (molw[i+1][j][k+1]-molw[i][j][k+1])/T_delta * (logT - temper_zero - i * T_delta) + (molw[i][j+1][k+1] - molw[i][j][k+1] + (molw[i+1][j+1][k+1]-molw[i][j+1][k+1])/T_delta * (logT - temper_zero - i * T_delta) - (molw[i+1][j][k+1]-molw[i][j][k+1])/T_delta * (logT - temper_zero - i * T_delta))/Z_delta * (metallicity - met_zero - j * Z_delta);

  if (logT < 4.0)
  {
    interp1 = molw[i][j][k] + (molw[i][j+1][k] - molw[i][j][k])/Z_delta * (metallicity - met_zero - j * Z_delta);
    interp2 = molw[i][j][k+1] + (molw[i][j+1][k+1] - molw[i][j][k+1])/Z_delta * (metallicity - met_zero - j * Z_delta);
  }

  mw = interp1 + (interp2 - interp1)/rho_delta * (log_rho - rho_zero - k * rho_delta);
  
  //if (logT < 4.0) printf("%e %e %e %e %e %e %e %d %d %d %e %e\n",logT,metallicity,dens,nh,mw, interp1,interp2,i,j,k,T_delta, (logT - temper_zero - i * T_delta)	
  return (mw * 1.66);
}


Real FindTemp(Real T_old, Real metallicity, Real dens)
{
  int iter, iter_max;
  int side = 0;
  Real temperature, T1, T2;
  Real f, f1, f2;
  Real t_tmp, corr, delta;
  

  T1 = T_old;
  T2 = T_old * MolWeight(T_old,metallicity,dens);

  f1 = T2 - T1;
  f2 = T_old * MolWeight(T2,metallicity,dens) - T2;
  
  // Faccio in modo che il prodotto di f1 per f2 sia negativo 
  // con f1 negativo e f2 positivo
  if ((f1 * f2) >= 0.)
  {
    if (f2 >= 0.)
    {
      do
      {
        T1 *= 1.1;
        f1 = T_old * MolWeight(T1,metallicity,dens) - T1;
      }
      while (f1 > 0.);
    }
    else
    {
      do
      {
        T2 /= 1.1;
        f2 = T_old * MolWeight(T2,metallicity,dens) - T2;
      }
      while (f2 < 0.);
    }
  }
  else 
  {
    if (f2 < 0.)
    {
      corr = T2, T2 = T1, T1 = corr;
      corr = f2, f2 = f1, f1 = corr;
    }
  }
  
  temperature = -1.0e30;
  corr = 1.0e30;
  t_tmp = T2;
  iter = 0;
  iter_max = 50;
  delta = 1.0e-6;
  
  do
  {
    temperature = T2 - f2 * (T2 - T1) / (f2 - f1);
    f = MolWeight(temperature,metallicity,dens) * T_old - temperature;
    corr = t_tmp - temperature;
    
    if(fabs(f) == 0.0) break;
    if (f < 0.)
    {
      T1 = temperature;
      f1 = f;
      if (side == -1) f2 *= 0.5;
      side = -1;
    }
    else 
    {
      T2 = temperature;
      f2 = f;
      if (side == +1) f1 *= 0.5;
      side = +1;
    }
    
    iter += 1;
  }
  while(fabs(corr/temperature) > delta && iter < iter_max);
  
  //printf("%e %e \n",temperature,T_old);
  return (temperature);
}

Real CoolHeat(const Real dens, const Real Press, const Real metal, const Real dt)
{
  int i,j,k;

//printf("%e \n", dens);
  Real lambda,H,lambda_net,T_old,metallicity,log_rho;
  Real T,logT,lambdamax,nh;
  Real interp1, interp2; 

  if(dens <= 0.0 || Press <= 0.0)
    ath_error("[SuthDop]: both density and pressure must be positive dens=%e,press=%e\n", dens, Press);
  
  T_old = Press / (dens);
  metallicity = metal / (dens) ;
  metallicity = log10(metallicity);
  
  T_old *= TempNorm;
  T = FindTemp (T_old,metallicity,dens);

  logT = log10(T);
  if (logT>=6.95) logT=6.85;
  log_rho = log10(dens);

  i = (int) ((logT - temper_zero) / T_delta);
  i = MAX(i, 0);
    
  j = (int) ((metallicity - met_zero) / Z_delta);
  j = MAX(j, 0);
    
  nh = ab_H[j] + (ab_H[j+1] - ab_H[j])/Z_delta * (metallicity - met_zero - j * Z_delta);
  nh *= dens * 1e-2 / 1.66;
    
  log_rho = log10(nh);
    
  k = (int) ((log_rho - rho_zero) / rho_delta);
  k = MAX(k, 0);
    
  interp1 = cool_net[i][j][k] + (cool_net[i+1][j][k]-cool_net[i][j][k])/T_delta * (logT - temper_zero - i * T_delta) + (cool_net[i][j+1][k] - cool_net[i][j][k] + (cool_net[i+1][j+1][k]-cool_net[i][j+1][k])/T_delta * (logT - temper_zero - i * T_delta) - (cool_net[i+1][j][k]-cool_net[i][j][k])/T_delta * (logT - temper_zero - i * T_delta))/Z_delta * (metallicity - met_zero - j * Z_delta);

  interp2 = cool_net[i][j][k+1] + (cool_net[i+1][j][k+1]-cool_net[i][j][k+1])/T_delta * (logT - temper_zero - i * T_delta) + (cool_net[i][j+1][k+1] - cool_net[i][j][k+1] + (cool_net[i+1][j+1][k+1]-cool_net[i][j+1][k+1])/T_delta * (logT - temper_zero - i * T_delta) - (cool_net[i+1][j][k+1]-cool_net[i][j][k+1])/T_delta * (logT - temper_zero - i * T_delta))/Z_delta * (metallicity - met_zero - j * Z_delta);

  lambda = interp1 + (interp2 - interp1)/rho_delta * (log_rho - rho_zero - k * rho_delta);

  interp1 = heat_net[i][j][k] + (heat_net[i+1][j][k]-heat_net[i][j][k])/T_delta * (logT - temper_zero - i * T_delta) + (heat_net[i][j+1][k] - heat_net[i][j][k] + (heat_net[i+1][j+1][k]-heat_net[i][j+1][k])/T_delta * (logT - temper_zero - i * T_delta) - (heat_net[i+1][j][k]-heat_net[i][j][k])/T_delta * (logT - temper_zero - i * T_delta))/Z_delta * (metallicity - met_zero - j * Z_delta);

  interp2 = heat_net[i][j][k+1] + (heat_net[i+1][j][k+1]-heat_net[i][j][k+1])/T_delta * (logT - temper_zero - i * T_delta) + (heat_net[i][j+1][k+1] - heat_net[i][j][k+1] + (heat_net[i+1][j+1][k+1]-heat_net[i][j+1][k+1])/T_delta * (logT - temper_zero - i * T_delta) - (heat_net[i+1][j][k+1]-heat_net[i][j][k+1])/T_delta * (logT - temper_zero - i * T_delta))/Z_delta * (metallicity - met_zero - j * Z_delta);
  
  H = interp1 + (interp2 - interp1)/rho_delta * (log_rho - rho_zero - k * rho_delta);	
  
  lambda_net = pow(10.,lambda) - pow(10.,H);
    
  if (logT < MIN_TEMPERATURE) 
  {
    lambda_net=0.0;
    return lambda_net;
  }
  lambdamax = dens * (T/MolWeight(T, metallicity, dens) - MIN_TEMPERATURE/MolWeight(MIN_TEMPERATURE, metallicity, dens)) / (TempNorm * Gamma_1 * dt);

  //lambda = MIN(Ncool*dens*dens*lambda, lambdamax);
  lambda_net = MIN(Ncool*dens*dens*lambda_net, lambdamax);

  return (lambda_net);
}




