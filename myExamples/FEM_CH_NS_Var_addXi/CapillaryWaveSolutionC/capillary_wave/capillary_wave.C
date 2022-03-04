//*******************************************************//
// Analytic solution for capillary wave 
// (Properetti, Phys. Fluids, 24, 1217, 1981)
//*******************************************************//

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "gsl/gsl_math.h"
#include "gsl/gsl_poly.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_complex.h"
#include "gsl/gsl_complex_math.h"
#include "dmath_util.h"
#include "dmath.h"


//***********************************************//
// Constants


#define OUTPUT_FILE "displace.dat"
#define PI 3.141592653589793238462643

#define DK (2.0*PI*1.0) // wave number

#define DENSITY_UPPER 1.0 // density of fluid on top, assumed to be lighter
#define DENSITY_LOWER 1.0 // density of fluid at bottom, assumed to be heavier

#define GRAVITATIONAL_ACCELERATION 1.0 // gravity

#define DENSITY_1 (DENSITY_UPPER) // rho_1, fluid on top
#define DENSITY_2 (DENSITY_LOWER) // rho_2, fluid at bottom

#define KINEMATIC_VISCOSITY 0.01 // kinematic viscosity, assume kinematic viscosities for
                                // for both fluids are the same
#define DYNAMIC_VISCOSITY_1 ((DENSITY_1)*(KINEMATIC_VISCOSITY)) // dynamic viscosity of fluid 1
#define DYNAMIC_VISCOSITY_2 ((DENSITY_2)*(KINEMATIC_VISCOSITY)) // dynamic viscosity of fluid 2


#define DH0 0.01 // initial amplitude of capillary wave
#define SURFACE_TENSION 1.0 // surface tension

#define T_INITIAL 0 // initial time
#define T_FINAL 0.10 // final time
#define N_STEPS 10000 // number of time steps

#define NROOTS 4


//************************************************//


int main(int argc, char **argv)
{
  double gravity = GRAVITATIONAL_ACCELERATION; // gravity
  double viscosity = KINEMATIC_VISCOSITY; // kinematic viscosity
  double dynamic_vis_1 = DYNAMIC_VISCOSITY_1,
	dynamic_vis_2 = DYNAMIC_VISCOSITY_2;
  double dk = DK;
  double sigma = SURFACE_TENSION; // surface tension
  double dh0 = DH0; // initial amplitude of capillary wave
  double rho_1 = DENSITY_1,
    rho_2 = DENSITY_2;

  double beta = rho_1*rho_2/((rho_1+rho_2)*(rho_1+rho_2));
  //double omega_0sq = sigma*dk*dk*dk/(rho_1+rho_2);
  double omega_0sq = sigma*dk*dk*dk/(rho_1+rho_2) + gravity*dk*(rho_2-rho_1)/(rho_1+rho_2);
  double omega_0 = sqrt(omega_0sq); 
  double nu_bar =  viscosity*dk*dk/omega_0;

  //----------------------------------------------//
  // solve for z_i, Z_i

  double A[(NROOTS)+1];
  double RootZ[2*(NROOTS)];

  double nu_omega = nu_bar*omega_0;
  A[0] = omega_0*omega_0 + (1.0-4.0*beta)*nu_omega*nu_omega; 
  A[1] = 4.0*(1.0-3.0*beta)*sqrt(nu_omega*nu_omega*nu_omega);
  A[2] = 2.0*(1.0-6.0*beta)*nu_omega;
  A[3] = -4.0*beta*sqrt(nu_omega);
  A[4] = 1.0;


  int NRoots = NROOTS;
  gsl_poly_complex_workspace *workspace = gsl_poly_complex_workspace_alloc(NRoots+1);
  
  if(gsl_poly_complex_solve(A, NRoots+1, workspace, RootZ) != GSL_SUCCESS) {
    std::cout << "ERROR: root finding failed!\n";
    gsl_poly_complex_workspace_free(workspace);
    return -1;
  }

  int i;

  gsl_complex c_z1 = gsl_complex_rect(RootZ[0], RootZ[1]),
    c_z2 = gsl_complex_rect(RootZ[2], RootZ[3]),
    c_z3 = gsl_complex_rect(RootZ[4],RootZ[5]),
    c_z4 = gsl_complex_rect(RootZ[6],RootZ[7]);

  gsl_complex c_Z1, c_Z2, c_Z3, c_Z4;

  c_Z1 = gsl_complex_sub(c_z2, c_z1);
  c_Z1 = gsl_complex_mul(c_Z1, gsl_complex_sub(c_z3, c_z1));
  c_Z1 = gsl_complex_mul(c_Z1, gsl_complex_sub(c_z4, c_z1));
  // c_Z1 = (z2-z1)*(z3-z1)*(z4-z1)
  
  c_Z2 = gsl_complex_sub(c_z3, c_z2);
  c_Z2 = gsl_complex_mul(c_Z2, gsl_complex_sub(c_z4, c_z2));
  c_Z2 = gsl_complex_mul(c_Z2, gsl_complex_sub(c_z1, c_z2));
  // Now c_Z2 = (z3-z2)*(z4-z2)*(z1-z2)

  c_Z3 = gsl_complex_sub(c_z4, c_z3);
  c_Z3 = gsl_complex_mul(c_Z3, gsl_complex_sub(c_z1, c_z3));
  c_Z3 = gsl_complex_mul(c_Z3, gsl_complex_sub(c_z2, c_z3));
  // Now c_Z3 = (z4-z3)*(z1-z3)*(z2-z3)

  c_Z4 = gsl_complex_sub(c_z1, c_z4);
  c_Z4 = gsl_complex_mul(c_Z4, gsl_complex_sub(c_z2, c_z4));
  c_Z4 = gsl_complex_mul(c_Z4, gsl_complex_sub(c_z3, c_z4));
  // Now c_Z4 = (z1-z4)*(z2-z4)*(z3-z4)

  //-----------------------------------------------//
  // compute analytic solution

  int Nsteps = N_STEPS;
  double T0 = T_INITIAL,
    Tf = T_FINAL;
  double step_t = (Tf-T0)/((double)(Nsteps));
  double temp_v, dtime;
  
  double *H = DMath::newD(Nsteps+1);
  gsl_complex temp_H, temp_R, temp_z, temp_erfc;
  double dtime_sqrt;
  
  double coeff_B_1 = 4.0*(1.0-4.0*beta)*nu_bar*nu_bar/(8.0*(1.0-4.0*beta)*nu_bar*nu_bar+1.0);

  gsl_complex c_exp_1, c_exp_2, c_exp_3, c_exp_4; // exponent

  c_exp_1 = gsl_complex_mul(c_z1, c_z1);
  c_exp_1 = gsl_complex_sub_real(c_exp_1, nu_omega); // z1^2 - nu_bar*omega_0

  c_exp_2 = gsl_complex_mul(c_z2, c_z2);
  c_exp_2 = gsl_complex_sub_real(c_exp_2, nu_omega); // z2^2 - nu_bar*omega_0

  c_exp_3 = gsl_complex_mul(c_z3, c_z3);
  c_exp_3 = gsl_complex_sub_real(c_exp_3, nu_omega); // z3^2 - nu_bar*omega_0

  c_exp_4 = gsl_complex_mul(c_z4, c_z4);
  c_exp_4 = gsl_complex_sub_real(c_exp_4, nu_omega); // z4^2 - nu_bar*omega_0

  gsl_complex c_coeff_1, c_coeff_2, c_coeff_3, c_coeff_4;

  c_coeff_1 = gsl_complex_div(c_z1, gsl_complex_mul(c_Z1, c_exp_1)); // z1/(Z1*(z1^2-nu_bar*omega_0))
  c_coeff_1 = gsl_complex_mul_real(c_coeff_1, omega_0sq); // z1*omega_0^2/(Z1*(z1^2-nu_bar*omega_0))

  c_coeff_2 = gsl_complex_div(c_z2, gsl_complex_mul(c_Z2, c_exp_2));
  c_coeff_2 = gsl_complex_mul_real(c_coeff_2, omega_0sq);

  c_coeff_3 = gsl_complex_div(c_z3, gsl_complex_mul(c_Z3, c_exp_3));
  c_coeff_3 = gsl_complex_mul_real(c_coeff_3, omega_0sq);

  c_coeff_4 = gsl_complex_div(c_z4, gsl_complex_mul(c_Z4, c_exp_4));
  c_coeff_4 = gsl_complex_mul_real(c_coeff_4, omega_0sq);

  int ierr;
  
  for(i=0;i<=Nsteps;i++) {

    dtime = T0 + i*step_t;
    dtime_sqrt = sqrt(dtime);

    temp_z = gsl_complex_mul_real(c_z1, dtime_sqrt); // z1*sqrt(t)
    dmath_cerfc(GSL_REAL(temp_z),GSL_IMAG(temp_z), &temp_erfc.dat[0], &temp_erfc.dat[1], &ierr);
    // Now temp_erfc contains erfc(z1*sqrt(t))

    temp_H = gsl_complex_exp(gsl_complex_mul_real(c_exp_1, dtime));
    temp_H = gsl_complex_mul(temp_H, temp_erfc);
    temp_H = gsl_complex_mul(temp_H, c_coeff_1);
    // Now temp_H contains contribution from z1

    temp_z = gsl_complex_mul_real(c_z2, dtime_sqrt); // z2*sqrt(t)
    dmath_cerfc(GSL_REAL(temp_z), GSL_IMAG(temp_z), &temp_erfc.dat[0], &temp_erfc.dat[1], &ierr);
    // Now temp_erfc contains erfc(z2*sqrt(t))
    
    temp_R = gsl_complex_exp(gsl_complex_mul_real(c_exp_2, dtime));
    temp_R = gsl_complex_mul(temp_R, temp_erfc);
    temp_R = gsl_complex_mul(temp_R, c_coeff_2);

    temp_H = gsl_complex_add(temp_H, temp_R);
    // Now temp_H includes contribution from z2

    temp_z = gsl_complex_mul_real(c_z3, dtime_sqrt); // z3*sqrt(t)
    dmath_cerfc(GSL_REAL(temp_z), GSL_IMAG(temp_z), &temp_erfc.dat[0], &temp_erfc.dat[1], &ierr);
    // Now temp_erfc contains erfc(z3*sqrt(t))

    temp_R = gsl_complex_exp(gsl_complex_mul_real(c_exp_3, dtime));
    temp_R = gsl_complex_mul(temp_R, temp_erfc);
    temp_R = gsl_complex_mul(temp_R, c_coeff_3);

    temp_H = gsl_complex_add(temp_H, temp_R);
    // Now temp_H includes contribution from z3

    temp_z = gsl_complex_mul_real(c_z4, dtime_sqrt);
    dmath_cerfc(GSL_REAL(temp_z), GSL_IMAG(temp_z), &temp_erfc.dat[0], &temp_erfc.dat[1], &ierr);
    // Now temp_erfc contains erfc(z4*sqrt(t))

    temp_R = gsl_complex_exp(gsl_complex_mul_real(c_exp_4, dtime));
    temp_R = gsl_complex_mul(temp_R, temp_erfc);
    temp_R = gsl_complex_mul(temp_R, c_coeff_4);

    temp_H = gsl_complex_add(temp_H, temp_R);
    // Now temp_H includes contributon from z4

    H[i] = coeff_B_1*erfc(sqrt(nu_omega*dtime)) + GSL_REAL(temp_H);
    H[i] *= DH0;

  } // for(i=0 ...
  


  //-----------------------------------------------//
  // output solution

  FILE *fp = fopen(OUTPUT_FILE, "w");
  // fprintf(fp, "# density_upper_fluid = %.9le, density_lower_fluid = %.9le\n# density ratio = %.9le\n"
	//   "# kinematic viscosity = %.9le, wave number = %.9le\n"
	//   "# dynamic viscosity of upper fluid = %.9le, dynamic viscosity of lower fluid = %.9le\n"
	//   "# dynamics viscosity ratio = %.9le\n"
	//   "# gravitational acceleration = %.9le\n"
	//   "# surface tension = %.9le, initial amplitude H0 = %.9le\n"
	//   "# omega_0 = %.9le, nu_bar = %.9le, beta = %.9le\n\n",
	//   rho_1, rho_2, rho_2/rho_1,
	//   viscosity, dk,
	//   dynamic_vis_1, dynamic_vis_2,
	//   dynamic_vis_2/dynamic_vis_1,
	//   gravity,
	//   sigma, dh0,
	//   omega_0, nu_bar, beta);
  // fprintf(fp, "variables = t, H, omega_t\n");

  for(i=0;i<=Nsteps;i++) {
    fprintf(fp, "%lf %.12le %.12le \n",
	    T0+i*step_t, H[i], (T0+i*step_t)*omega_0);
  }
  fflush(fp);
  fclose(fp);




  //-----------------------------------------------//
  // clean up
  DMath::del(H);
  gsl_poly_complex_workspace_free(workspace);


  return 0;
}
