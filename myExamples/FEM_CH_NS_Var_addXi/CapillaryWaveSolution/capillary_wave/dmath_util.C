
#include "gsl/gsl_complex.h"
#include "gsl/gsl_complex_math.h"
#include "dmath_util.h"

static void my_wofz(double xi, double yi, double *u, double *v, int *info)
{
  wofz_(&xi, &yi, u, v,info);
}



static void my_merrcz(double *z, double *w, int *info)
{
  merrcz_(z, w, info);
}

static void my_merrcz(double xi, double yi, double *wr, double *wi, int *info)
{
  double z[2], w[2];
  z[0] = xi;
  z[1] = yi;
  merrcz_(z, w, info);
  
  *wr = w[0];
  *wi = w[1];
}

static void my_merrcz(double xi, double yi, double *w, int *info)
{
  double z[2];
  z[0] = xi;
  z[1] = yi;
  merrcz_(z, w, info);
}


void dmath_cerf(double xi, double yi, double *u, double *v, int *info)
{
  // compute erf(z) = 1/sqrt(pi) * \int_{0 to z} exp(-t^2) dt
  // where z = (xi,yi) is complex number
  //
  // Note: erfc(z) = exp(-z^2)*W(i*z), erf(z) = 1-erfc(z)
  //       where W(z) is the Faddeeva function, implemented in wofz(...)
  //

  double wr, wi;
  double zr, zi;
  int ierr;

  // wofz, W(i*z)
  // i*(xi,yi) = (-yi, xi)
  zr = -yi;
  zi = xi;
  my_wofz(zr, zi, &wr, &wi, &ierr);

  gsl_complex Z = gsl_complex_rect(xi, yi);
  gsl_complex W = gsl_complex_rect(wr, wi);
  gsl_complex res;

  res = gsl_complex_negative(gsl_complex_mul(Z, Z)); // Z*Z
  res = gsl_complex_exp(res); // exp(-Z*Z)
  res = gsl_complex_mul(res, W); // exp(-Z*Z)*W(i*Z)
  res = gsl_complex_sub_real(res, 1.0); // exp(-Z*Z)*W(i*Z) - 1.0
  res = gsl_complex_negative(res); // 1.0 - exp(-Z*Z)*W(i*Z)

  *u = GSL_REAL(res);
  *v = GSL_IMAG(res);

  *info = 0;
}


void dmath_cerf_imsl(double xi, double yi, double *u, double *v, int *info)
{
  // compute erf(z) = 1/sqrt(pi) * \int_{0 to z} exp(-t^2) dt
  // where z = (xi,yi) is complex number
  //
  // Note: erfc(z) = exp(-z^2)*W(i*z), erf(z) = 1 - erf(z)
  //       where W(z) is the Faddeeva function, implemented in wofz(...)
  //

  double wr, wi;
  double zr, zi;
  int ierr;

  // wofz, W(i*z)
  // i*(xi,yi) = (-yi, xi)
  zr = -yi;
  zi = xi;
  my_merrcz(zr, zi, &wr, &wi, &ierr);

  gsl_complex Z = gsl_complex_rect(xi, yi);
  gsl_complex W = gsl_complex_rect(wr, wi);
  gsl_complex res;

  res = gsl_complex_negative(gsl_complex_mul(Z, Z)); // Z*Z
  res = gsl_complex_exp(res); // exp(-Z*Z)
  res = gsl_complex_mul(res, W); // exp(-Z*Z)*W(i*Z)
  res = gsl_complex_sub_real(res, 1.0); // exp(-Z*Z)*W(i*Z) - 1.0
  res = gsl_complex_negative(res); // 1.0 - exp(-Z*Z)*W(i*Z)

  *u = GSL_REAL(res);
  *v = GSL_IMAG(res);

  *info = 0;
}



void dmath_cerfc(double xi, double yi, double *u, double *v, int *info)
{
  // compute erfc(z) = 1- 1/sqrt(pi) * \int_{0 to z} exp(-t^2) dt
  // where z = (xi,yi) is complex number
  //
  // Note: erfc(z) = exp(-z^2)*W(i*z),
  //       where W(z) is the Faddeeva function, implemented in wofz(...)
  //

  double wr, wi;
  double zr, zi;
  int ierr;

  // wofz, W(i*z)
  // i*(xi,yi) = (-yi, xi)
  zr = -yi;
  zi = xi;
  my_wofz(zr, zi, &wr, &wi, &ierr);

  gsl_complex Z = gsl_complex_rect(xi, yi);
  gsl_complex W = gsl_complex_rect(wr, wi);
  gsl_complex res;

  res = gsl_complex_negative(gsl_complex_mul(Z, Z)); // Z*Z
  res = gsl_complex_exp(res); // exp(-Z*Z)
  res = gsl_complex_mul(res, W); // exp(-Z*Z)*W(i*Z)
  //res = gsl_complex_sub_real(res, 1.0); // exp(-Z*Z)*W(i*Z) - 1.0
  //res = gsl_complex_negative(res); // 1.0 - exp(-Z*Z)*W(i*Z)

  *u = GSL_REAL(res);
  *v = GSL_IMAG(res);

  *info = 0;
}


void dmath_cerfc_imsl(double xi, double yi, double *u, double *v, int *info)
{
  // compute erfc(z) = 1- 1/sqrt(pi) * \int_{0 to z} exp(-t^2) dt
  // where z = (xi,yi) is complex number
  //
  // Note: erfc(z) = exp(-z^2)*W(i*z),
  //       where W(z) is the Faddeeva function, implemented in wofz(...)
  //

  double wr, wi;
  double zr, zi;
  int ierr;

  // wofz, W(i*z)
  // i*(xi,yi) = (-yi, xi)
  zr = -yi;
  zi = xi;
  my_merrcz(zr, zi, &wr, &wi, &ierr);

  gsl_complex Z = gsl_complex_rect(xi, yi);
  gsl_complex W = gsl_complex_rect(wr, wi);
  gsl_complex res;

  res = gsl_complex_negative(gsl_complex_mul(Z, Z)); // Z*Z
  res = gsl_complex_exp(res); // exp(-Z*Z)
  res = gsl_complex_mul(res, W); // exp(-Z*Z)*W(i*Z)
  //res = gsl_complex_sub_real(res, 1.0); // exp(-Z*Z)*W(i*Z) - 1.0
  //res = gsl_complex_negative(res); // 1.0 - exp(-Z*Z)*W(i*Z)

  *u = GSL_REAL(res);
  *v = GSL_IMAG(res);

  *info = 0;
}


void dmath_cerf_cerfc(double xi, double yi,
		      double *u_erf, double *v_erf,
		      double *u_erfc, double *v_erfc,
		      int *info)
{
  // compute cerf(z) and cerfc(z) at the same time

  // compute erf(z) = 1/sqrt(pi) * \int_{0 to z} exp(-t^2) dt
  // where z = (xi,yi) is complex number
  //
  // Note: erfc(z) = exp(-z^2)*W(i*z), erf(z) = 1-erfc(z)
  //       where W(z) is the Faddeeva function, implemented in wofz(...)
  //

  if(!u_erf && !v_erf && !u_erfc && !v_erfc) {
    *info = 0;
    return;
  }

  double wr, wi;
  double zr, zi;
  //int ierr;

  // wofz, W(i*z)
  // i*(xi,yi) = (-yi, xi)
  zr = -yi;
  zi = xi;
  my_wofz(zr, zi, &wr, &wi, info);

  gsl_complex Z = gsl_complex_rect(xi, yi);
  gsl_complex W = gsl_complex_rect(wr, wi);
  gsl_complex res;

  res = gsl_complex_negative(gsl_complex_mul(Z, Z)); // Z*Z
  res = gsl_complex_exp(res); // exp(-Z*Z)
  res = gsl_complex_mul(res, W); // exp(-Z*Z)*W(i*Z)

  // Now res contains erfc(z)
  if(u_erfc) *u_erfc = GSL_REAL(res);
  if(v_erfc) *v_erfc = GSL_IMAG(res);

  res = gsl_complex_sub_real(res, 1.0); // exp(-Z*Z)*W(i*Z) - 1.0
  res = gsl_complex_negative(res); // 1.0 - exp(-Z*Z)*W(i*Z)

  // Now res contains erf(z)

  if(u_erf) *u_erf = GSL_REAL(res);
  if(v_erf) *v_erf = GSL_IMAG(res);

  //*info = 0;  
}


void dmath_cerf_cerfc_imsl(double xi, double yi,
			   double *u_erf, double *v_erf,
			   double *u_erfc, double *v_erfc,
			   int *info)
{
  // compute cerf(z) and cerfc(z) at the same time

  // compute erf(z) = 1/sqrt(pi) * \int_{0 to z} exp(-t^2) dt
  // where z = (xi,yi) is complex number
  //
  // Note: erfc(z) = exp(-z^2)*W(i*z), erf(z) = 1-erfc(z)
  //       where W(z) is the Faddeeva function, implemented in wofz(...)
  //

  if(!u_erf && !v_erf && !u_erfc && !v_erfc) {
    *info = 0;
    return;
  }

  double wr, wi;
  double zr, zi;
  //int ierr;

  // wofz, W(i*z)
  // i*(xi,yi) = (-yi, xi)
  zr = -yi;
  zi = xi;
  my_merrcz(zr, zi, &wr, &wi, info);

  gsl_complex Z = gsl_complex_rect(xi, yi);
  gsl_complex W = gsl_complex_rect(wr, wi);
  gsl_complex res;

  res = gsl_complex_negative(gsl_complex_mul(Z, Z)); // Z*Z
  res = gsl_complex_exp(res); // exp(-Z*Z)
  res = gsl_complex_mul(res, W); // exp(-Z*Z)*W(i*Z)

  // Now res contains erfc(z)
  if(u_erfc) *u_erfc = GSL_REAL(res);
  if(v_erfc) *v_erfc = GSL_IMAG(res);

  res = gsl_complex_sub_real(res, 1.0); // exp(-Z*Z)*W(i*Z) - 1.0
  res = gsl_complex_negative(res); // 1.0 - exp(-Z*Z)*W(i*Z)

  // Now res contains erf(z)

  if(u_erf) *u_erf = GSL_REAL(res);
  if(v_erf) *v_erf = GSL_IMAG(res);

  //*info = 0;  
}
