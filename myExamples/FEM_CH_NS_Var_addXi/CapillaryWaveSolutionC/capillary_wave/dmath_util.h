//******************************************************//
// DMath utility routines
//******************************************************//

#ifndef DMATH_UTIL_H
#define DMATH_UTIL_H


#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

  void wofz_(double *xi, double *yi, double *u, double *v, int *flag);
  void merrcz_(double *z, double *w, int *ierr);

#ifdef __cplusplus
}
#endif // __cplusplus

static void my_wofz(double xi, double yi, double *u, double *v, int *info);

static void my_merrcz(double *z, double *w, int *info);
static void my_merrcz(double xi, double yi, double *wr, double *wi, int *info);
static void my_merrcz(double xi, double yi, double *w, int *info);

void dmath_cerf(double xi, double yi, double *u, double *v, int *info);
void dmath_cerfc(double xi, double yi, double *u, double *v, int *info);

void dmath_cerf_imsl(double xi, double yi, double *u, double *v, int *info);
void dmath_cerfc_imsl(double xi, double yi, double *u, double *v, int *info);

void dmath_cerf_cerfc(double xi, double yi,
		      double *u_erf, double *v_erf,
		      double *u_erfc, double *v_erfc,
		      int *info);
void dmath_cerf_cerfc_imsl(double xi, double yi,
			   double *u_erf, double *v_erf,
			   double *u_erfc, double *v_erfc,
			   int *info);


#endif // DMATH_UTIL_H
