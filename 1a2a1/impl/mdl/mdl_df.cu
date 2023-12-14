#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

//	Modes :
//		0 - cpu
//		1 - naive
//		2 - naive_filtres & shared_dot1d
//		3 - naive_filtres & shared_2_dot1d

#define MODE_CPU 0
#define MODE_NAIF 1
#define MODE_NAIF_SHARED 2
#define MODE_NAIF_SHARED2 3

static void mdl_df_patron(
	Mdl_t * mdl, uint t0, uint t1,
	int perceptron_mode)
{
	for (int c=C-1; c >= 1; c--) {
		if (perceptron_mode == -1) {
			d_intel_dot1d(
				mdl->ST[c-1], mdl->ST[c],
				t0, (t1-t0),
				mdl->y[c-1], mdl->y[c],
				mdl->p[c],
				mdl->locd[c],
				mdl->dy[c],
				mdl->dy[c-1],
				mdl->dp[c]);
		} else {
			d_nvidia_dot1d(
				mdl->ST[c-1], mdl->ST[c],
				t0, (t1-t0),
				mdl->y__d[c-1], mdl->y__d[c],
				mdl->p__d[c],
				mdl->locd__d[c],
				mdl->dy__d[c],
				mdl->dy__d[c-1],
				mdl->dp__d[c],
				perceptron_mode);
		}
	}
};

void mdl_df(Mdl_t * mdl, uint t0, uint t1, uint mode) {
	if (mode == MODE_CPU) {
		int perceptron_mode = -1;
		mdl_df_patron(mdl, t0, t1, perceptron_mode);
	} else if (mode == MODE_NAIF) {
		int perceptron_mode = 0;
		mdl_df_patron(mdl, t0, t1, perceptron_mode);
	} else if (mode == MODE_NAIF_SHARED) {
		int perceptron_mode = 1;
		mdl_df_patron(mdl, t0, t1, perceptron_mode);
	} else if (mode == MODE_NAIF_SHARED2) {
		int perceptron_mode = 2;
		mdl_df_patron(mdl, t0, t1, perceptron_mode);
	} else {
		ERR("Pas de mode %i pour mdl_f()", mode);
	}
};