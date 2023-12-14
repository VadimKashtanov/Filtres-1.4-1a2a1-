#include "opti.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

static __global__ void kerd_opti_simple(
	float * p, float * dp, float alpha, uint P, float div)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < P) {
		p[thx] -= alpha * dp[thx] / div;
	}
};

static __global__ void kerd_opti_simple_masque(
	float * p, float * dp, float alpha, uint P, float div, uint * masque)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < P) {
		if (masque[thx] == 0)
			p[thx] -= alpha * dp[thx] / div;
	}
};

void opti_simple(Mdl_t * mdl, float alpha, float div, uint ** masque) {
	FOR(1, c, C) {
		uint P = (mdl->ST[c-1]+1)*mdl->ST[c];
		if (masque == 0) {
			kerd_opti_simple<<<dim3(KERD(P, 1024)), dim3(1024)>>>(
				mdl->p__d[c], mdl->dp__d[c], alpha, P, div
			);
		} else {
			kerd_opti_simple_masque<<<dim3(KERD(P, 1024)), dim3(1024)>>>(
				mdl->p__d[c], mdl->dp__d[c], alpha, P, div, masque[c]
			);
		}
	};
	ATTENDRE_CUDA();
};