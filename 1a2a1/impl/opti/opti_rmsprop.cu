#include "opti.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

#define rms_alpha 0.95

static __global__ void kerd_opti_rmsprop(
	float * p, float * dp, float * g,
	float alpha, uint P, float div)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < P) {
		float _grad = dp[thx] / div;
		float _g = rms_alpha*g[thx] + _grad*_grad;
		p[thx] -= alpha * _grad / (sqrtf(_g) + 1e-5);
		g[thx] = _g;
	}
};

static __global__ void kerd_opti_rmsprop_masque(
	float * p, float * dp, float * g,
	float alpha, uint P, float div, uint * masque)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < P) {
		if (masque[thx] == 0) {
			float _grad = dp[thx] / div;
			float _g = rms_alpha*g[thx] + _grad*_grad;
			p[thx] -= alpha * _grad / (sqrtf(_g) + 1e-5);
			g[thx] = _g;
		}
	}
};

Rmsprop_t * cree_rmsprop(
	Mdl_t * mdl)
{
	Rmsprop_t * ret = alloc<Rmsprop_t>(1);
	ret->g[0] = 0;
	FOR(1, c, C) ret->g[c] = cudazero<float>(poids_couche(mdl, c));
	return ret;
};

void liberer_rmsprop(Rmsprop_t * rmsprop) {
	FOR(1, c, C) cudafree<float>(rmsprop->g[c]);
	free(rmsprop);
};

void opti_rmsprop(
	Mdl_t * mdl, Rmsprop_t * rmsprop,
	float alpha, float div, uint ** masque)
{
	FOR(1, c, C) {
		uint P = (mdl->ST[c-1]+1)*mdl->ST[c];
		
		if (masque == 0) {
			kerd_opti_rmsprop<<<dim3(KERD(P, 1024)), dim3(1024)>>>(
				mdl->p__d[c], mdl->dp__d[c], rmsprop->g[c],
				alpha, P, div
			);
		} else {
			kerd_opti_rmsprop_masque<<<dim3(KERD(P, 1024)), dim3(1024)>>>(
				mdl->p__d[c], mdl->dp__d[c], rmsprop->g[c],
				alpha, P, div, masque[c]
			);
		}
	};
	ATTENDRE_CUDA();
};