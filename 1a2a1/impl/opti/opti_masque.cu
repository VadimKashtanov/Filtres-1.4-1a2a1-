#include "opti.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

#define PSEUDO_ALEA(x) (float)((1234*x + 4563) % 1000)/1000.0

static __global__ void kerd_cree_masque(
	uint graine,
	uint * masque, uint POIDS, float p)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thx < POIDS) {
		masque[thx] = (PSEUDO_ALEA(graine+thx) >= p ? 0 : 1);
		//	0 - pas masqué
		//	1 - masqué

		//	masqué = ne sera pas optimisé
	}
};

uint ** cree_masque(
	Mdl_t * mdl, float * p)
{
	uint ** masque = alloc<uint*>(C);
	FOR (1, c, C) {
		uint POIDS = poids_couche(mdl, c);
		masque[c] = cudalloc<uint>(POIDS);
		kerd_cree_masque<<<dim3(KERD(POIDS,1024)), dim3(1024)>>>(
			rand() % 10000,
			masque[c], POIDS, p[c]
		);
		ATTENDRE_CUDA();
	};
	return masque;
};

void liberer_masque(uint ** masque) {
	FOR(1, c, C) cudafree<uint>(masque[c]);
	free(masque);
}

void optimiser_masque(
	Mdl_t * mdl,
	uint t0, uint t1,
	float alpha, float div,
	uint I,
	uint methode,
	uint ** masque)
{
	__interne_optimiser(
		mdl,
		t0, t1,
		alpha, div,
		I,
		methode,
		masque);
}