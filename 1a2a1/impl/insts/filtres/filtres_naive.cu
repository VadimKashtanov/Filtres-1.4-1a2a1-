#include "filtres.cuh"

#define BLOQUE_T  8
#define BLOQUE_B  8
#define BLOQUE_FB 16

static __device__ float filtre_device(float * x, float * dif_x, float * f, float * dif_f) {
	float s = 0, d = 0;
	FOR(0, i, N-1) {
		s += sqrtf(1 + fabs(  x[i]   -   f[i]  ));
		d += powf((1 + fabs(dif_x[i] - dif_f[i])), 2);
	};
	s += sqrtf(1 + fabs(x[N-1] - f[N-1]));

	s = s/8-1;
	d = d/7-1;

	return 2*expf(-s*s -d*d)-1;
};


static __global__ void kerd_filtre_naive(	//	2 version : 1 stricte et une non stricte
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _b = threadIdx.y + blockIdx.y * blockDim.y;
	uint _f = threadIdx.z + blockIdx.z * blockDim.z;

	if (_t < T && _b < bloques && _f < f_par_bloque) {
		y[(depart+_t)*(bloques*f_par_bloque) + _b*f_par_bloque + _f] = filtre_device(
			x + ligne[_b]*PRIXS*N_FLTR + (depart+_t)*N_FLTR,
			dif_x + ligne[_b]*PRIXS*N_FLTR + (depart+_t)*N_FLTR,
			f     + _b*f_par_bloque*N     + _f*N,
			dif_f + _b*f_par_bloque*(N-1) + _f*(N-1)
		);
	}
};

void nvidia_filtres_naive(	//	2 version : 1 stricte et une non stricte
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y)
{
	kerd_filtre_naive<<<dim3(KERD(T, BLOQUE_T), KERD(bloques, BLOQUE_B), KERD(f_par_bloque, BLOQUE_FB)), dim3(BLOQUE_T, BLOQUE_B, BLOQUE_FB)>>>(
		depart, T,
		bloques, f_par_bloque, ligne,
		x, dif_x,
		f, dif_f,
		y);
	ATTENDRE_CUDA();
}