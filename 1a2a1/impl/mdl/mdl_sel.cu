#include "mdl.cuh"

void mdl_flt_melange(
	float * f,
	uint bloques, uint f_par_bloque,
	float p_filtre, float p_changement)
{
	float autres[N];
	
	FOR(0, i, bloques*f_par_bloque)
	{
		//
		if (rnd() > p_filtre) continue;

		//
		FOR(0, j, N) autres[j] = rnd();

		//	Normalisation
		float max=autres[0], min=autres[1];
		FOR(0, j, N) {
			if (autres[j] > max) max = autres[j];
			if (autres[j] < min) min = autres[j];
		}
		FOR(0, j, N) autres[j] = (autres[j] - min)/(max - min);

		//	Mutation
		FOR(0, j, N) autres[j] = (1-p_changement)*f[i*N + j] + autres[j]*p_changement;

		//	Normalisation 
		max=autres[0]; min=autres[1];
		FOR(0, j, N) {
			if (autres[j] > max) max = autres[j];
			if (autres[j] < min) min = autres[j];
		}
		FOR(0, j, N) f[i*N + j] = (autres[j] - min)/(max - min);
	};
};