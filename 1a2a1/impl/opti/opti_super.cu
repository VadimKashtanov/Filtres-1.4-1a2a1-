#include "opti.cuh"

void optimisation_super(
	Mdl_t * mdl,
	uint t0, uint t1,
	float alpha, float div,
	uint Meta_Iterations_total,
	uint methode,
	float * pourcent_masque,
	uint _T_mini_paquet,
	uint Iteration_masqués, uint Itération_normale)
{
	FOR(0, k, Meta_Iterations_total) {
		uint ** masque = cree_masque(mdl, pourcent_masque);
		//
		uint _t0 = t0 + (rand()%(t1-_T_mini_paquet-t0));
		optimiser_masque(
			mdl, _t0, _t0+_T_mini_paquet, alpha, div,
			Iteration_masqués,
			RMSPROP,
			masque
		);
		//
		liberer_masque(masque);
		//	Optimisation pour reequilibrer les poids
		optimiser(mdl, t0, t1, alpha, div, Itération_normale, RMSPROP);
	}
};