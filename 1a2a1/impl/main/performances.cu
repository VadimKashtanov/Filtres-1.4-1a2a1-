#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void  performances() {
	/*ASSERT(C == 6);
	uint ST[C] = {1024, 256, 64, 32, 8, 1};
	uint bloques      = 64;
	uint f_par_bloque = 16;
	uint lignes[bloques];
	FOR(0, i, bloques) lignes[i] = 0;//rand() % EMA_INTS;
	Mdl_t * mdl = cree_mdl(ST, bloques, f_par_bloque, lignes);

	uint t0 = DEPART;
	uint t1 = FIN-(FIN-DEPART)%16;
	printf("t0=%i t1=%i FIN=%i (t1-t0=%i, %%16=%i)\n", t0, t1, FIN, t1-t0, (t1-t0)%16);
	mdl_verif(mdl);

	printf("Mode 0 : ");
	MESURER(mdl_aller_retour(mdl, t0, t1, 0));
	mdl_verif(mdl);

	printf("Mode 1 : ");
	MESURER(mdl_aller_retour(mdl, t0, t1, 1));
	mdl_verif(mdl);

	printf("Mode 2 : ");
	MESURER(mdl_aller_retour(mdl, t0, t1, 2));
	mdl_verif(mdl);

	printf("Mode 3 : ");
	MESURER(mdl_aller_retour(mdl, t0, t1, 3));
	mdl_verif(mdl);

	liberer_mdl(mdl);*/
};