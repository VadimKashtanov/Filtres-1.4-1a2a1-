#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

float pourcent_masque[C] = {
	0,
	0.30,
	0.50,
	0.80,
	0.60,
	0.30,
	0.10,
	0.0
};

float alternative(
	uint __srand,
	char * source, char * sortie,
	float p_f, float p_ch,
	uint t0, uint t1)
{
	Mdl_t * mdl = ouvrire_mdl(source);
	//
	mdl_flt_melange(mdl->f, mdl->bloques, mdl->f_par_bloque, p_f, p_ch);
	CONTROLE_CUDA(cudaMemcpy(mdl->f__d, mdl->f, sizeof(float)*mdl->bloques*mdl->f_par_bloque*N, cudaMemcpyHostToDevice));
	mdl_diff_f(mdl);
	mdl_verif(mdl);
	//
	srand(__srand);
	uint Meta_Iterations_total =   3;
	uint _T_mini_paquet        = 16*100;
	uint Iteration_masqués     =  70;
	uint Itération_normale     =  30;
	optimisation_super(
		mdl, t0, t1,
		1e-5, 1000.0,
		Meta_Iterations_total,
		RMSPROP,
		pourcent_masque,
		_T_mini_paquet,
		Iteration_masqués, Itération_normale
	);
	//
	float ret = 100*mdl_pred(mdl, t0, t1, 3);
	//
	mdl_gpu_vers_cpu(mdl);
	mdl_verif(mdl);
	ecrire_mdl(mdl, sortie);
	liberer_mdl(mdl);
	//
	return ret;
};

PAS_OPTIMISER()
int main(int argc, char ** argv) {
	//	-- Init --
	srand(0);
	cudaSetDevice(0);

	titre(" Charger tout ");  charger_tout();
	MSG("Shared filtres pas encore fait. Juste un copier-coller de la version naive");
	//titre("Verifier  FILTRES"); verif_filtres();
	//titre("Verifier  DOT1D"); 	 verif_do1d();
	//titre("Verifier  S"); 		 verifier_S();
	//titre(" Performances "); performances();

	//===============
	titre("  Programme Generale  ");

	/*uint ST[C] = {256, 128, 64, 32, 16, 8, 4, 1};
	uint bloques      = 16;
	uint f_par_bloque = 16;
	uint lignes[bloques] = {
		0,0,0,0,0,0,0,
		1, 1,
		2, 2,
		3, 4,
		6,
		8,
		10
	};
	Mdl_t * mdl = cree_mdl(ST, bloques, f_par_bloque, lignes);*/
	Mdl_t * mdl = ouvrire_mdl("mdl.bin");

	plumer_mdl(mdl);

	//	================= Initialisation ==============
	uint t0 = DEPART;
	uint t1 = ROND_MODULO(FIN, 16);
	printf("t0=%i t1=%i FIN=%i (t1-t0=%i, %%16=%i)\n", t0, t1, FIN, t1-t0, (t1-t0)%16);

	/*uint Meta_Iterations_total = 10;
	uint _T_mini_paquet        = 16*500;
	uint Iteration_masqués     = 3000;
	uint Itération_normale     = 1000;
	optimisation_super(
		mdl, t0, t1,
		1e-5, 1000.0,
		Meta_Iterations_total,
		RMSPROP,
		liste<float>(C,
			0,
			0.40,
			0.40,
			0.80,
			0.60,
			0.30,
			0.10,
			0.0
		),
		_T_mini_paquet,
		Iteration_masqués, Itération_normale
	);*/

	float ancien = 100*mdl_pred(mdl, t0, t1, 3);
	printf("%%%% pred = %f\n", ancien);

	printf("gain = %f\n", mdl_gain(mdl, t0, t0+15*16, 3));
	printf("gain = %f\n", mdl_gain(mdl, t0+10000, t0+10000+15*16, 3));
	printf("gain = %f\n", mdl_gain(mdl, t0+30000, t0+30000+15*16, 3));
	printf("gain = %f\n", mdl_gain(mdl, t1-16*15, t1, 3));
	//
	mdl_gpu_vers_cpu(mdl);
	ecrire_mdl(mdl, "mdl.bin");
	liberer_mdl(mdl);


	//	=================    Versions   =================
	uint __srand = time(NULL);
	//
	float p_filtres    = 0.10;
	float p_changement = 0.20;
	srand(time(NULL));
	float MEILLEUR_PERF = alternative(
		__srand,
		"mdl.bin", "mdl_nouveau.bin",
		p_filtres, p_changement,
		t0, t1);
	//
	uint meilleur = 0;
	//
	uint VERSIONS = 20;
	FOR(0, v, VERSIONS) {
		srand(time(NULL));
		float perf_tmp = alternative(
			__srand,	//srand
			"mdl.bin", "mdl_tmp.bin",
			p_filtres, p_changement,
			t0, t1);
		//
		if (perf_tmp > MEILLEUR_PERF) {
			MEILLEUR_PERF = perf_tmp;
			ASSERT(system("mv mdl_tmp.bin mdl_nouveau.bin") == 0);
			meilleur = v + 1;
		}
	}


	//	================= Optimisation Finale ===================
	printf("------------ Meilleur = %i ----------\n", meilleur);

	mdl = ouvrire_mdl("mdl_nouveau.bin");
	//	Sur toutes les donnes
	uint Meta_Iterations_total = 10;
	uint _T_mini_paquet        = 16*300;
	uint Iteration_masqués     = 1000;
	uint Itération_normale     =  500;
	optimisation_super(
		mdl, t0, t1,
		1e-5, 1000.0,
		Meta_Iterations_total,
		RMSPROP,
		pourcent_masque,
		_T_mini_paquet,
		Iteration_masqués, Itération_normale
	);
	//
	printf("Meilleure mutation : perf=%+f%%\n", 100*mdl_pred(mdl, t0, t1, 3));
	printf(" Ancienne version  : perf=%+f%%\n", ancien);
	//
	mdl_gpu_vers_cpu(mdl);
	ecrire_mdl(mdl, "mdl.bin");
	liberer_mdl(mdl);

	//	-- Fin --
	liberer_tout();
};