#pragma once

#include "mdl.cuh"

#define SGD 0
#define RMSPROP 1

//	======= SGD =============

void opti_simple(
	Mdl_t * mdl,
	float alpha, float div,
	uint ** masque);

//	======= RMSPROP ========

typedef struct {
	float * g[C];
} Rmsprop_t;

Rmsprop_t * cree_rmsprop(
	Mdl_t * mdl);

void opti_rmsprop(
	Mdl_t * mdl, Rmsprop_t * rmsprop,
	float alpha, float div,
	uint ** masque);

void liberer_rmsprop(Rmsprop_t * rmsprop);

//	======================================
//	======= Optimisation Generale ========
//	======================================

typedef union {
	uint sgd;
	Rmsprop_t * rmsprop;
} Opti_classe_t;

void __interne_optimiser(
	Mdl_t * mdl,
	uint t0, uint t1,
	float alpha, float div,
	uint I,
	uint methode,
	uint ** masque);

void optimiser(
	Mdl_t * mdl,
	uint t0, uint t1,
	float alpha, float div,
	uint I,
	uint methode);

//	--- Version avec Masque ---

uint ** cree_masque(
	Mdl_t * mdl, float * p);

void optimiser_masque(
	Mdl_t * mdl,
	uint t0, uint t1,
	float alpha, float div,
	uint I,
	uint methode,
	uint ** masque);

void liberer_masque(uint ** masque);

//	--- Super optimisation ---
//	normale et masque

void optimisation_super(
	Mdl_t * mdl,
	uint t0, uint t1,
	float alpha, float div,
	uint Meta_Iterations_total,
	uint methode,
	float * p,
	uint _T_mini_paquet,
	uint Iteration_masqués, uint Itération_normale);