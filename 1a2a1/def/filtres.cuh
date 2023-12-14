#pragma once

#include "marchee.cuh"

void  intel_filtres(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y);

//	----------------------------

void nvidia_filtres_naive(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y);

void nvidia_filtres_shared(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y);

void nvidia_filtres(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f, float * dif_f,
	float * y,
	uint mode);

//	----------------------------

void verif_filtres();