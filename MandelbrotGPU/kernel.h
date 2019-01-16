#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Helpers\pfc_types.h"
#include <cstddef>

cudaError_t call_kernel(
	dim3 big, 
	dim3 tib, 
	pfc::BGR_4_t * p_dst, 
	std::size_t size_x, 
	double imag_max,
	double imag_min,
	double real_max,
	double real_min,
	int const threshold,
	int const iteration, 
	int const bmp_width, 
	int const bmp_height,
	int const amount_of_images, 
	double const point_real, 
	double const point_imag, 
	double const zoom_factor,
	int const image_number);