#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "pfc_types.h"
#include <cstddef>

cudaError_t call_kernel(dim3 big, dim3 tib, pfc::BGR_4_t * p_dst, pfc::BGR_4_t const  * p_src, std::size_t size, int const imag_max, int const imag_min, int const real_max, int const real_min, int const threshold, int const iteration, int const bmp_width, int const bmp_height);