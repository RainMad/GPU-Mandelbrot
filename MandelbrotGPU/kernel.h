#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstddef>

cudaError_t call_kernel(dim3 big, dim3 tib, char * p_dst, char const * p_src, std::size_t size);