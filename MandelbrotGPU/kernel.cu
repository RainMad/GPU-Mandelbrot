#include "./kernel.h"

#include <cstddef>



// per reference ! --> immer per value
// es wird auf einen anderen speicher gegriffen -> es kracht
// es werden immer ein vielfaches von 32 an Thread gestartet -> bei 33 Zeichen werden 64 Threads gestartet
// deswegen wir auch size mitübergeben

// thread nummer ist relative zum block
// block nummer ist relativ zur grafikkarte
// daraus muss ide absolute threadnummer berechnet werden

// divergenten code vermeiden! -> eine der größten Bremsen
__global__ void kernel(pfc::BGR_4_t * const p_dst, pfc::BGR_4_t const * const p_src, std::size_t const size, int const imag_max, int const imag_min, int const real_max, int const real_min, int const threshold, int const iteration, int const bmp_width, int const bmp_height) {
	// blockDim Anzahl der Threads pro block
	auto const t{ blockIdx.x * blockDim.x + threadIdx.x }; // -> absolute Threadnumber

	//double x_normalize{ x*1.0 / bitMapWidth * (real_max - real_min) + real_min };
	//double y_normalize{ y*y_help_value + imag_min };


	if (t < size) {
		p_dst[t] = p_src[t];
	}
}

cudaError_t call_kernel(dim3 const big, dim3 const tib, pfc::BGR_4_t * p_dst, pfc::BGR_4_t const * const p_src, std::size_t const size, int const imag_max, int const imag_min, int const real_max, int const real_min, int const threshold, int const iteration, int const bmp_width, int const bmp_height) {
	// blocks in grid
	// threads in block
	// 3 kernel a 512 threads

	kernel << <big, tib >> > (p_dst, p_src, size, imag_max, imag_min, real_max, real_min, threshold, iteration, bmp_width, bmp_height);
	return cudaGetLastError();
}