#include "./kernel.h"

#include <cuComplex.h>
#include <cstddef>



// per reference ! --> immer per value
// es wird auf einen anderen speicher gegriffen -> es kracht
// es werden immer ein vielfaches von 32 an Thread gestartet -> bei 33 Zeichen werden 64 Threads gestartet
// deswegen wir auch size mitübergeben

// thread nummer ist relative zum block
// block nummer ist relativ zur grafikkarte
// daraus muss ide absolute threadnummer berechnet werden

// divergenten code vermeiden! -> eine der größten Bremsen
__global__ void kernel(pfc::BGR_4_t * const p_dst, std::size_t const size, int const imag_max, int const imag_min, int const real_max, int const real_min, int const threshold, int const iteration, int const bmp_width, int const bmp_height) {
	// blockDim Anzahl der Threads pro block
	auto const t{ blockIdx.x * blockDim.x + threadIdx.x }; // -> absolute Threadnumber

	//TODO: Get the x and y position of the threadnumber

	//TODO: Calculate and color the appropriate pixel
	int x_pos = t % bmp_width;
	int y_pos = t / bmp_width;

	double x_normalize = { x_pos * 1.0 / bmp_width * (real_max - real_min) + real_min };
	double y_normalize = { y_pos * 1.0 / bmp_height * (imag_max - imag_min) + imag_min };

	cuDoubleComplex c = make_cuDoubleComplex(x_normalize, y_normalize);;
	cuDoubleComplex zi = make_cuDoubleComplex(0,0);
	cuDoubleComplex zn = make_cuDoubleComplex(0,0);
	for (size_t i = 0; i < iteration; i++) {
		zn = cuCadd(cuCmul(zi, zi),c);
		zi = zn;
		if (cuCabs(zn) > threshold)
			break;
	}

	if (cuCabs(zn) > threshold)
		p_dst[t] = { 255, 255, 255 };
	else
		p_dst[t] = {0, 0, 0 };
}

cudaError_t call_kernel(dim3 const big, dim3 const tib, pfc::BGR_4_t * p_dst, std::size_t const size, int const imag_max, int const imag_min, int const real_max, int const real_min, int const threshold, int const iteration, int const bmp_width, int const bmp_height) {
	// blocks in grid
	// threads in block
	// 3 kernel a 512 threads

	kernel << <big, tib >> > (p_dst, size, imag_max, imag_min, real_max, real_min, threshold, iteration, bmp_width, bmp_height);
	return cudaGetLastError();
}