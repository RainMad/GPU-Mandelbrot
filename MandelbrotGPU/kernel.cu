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

__constant__ pfc::BGR_4_t static lookUp[32] = {
	{0, 0, 0},
	{66, 30, 15},
	{25, 7, 26},
	{9, 1, 47},
	{4, 4, 73},
	{0, 7, 100},
	{12, 44, 138},
	{57, 125, 209},
	{134, 181, 229},
	{211, 236, 248},
	{241, 233, 191},
	{248, 201, 95},
	{255, 170, 0},
	{204, 128, 0},
	{153, 87, 0},
	{116, 62, 3},
	{126, 72, 13},
	{136, 82, 23},
	{146, 92, 33},
	{156, 102, 43},
	{166, 112, 53},
	{176, 122, 63},
	{186, 132, 73},
	{196, 142, 83},
	{206, 152, 93},
	{216, 162, 103},
	{226, 172, 113},
	{236, 182, 123},
	{246, 192, 133},
	{250, 202, 143},
	{253, 212, 153},
	{255, 222, 163},
};


__device__ double pow(double x, double y);

// divergenten code vermeiden! -> eine der größten Bremsen
__global__ void kernel(pfc::BGR_4_t * const p_dst, 
	std::size_t const size, 
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
	int const image_number) {
	// blockDim Anzahl der Threads pro block
	auto const t{ blockIdx.x * blockDim.x + threadIdx.x }; // -> absolute Threadnumber

	if (t > size)
		return;

	int x_pos = (t - (t / (bmp_width*bmp_height) * bmp_width*bmp_height)) % bmp_width;
	int y_pos = (t - (t / (bmp_width*bmp_height) * bmp_width*bmp_height)) / bmp_width;

	int image_n = t / (bmp_width*bmp_height);
	real_min = point_real - (point_real - real_min) * pow(zoom_factor, image_number + image_n);
	real_max = point_real + (real_max - point_real) * pow(zoom_factor, image_number + image_n);
	imag_max = point_imag + (imag_max - point_imag) * pow(zoom_factor, image_number + image_n);
	imag_min = point_imag - (point_imag - imag_min) * pow(zoom_factor, image_number + image_n);

	double x_normalize = { x_pos * 1.0 / bmp_width * (real_max - real_min) + real_min };
	double y_normalize = { y_pos * 1.0 / bmp_height * (imag_max - imag_min) + imag_min };

	cuDoubleComplex c = make_cuDoubleComplex(x_normalize, y_normalize);
	cuDoubleComplex zi = make_cuDoubleComplex(0,0);
	cuDoubleComplex zn = make_cuDoubleComplex(0,0);

	int value = 0;
	for (size_t i = 0; i < iteration; i++) {
		zn = cuCadd(cuCmul(zi, zi),c);
		zi = zn;
		if (cuCabs(zn) > threshold) {
			p_dst[t] = lookUp[i % iteration];
			break;
		}
	}

	if (cuCabs(zn) <= threshold)
		p_dst[t] = lookUp[0];	
}

cudaError_t call_kernel(
	dim3 const big, 
	dim3 const tib, 
	pfc::BGR_4_t * p_dst, 
	std::size_t const size, 
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
	int const image_number) {
	// blocks in grid
	// threads in block
	// 3 kernel a 512 threads

	kernel << <big, tib >> > (p_dst, size, imag_max, imag_min, real_max, real_min, threshold, iteration, bmp_width, bmp_height, amount_of_images, point_real, point_imag, zoom_factor, image_number);
	return cudaGetLastError();
}