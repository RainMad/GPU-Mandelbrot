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

__constant__ pfc::BGR_4_t static lookUp[128] = {
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
	{217, 162, 103},
	{218, 163, 104},
	{219, 164, 105},
	{220, 165, 106},
	{221, 166, 107},
	{222, 167, 108},
	{223, 168, 109},
	{224, 169, 110},
	{225, 170, 111},
	{226, 171, 112},
	{227, 172, 113},
	{228, 173, 114},
	{229, 174, 115},
	{230, 175, 116},
	{231, 176, 117},
	{232, 177, 118},
	{233, 178, 119},
	{234, 179, 120},
	{235, 180, 121},
	{236, 181, 122},
	{237, 182, 123},
	{238, 183, 124},
	{239, 184, 125},
	{240, 185, 126},
	{241, 186, 126},
	{242, 187, 127},
	{243, 188, 128},
	{244, 189, 129},
	{245, 190, 130},
	{246, 191, 131},
	{247, 192, 132},
	{248, 193, 133},
	{249, 194, 134},
	{250, 195, 135},
	{251, 196, 136},
	{252, 197, 137},
	{253, 198, 138},
	{254, 199, 139},
	{255, 20, 140},
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
	{217, 162, 103},
	{218, 163, 104},
	{219, 164, 105},
	{220, 165, 106},
	{221, 166, 107},
	{222, 167, 108},
	{223, 168, 109},
	{224, 169, 110},
	{225, 170, 111},
	{226, 171, 112},
	{227, 172, 113},
	{228, 173, 114},
	{229, 174, 115},
	{230, 175, 116},
	{231, 176, 117},
	{232, 177, 118},
	{233, 178, 119},
	{234, 179, 120},
	{235, 180, 121},
	{236, 181, 122},
	{237, 182, 123},
	{238, 183, 124},
	{239, 184, 125},
	{240, 185, 126},
	{241, 186, 126},
	{242, 187, 127},
	{243, 188, 128},
	{244, 189, 129},
	{245, 190, 130},
	{246, 191, 131},
	{247, 192, 132},
	{248, 193, 133},
	{249, 194, 134},
	{250, 195, 135},
	{251, 196, 136},
	{252, 197, 137},
	{253, 198, 138},
};


__device__ double pow(double x, double y);

// divergenten code vermeiden! -> eine der größten Bremsen
__global__ void kernel(pfc::BGR_4_t * const p_dst, 
	std::size_t const size_x, 
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
	auto const u{ blockIdx.y * blockDim.y + threadIdx.y };

	if (t > bmp_width || u > bmp_height)
		return;
	int image_size = bmp_width * bmp_height;

	int x_pos = t;
	int y_pos = u;

	int image_n = t / (image_size);
	double pow_result = pow(zoom_factor, image_number + image_n);
	real_min = point_real - (point_real - real_min) * pow_result;
	real_max = point_real + (real_max - point_real) * pow_result;
	imag_max = point_imag + (imag_max - point_imag) * pow_result;
	imag_min = point_imag - (point_imag - imag_min) * pow_result;

	double x_normalize = { x_pos * 1.0 / bmp_width * (real_max - real_min) + real_min };
	double y_normalize = { y_pos * 1.0 / bmp_height * (imag_max - imag_min) + imag_min };

	pfc::complex<float> c{ x_normalize, y_normalize };
	pfc::complex<float> zi{ 0.0,0.0 };
	pfc::complex<float> zn{ 0.0,0.0 };
	int value = 0;
	for (size_t i = 0; i < iteration; i++) {
		zn = zi * zi + c;
		zi = zn;
		if (norm(zn) > threshold) {
			p_dst[u*bmp_width + t] = lookUp[i % iteration];
			break;
		}
	}

	if (norm(zn) <= threshold)
		p_dst[u*bmp_width + t] = lookUp[0];
}

// divergenten code vermeiden! -> eine der größten Bremsen
__global__ void kernel1(pfc::BGR_4_t * const p_dst,
	std::size_t const size_x,
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
	//blockDim Anzahl der Threads pro block
	auto const t{ blockIdx.x * blockDim.x + threadIdx.x }; // -> absolute Threadnumber
	auto const u{ blockIdx.y * blockDim.y + threadIdx.y };

	if (t > bmp_width || u > bmp_height)
		return;
	int image_size = bmp_width * bmp_height;

	int x_pos = t;
	int y_pos = u;

	int image_n = t / (image_size);
	double pow_result = pow(zoom_factor, image_number + image_n);
	real_min = point_real - (point_real - real_min) * pow_result;
	real_max = point_real + (real_max - point_real) * pow_result;
	imag_max = point_imag + (imag_max - point_imag) * pow_result;
	imag_min = point_imag - (point_imag - imag_min) * pow_result;

	double x_normalize = { x_pos * 1.0 / bmp_width * (real_max - real_min) + real_min };
	double y_normalize = { y_pos * 1.0 / bmp_height * (imag_max - imag_min) + imag_min };


	double p = sqrt((x_normalize - 0.25)*(x_normalize - 0.25) + y_normalize * y_normalize);
	if (x_normalize <= (p - 2*p*p + 0.25) || ((x_normalize + 1)*(x_normalize + 1) + y_normalize <= 1.0/16) <= 0.25*y_normalize*y_normalize) {
		p_dst[u*bmp_width + t] = lookUp[0];
		return;
	}

	pfc::complex<float> c{ x_normalize, y_normalize };
	pfc::complex<float> zi{ 0.0,0.0 };
	pfc::complex<float> zn{ 0.0,0.0 };
	int value = 0;
	for (size_t i = 0; i < iteration; i++) {
		zn = zi * zi + c;
		zi = zn;
		if (norm(zn) > threshold) {
			p_dst[u*bmp_width + t] = lookUp[i % iteration];
			break;
			return;
		}
	}

	if (norm(zn) <= threshold)
		p_dst[u*bmp_width + t] = lookUp[0];
}

cudaError_t call_kernel(
	dim3 const big, 
	dim3 const tib, 
	pfc::BGR_4_t * p_dst, 
	std::size_t const size_x, 
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

	//kernel << <big, tib >> > (p_dst, size_x, imag_max, imag_min, real_max, real_min, threshold, iteration, bmp_width, bmp_height, amount_of_images, point_real, point_imag, zoom_factor, image_number);
	return cudaGetLastError();
}

cudaError_t call_kernel_1(
	dim3 const big,
	dim3 const tib,
	pfc::BGR_4_t * p_dst,
	std::size_t const size_x,
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

	kernel1 << <big, tib >> > (p_dst, size_x, imag_max, imag_min, real_max, real_min, threshold, iteration, bmp_width, bmp_height, amount_of_images, point_real, point_imag, zoom_factor, image_number);
	return cudaGetLastError();
}
