#include "./kernel.h"



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
	{255, 200, 140},
	{255, 201, 141},
	{255, 202, 142},
	{255, 203, 143},
	{255, 204, 144},
	{255, 205, 145},
	{255, 206, 146},
	{255, 207, 147},
	{255, 208, 148},
	{255, 209, 149},
	{255, 210, 150},
	{255, 212, 151},
	{255, 213, 152},
	{255, 214, 153},
	{255, 215, 154},
	{255, 216, 155},
	{255, 217, 156},
	{255, 218, 157},
	{255, 219, 158},
	{255, 220, 159},
	{255, 221, 160},
	{255, 222, 161},
	{255, 223, 162},
	{255, 224, 163},
	{255, 225, 164},
	{255, 226, 165},
	{255, 227, 166},
	{255, 228, 167},
	{255, 229, 168},
	{255, 230, 169},
	{255, 231, 170},
	{255, 232, 171},
	{255, 233, 172},
	{255, 234, 173},
	{255, 235, 174},
	{255, 236, 175},
	{255, 237, 176 },
	{255, 238, 177 },
	{255, 239, 178 },
	{255, 240, 179 },
	{ 255, 241, 180 },
	{ 255, 242, 181 },
	{ 255, 243, 182 },
	{ 255, 244, 183 },
	{ 255, 245, 184 },
	{ 255, 246, 185 },
	{ 255, 247, 186 },
	{ 255, 248, 187 },
	{ 255, 249, 188 },
	{ 255, 250, 189 },
	{ 255, 251, 190 },
	{ 255, 252, 191 },
	{ 255, 253, 192 },
	{ 255, 254, 193 },
	{ 255, 255, 194 },
	{ 255, 255, 195 },
	{ 255, 255, 196 },
	{ 255, 255, 197 },
	{ 255, 255, 198 },
	{ 255, 255, 199 },
	{ 255, 255, 200 },
	{ 255, 255, 201 },
	{ 255, 255, 202 },
};


__device__ double pow(double x, double y);

__device__ double log(double x);
__device__ double exp(double x);


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

__global__ void kernel3(pfc::BGR_4_t * const p_dst,
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



__global__ void kernel4(pfc::BGR_4_t * const p_dst,
	float imag_max,
	float imag_min,
	float real_max,
	float real_min,
	int const threshold,
	int const iteration,
	int const bmp_width,
	int const bmp_height) {
	// blockDim Anzahl der Threads pro block
	auto const x_pos{ blockIdx.x * blockDim.x + threadIdx.x }; // -> absolute Threadnumber
	auto const y_pos{ blockIdx.y * blockDim.y + threadIdx.y };

	if (x_pos > bmp_width || y_pos > bmp_height)
		return;

	float c_real{ x_pos * 1.0 / bmp_width * (real_max - real_min) + real_min };
	float c_imag{ y_pos * 1.0 / bmp_height * (imag_max - imag_min) + imag_min };

	float zi_real{ 0 };
	float zi_imag{ 0 };

	float zn_real{ 0 };
	float zn_imag{ 0 };

	int index = y_pos * bmp_width + x_pos;

	p_dst[index] = lookUp[0];
	for (size_t i = 0; i < iteration; i++) {
		zn_real = zi_real * zi_real - zi_imag * zi_imag + c_real;
		zn_imag = 2 * zi_real * zi_imag + c_imag;

		zi_real = zn_real;
		zi_imag = zn_imag;

		if ((zn_real * zn_real + zn_imag * zn_imag) > threshold) {
			p_dst[index] = lookUp[i];
			break;
		}
	}
}

__global__ void kernel5(pfc::BGR_4_t * const p_dst,
	float imag_max,
	float imag_min,
	float real_max,
	float real_min,
	int const threshold,
	int const iteration,
	int const bmp_width,
	int const bmp_height) {
	// blockDim Anzahl der Threads pro block
	auto const x_pos{ blockIdx.x * blockDim.x + threadIdx.x }; // -> absolute Threadnumber
	auto const y_pos{ blockIdx.y * blockDim.y + threadIdx.y };

	if (x_pos > bmp_width || y_pos > bmp_height)
		return;

	float c_real{ x_pos * 1.0 / bmp_width * (real_max - real_min) + real_min };
	float c_imag{ y_pos * 1.0 / bmp_height * (imag_max - imag_min) + imag_min };

	float zi_real{ 0 };
	float zi_imag{ 0 };

	float zn_real{ 0 };
	float zn_imag{ 0 };

	int index = y_pos * bmp_width + x_pos;

	p_dst[index] = lookUp[0];

	for (size_t i = 0; i < iteration; i++) {
		// TODO check
		zn_real = zi_real * zi_real - zi_imag * zi_imag + c_real;
		zn_imag = 2 * zi_real * zi_imag + c_imag;

		zi_real = zn_real;
		zi_imag = zn_imag;

		if ((zn_real * zn_real + zn_imag * zn_imag) > threshold) {
			p_dst[index] = lookUp[i];
			break;
		}
	}
}

__global__ void kernel6(pfc::BGR_4_t * const p_dst,
	float imag_max,
	float imag_min,
	float real_max,
	float real_min,
	int const threshold,
	int const iteration,
	int const bmp_width,
	int const bmp_height) {
	// blockDim Anzahl der Threads pro block
	auto const x_pos{ blockIdx.x * blockDim.x + threadIdx.x }; // -> absolute Threadnumber
	auto const y_pos{ blockIdx.y * blockDim.y + threadIdx.y };

	if (x_pos > bmp_width || y_pos > bmp_height)
		return;

	float c_real{ x_pos * 1.0 / bmp_width * (real_max - real_min) + real_min };
	float c_imag{ y_pos * 1.0 / bmp_height * (imag_max - imag_min) + imag_min };

	float zi_real{ 0 };
	float zi_imag{ 0 };

	float zn_real{ 0 };
	float zn_imag{ 0 };

	int index = y_pos * bmp_width + x_pos;

	p_dst[index] = lookUp[0];

	// bulb checking
	float helper1 = c_real - 0.25;
	float helper2 = c_real + 1;
	float c_imag_c_imag = c_imag * c_imag;
	float p = sqrt(helper1*helper1 + c_imag_c_imag);
	if (c_real <= p - 2 * p*p + 0.25 || helper2 * helper2 + c_imag_c_imag <= 0.0625) {
		p_dst[index] = lookUp[0];
	}
	else {
		for (size_t i = 0; i < iteration; i++) {
			// TODO check
			zn_real = zi_real * zi_real - zi_imag * zi_imag + c_real;
			zn_imag = 2 * zi_real * zi_imag + c_imag;

			zi_real = zn_real;
			zi_imag = zn_imag;

			if ((zn_real * zn_real + zn_imag * zn_imag) > threshold) {
				p_dst[index] = lookUp[i];
				break;
			}
		}
	}
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

cudaError_t call_kernel_3(
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
	cudaFuncSetCacheConfig(kernel3, cudaFuncCachePreferL1);
	kernel3 << <big, tib >> > (p_dst, size_x, imag_max, imag_min, real_max, real_min, threshold, iteration, bmp_width, bmp_height, amount_of_images, point_real, point_imag, zoom_factor, image_number);
	return cudaGetLastError();
}


cudaError_t call_kernel_4(
	dim3 const big,
	dim3 const tib,
	pfc::BGR_4_t * p_dst,
	float imag_max,
	float imag_min,
	float real_max,
	float real_min,
	int const threshold,
	int const iteration,
	int const bmp_width,
	int const bmp_height) {
	// blocks in grid
	// threads in block
	// 3 kernel a 512 threads
	cudaFuncSetCacheConfig(kernel4, cudaFuncCachePreferL1);
	kernel4 << < big, tib >> > (p_dst, imag_max, imag_min, real_max, real_min, threshold, iteration, bmp_width, bmp_height);
	return cudaGetLastError();
}

cudaError_t call_kernel_5(
	dim3 const big,
	dim3 const tib,
	pfc::BGR_4_t * p_dst,
	float imag_max,
	float imag_min,
	float real_max,
	float real_min,
	int const threshold,
	int const iteration,
	int const bmp_width,
	int const bmp_height,
	cudaStream_t s1) {
	// blocks in grid
	// threads in block
	// 3 kernel a 512 threads
	cudaFuncSetCacheConfig(kernel5, cudaFuncCachePreferL1);
	kernel5 << < big, tib, 0, s1 >> > (p_dst, imag_max, imag_min, real_max, real_min, threshold, iteration, bmp_width, bmp_height);
	cudaDeviceSynchronize();
	return cudaGetLastError();
}

cudaError_t call_kernel_6(
	dim3 const big,
	dim3 const tib,
	pfc::BGR_4_t * p_dst,
	float imag_max,
	float imag_min,
	float real_max,
	float real_min,
	int const threshold,
	int const iteration,
	int const bmp_width,
	int const bmp_height,
	cudaStream_t s1,
	cudaStream_t s2) {
	// blocks in grid
	// threads in block
	// 3 kernel a 512 threads
	cudaFuncSetCacheConfig(kernel6, cudaFuncCachePreferL1);
	kernel6 << < big, tib, 0, s1 >> > (p_dst, imag_max, imag_min, real_max, real_min, threshold, iteration, bmp_width, bmp_height);
	cudaDeviceSynchronize();
	return cudaGetLastError();
}
