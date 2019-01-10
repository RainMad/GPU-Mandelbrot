#include "./kernel.h"


#include <complex>
#include <thread>
#include <cstdlib>
#include <iostream>
#include <string>
#include <memory>
#include "chrono.h"

#include "./pfc_bitmap_3.h"

using namespace std::string_literals;


void check(cudaError_t const error) {
	if (error != cudaSuccess) {
		std::cerr << cudaGetErrorString(error) << std::endl;
		std::exit(1);
	}
}

int main() {
	int count{ -1 };
	check(cudaGetDeviceCount(&count));
	if (count > 0) {
		cudaSetDevice(0);

		cudaDeviceProp prop;
		check(cudaGetDeviceProperties(&prop, 0));
		std::cout << "name: " << prop.name << '\n' << "cc: " << prop.major << " " << prop.minor << std::endl;

		int const amount_of_images = 5;

		double point_real = -0.745289981;
		double point_imag = 0.113075003;
		double const real_max = 1.25470996;
		double const real_min = -2.74529005;
		double const imag_max = 1.23807502;
		double const imag_min = -1.01192498;

		double const zoom_factor = 0.95;


		int const iterations = 30;
		int const threshold = 4;

		int const bitmap_width = 4024;
		int const bitmap_height = 2152;


		pfc::bitmap bmp{ bitmap_width, bitmap_height };
		auto & span{ bmp.pixel_span() };
		auto * const p_buffer{ std::data(span) };

		int p_buffer_size = bitmap_width * bitmap_height * sizeof(pfc::BGR_4_t)*amount_of_images;
		pfc::BGR_4_t * bmp_dst{}; cudaMalloc(&bmp_dst, p_buffer_size);


		std::unique_ptr< pfc::BGR_4_t[]>			hp_dst{ std::make_unique <pfc::BGR_4_t[]>(p_buffer_size) };

		pfc::bitmap bmpCp{ bitmap_width, bitmap_height };
		//check(cudaMemcpy(bmp_src, p_buffer, p_buffer_size, cudaMemcpyHostToDevice));
		for (int k = 0; k < amount_of_images; k++) {
			check(call_kernel(bitmap_width *bitmap_height / 32 + 1, 32, bmp_dst, bitmap_width *bitmap_height, imag_max, imag_min, real_max, real_min, threshold, iterations, bitmap_width, bitmap_height, amount_of_images, point_real, point_imag, zoom_factor, k));

			check(cudaMemcpy(hp_dst.get(), bmp_dst, p_buffer_size, cudaMemcpyDeviceToHost));

		for (int i = 0; i < bitmap_width *bitmap_height; i++) {
				auto val = hp_dst[i];
				auto val1 = hp_dst.get()[i];
				pfc::byte_t r = hp_dst.get()[i].red;
				pfc::byte_t g = hp_dst.get()[i].green;
				pfc::byte_t b = hp_dst.get()[i].blue;
				bmpCp.pixel_span()[i] = { r, g, b };//hp_dst.get()[i];
			}
			bmpCp.to_file("./test" + std::to_string(k) + ".bmp");
		}

		check(cudaFree(bmp_dst));

		//int count = 0;
		//int img_count = 0;
		//pfc::bitmap bmpCp{ bitmap_width, bitmap_height };
		//for (int i = 0; i < bitmap_width *bitmap_height*amount_of_images; i++) {
		//	auto val = hp_dst[i];
		//	auto val1 = hp_dst.get()[i];
		//	pfc::byte_t r = hp_dst.get()[i].red;
		//	pfc::byte_t g = hp_dst.get()[i].green;
		//	pfc::byte_t b = hp_dst.get()[i].blue;
		//	bmpCp.pixel_span()[count++] = { r, g, b };//hp_dst.get()[i];
		//	if ((i+1) % (bitmap_width *bitmap_height) == 0) {
		//		count = 0;
		//		bmpCp.to_file("./test" + std::to_string(img_count) + ".bmp");
		//		img_count++;
		//	}
		//}
	}

	cudaDeviceReset();
}