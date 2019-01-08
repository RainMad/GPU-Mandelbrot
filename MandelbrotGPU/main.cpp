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

		int const amount_of_images = 10;

		double point_real = -0.745289981;
		double point_imag = 0.113075003;
		double const real_max = 1.25470996;
		double const real_min = -2.74529005;
		double const imag_max = 1.23807502;
		double const imag_min = -1.01192498;

		double const zoom_factor = 0.95;


		int const iterations = 30;
		int const threshold = 4;

		int const bitmap_width = 8024;
		int const bitmap_height = 4152;


		pfc::bitmap bmp{ bitmap_width, bitmap_height };
		auto & span{ bmp.pixel_span() };
		auto * const p_buffer{ std::data(span) };

		int p_buffer_size = bitmap_width * bitmap_height * sizeof(pfc::BGR_4_t);
		pfc::BGR_4_t * bmp_dst{}; cudaMalloc(&bmp_dst, p_buffer_size);


		std::unique_ptr< pfc::BGR_4_t[]>			hp_dst{ std::make_unique <pfc::BGR_4_t[]>(p_buffer_size) };
		//check(cudaMemcpy(bmp_src, p_buffer, p_buffer_size, cudaMemcpyHostToDevice));

		check(call_kernel(bitmap_width *bitmap_height / 512 + 1, 512, bmp_dst, bitmap_width *bitmap_height, imag_max, imag_min, real_max, real_min, threshold, iterations, bitmap_width, bitmap_height));

		check(cudaMemcpy(hp_dst.get(), bmp_dst, p_buffer_size, cudaMemcpyDeviceToHost));

		check(cudaFree(bmp_dst));

		pfc::bitmap bmpCp{ bitmap_width, bitmap_height/*, gsl::span{hp_dst.get(), bitmap_width *bitmap_height} */};
		/*auto & spanCp{ bmpCp.pixel_span() };
		bmpCp.data(hp_dst.get());
		spanCp = hp_dst.get()[0];*/
		//auto * const p_buffer_serial_Cp{ std::data(spanCp) };
		//bmpCp.pixel_span()[0]= hp_dst.get()[0];
		for (int i = 0; i < bitmap_width *bitmap_height; i++) {
			auto val = hp_dst[i];
			auto val1 = hp_dst.get()[i];
			pfc::byte_t r = hp_dst.get()[i].red;
			pfc::byte_t g = hp_dst.get()[i].green;
			pfc::byte_t b = hp_dst.get()[i].blue;
			bmpCp.pixel_span()[i] = {r, g, b};//hp_dst.get()[i];
			//std::cout <<  i << ": " << (int)r << ", " << (int)g << ", " << (int)b << std::endl;
		//	p_buffer_serial_Cp[i] = { (pfc::byte_t)hp_dst.get()[i].red,(pfc::byte_t)hp_dst.get()[i].green, (pfc::byte_t)hp_dst.get()[i].blue };
		}

		bmpCp.to_file("./test2.bmp");
	}

	cudaDeviceReset();
}