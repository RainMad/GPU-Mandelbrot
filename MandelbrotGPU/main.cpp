#include "./kernel.h"


#include <complex>
#include <thread>
#include <cstdlib>
#include <iostream>
#include <string>
#include <memory>
#include "chrono.h"

#include "./Helpers/pfc_bitmap_3.h"
#include "./Helpers/pfc_cuda_timer.h"

#include <iomanip>

using namespace std::string_literals;


void check(cudaError_t const error) {
	if (error != cudaSuccess) {
		std::cerr << cudaGetErrorString(error) << std::endl;
		std::exit(1);
	}
}

inline std::string ms_to_string(pfc::cuda::timer const & timer) {
	if (timer.is_running()) {
		return "timer is running";

	}
	else if (!timer.did_run()) {
		return "timer did never run";
	}

	return std::to_string(timer.get_elapsed_ms()) + " ms";
}

inline std::ostream & print_time(std::string header, pfc::cuda::timer const & time, int const width = 7, std::ostream & out = std::cout) {
	auto const state = out.rdstate();

	out << header
		<< std::setw(width) << std::setfill(' ') << std::right << ms_to_string(time);

	out.setstate(state); return out;
}

int main() {
	int count{ -1 };
	check(cudaGetDeviceCount(&count));
	if (count > 0) {
		cudaSetDevice(0);

		cudaDeviceProp prop;
		check(cudaGetDeviceProperties(&prop, 0));
		std::cout << "name: " << prop.name << '\n' << "cc: " << prop.major << " " << prop.minor << std::endl;

		int const amount_of_images =500;

		double point_real = -0.745289981;
		double point_imag = 0.113075003;
		double const real_max = 1.25470996;
		double const real_min = -2.74529005;
		double const imag_max = 1.23807502;
		double const imag_min = -1.01192498;

		double const zoom_factor = 0.95;

		int const iterations = 50;
		int const threshold = 4;

		int const bitmap_width = 2048;
		int const bitmap_height = 1024;

		dim3 threads_per_block(16, 8);
		dim3 num_blocks(bitmap_width / threads_per_block.x,
			bitmap_height / threads_per_block.y);

		int const amount_of_images_processed_at_the_same_time = 1;
		bool const print_images = false;


		pfc::bitmap bmp{ bitmap_width, bitmap_height };
		auto & span{ bmp.pixel_span() };
		auto * const p_buffer{ std::data(span) };

		int p_buffer_size = bitmap_width * bitmap_height * sizeof(pfc::BGR_4_t) * amount_of_images_processed_at_the_same_time;
		pfc::BGR_4_t * bmp_dst{}; cudaMalloc(&bmp_dst, p_buffer_size);

		std::unique_ptr< pfc::BGR_4_t[]>			hp_dst{ std::make_unique <pfc::BGR_4_t[]>(p_buffer_size) };

		pfc::bitmap bmpCp{ bitmap_width, bitmap_height };

		dim3 gridSize(bitmap_height, amount_of_images);

		int count = 0;
		pfc::cuda::timer timer(true);
		for (int k = 0; k < amount_of_images / amount_of_images_processed_at_the_same_time; k++) {
			check(call_kernel(num_blocks, threads_per_block, bmp_dst, bitmap_width * bitmap_height*amount_of_images, imag_max, imag_min, real_max, real_min, threshold, iterations, bitmap_width, bitmap_height, amount_of_images, point_real, point_imag, zoom_factor, k));

			check(cudaMemcpy(hp_dst.get(), bmp_dst, p_buffer_size, cudaMemcpyDeviceToHost));

			if (!print_images)
				continue;

			for (int i = 0; i < bitmap_width * bitmap_height * amount_of_images_processed_at_the_same_time; i++) {
				bmpCp.pixel_span()[count++] = { hp_dst.get()[i] };
				if ((i + 1) % (bitmap_width * bitmap_height) == 0) {
					int pos = k + i / (bitmap_width * bitmap_height);
					bmpCp.to_file("./images/test" + std::to_string(k*amount_of_images_processed_at_the_same_time +i/ (bitmap_width * bitmap_height)) + ".bmp");
					count = 0;
				}
			}
		}
		timer.stop();

		std::cout << "Amount of images: " << amount_of_images << std::endl;
		std::cout << "Size of one image - Width: " << bitmap_width << " / Height: " << bitmap_height << std::endl;
		std::cout << "Iterations: " << iterations << std::endl;
		std::cout << "Threshold: " << threshold << std::endl;

		print_time("Mandelbrot GPU:   ", timer) << '\n';

		check(cudaFree(bmp_dst));
	}

	cudaDeviceReset();
}