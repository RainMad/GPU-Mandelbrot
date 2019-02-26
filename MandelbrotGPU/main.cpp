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

#include "./pfc_config.h"

#include <iomanip>

using namespace std::string_literals;


void check(cudaError_t const error) {
	if (error != cudaSuccess) {
		std::cerr << cudaGetErrorString(error) << std::endl;
		std::exit(1);
	}
}

inline std::string ms_to_string(pfc::cuda::timer const & timer, int const runs = 1) {
	if (timer.is_running()) {
		return "timer is running";

	}
	else if (!timer.did_run()) {
		return "timer did never run";
	}

	return std::to_string(timer.get_elapsed_ms()/runs) + " ms";
}

inline std::ostream & print_time(std::string header, pfc::cuda::timer const & time, int const runs, int const width = 7, std::ostream & out = std::cout) {
	auto const state = out.rdstate();

	out << header
		<< std::setw(width) << std::setfill(' ') << std::right << ms_to_string(time,runs);

	out.setstate(state); return out;
}

void print_images(pfc::bitmap const & bmpCp, std::unique_ptr< pfc::BGR_4_t[]> const & hp_dst, int const k) {
	int count = 0;
	for (int i = 0; i < pfc::config::bitmap_width * pfc::config::bitmap_height; i++) {
		bmpCp.pixel_span()[count++] = { hp_dst.get()[i] };
		if ((i + 1) % (pfc::config::bitmap_width * pfc::config::bitmap_height) == 0) {
			bmpCp.to_file("images/test" + std::to_string(k + i / (pfc::config::bitmap_width * pfc::config::bitmap_height)) + ".bmp");
			count = 0;
		}
	}
}

void print_images(pfc::bitmap const & bmpCp, pfc::BGR_4_t * hp_dst, int const k) {
	int count = 0;
	for (int i = 0; i < pfc::config::bitmap_width * pfc::config::bitmap_height; i++) {
		bmpCp.pixel_span()[count++] = { hp_dst[i] };
		if ((i + 1) % (pfc::config::bitmap_width * pfc::config::bitmap_height) == 0) {
			bmpCp.to_file("images/test" + std::to_string(k + i / (pfc::config::bitmap_width * pfc::config::bitmap_height)) + ".bmp");
			count = 0;
		}
	}
}

pfc::cuda::timer cuda_wrapper_version1(
	pfc::BGR_4_t * bmp_dst,
	std::unique_ptr< pfc::BGR_4_t[]> &hp_dst,
	int const p_buffer_size,
	pfc::bitmap  &bmpCp,
	dim3 const threads_per_block,
	dim3 const num_blocks,
	int const runs) 
{
	pfc::cuda::timer timer(true);
	for (size_t i = 0; i < runs; i++)
	{
		for (int k = 0; k < pfc::config::amount_of_images; k++) {
			check(call_kernel_1(num_blocks,
				threads_per_block,
				bmp_dst,
				pfc::config::bitmap_width * pfc::config::bitmap_height * pfc::config::amount_of_images,
				pfc::config::imag_max,
				pfc::config::imag_min,
				pfc::config::real_max,
				pfc::config::real_min,
				pfc::config::threshold,
				pfc::config::iterations,
				pfc::config::bitmap_width,
				pfc::config::bitmap_height,
				pfc::config::amount_of_images,
				pfc::config::point_real,
				pfc::config::point_imag,
				pfc::config::zoom_factor, k));

			if (pfc::config::print_images && runs < 2) {
				check(cudaMemcpy(hp_dst.get(), bmp_dst, p_buffer_size, cudaMemcpyDeviceToHost));
				print_images(bmpCp, hp_dst, k);
			}
		}
	}

	return std::move(timer.stop());
}

pfc::cuda::timer cuda_wrapper_version2(
	pfc::BGR_4_t * bmp_dst,
	std::unique_ptr< pfc::BGR_4_t[]> &hp_dst,
	int const p_buffer_size,
	pfc::bitmap  &bmpCp,
	dim3 const threads_per_block,
	dim3 const num_blocks,
	int const runs)
{
	pfc::cuda::timer timer(true);
	for (size_t i = 0; i < runs; i++)
	{
		for (int k = 0; k < pfc::config::amount_of_images; k++) {
			check(call_kernel_2(num_blocks,
				threads_per_block,
				bmp_dst,
				pfc::config::bitmap_width * pfc::config::bitmap_height * pfc::config::amount_of_images,
				pfc::config::imag_max,
				pfc::config::imag_min,
				pfc::config::real_max,
				pfc::config::real_min,
				pfc::config::threshold,
				pfc::config::iterations,
				pfc::config::bitmap_width,
				pfc::config::bitmap_height,
				pfc::config::amount_of_images,
				pfc::config::point_real,
				pfc::config::point_imag,
				pfc::config::zoom_factor, k));

			if (pfc::config::print_images && runs < 2) {
				cudaMemcpy(hp_dst.get(), bmp_dst, p_buffer_size, cudaMemcpyDeviceToHost);
				print_images(bmpCp, hp_dst, k);
			}
		}
	}

	return std::move(timer.stop());
}

pfc::cuda::timer cuda_wrapper_version3(
	pfc::BGR_4_t * bmp_dst,
	std::unique_ptr< pfc::BGR_4_t[]> &hp_dst,
	int const p_buffer_size,
	pfc::bitmap  &bmpCp,
	dim3 const threads_per_block,
	dim3 const num_blocks,
	int const runs)
{
	pfc::cuda::timer timer(true);
	for (size_t i = 0; i < runs; i++)
	{
		for (int k = 0; k < pfc::config::amount_of_images; k++) {
			check(call_kernel_3(num_blocks,
				threads_per_block,
				bmp_dst,
				pfc::config::bitmap_width * pfc::config::bitmap_height * pfc::config::amount_of_images,
				pfc::config::imag_max,
				pfc::config::imag_min,
				pfc::config::real_max,
				pfc::config::real_min,
				pfc::config::threshold,
				pfc::config::iterations,
				pfc::config::bitmap_width,
				pfc::config::bitmap_height,
				pfc::config::amount_of_images,
				pfc::config::point_real,
				pfc::config::point_imag,
				pfc::config::zoom_factor, k));

			cudaMemcpy(hp_dst.get(), bmp_dst, p_buffer_size, cudaMemcpyDeviceToHost);

			if (pfc::config::print_images && runs < 2) {
				
				print_images(bmpCp, hp_dst, k);
			}
		}
	}

	return std::move(timer.stop());
}

pfc::cuda::timer cuda_wrapper_version4(
	pfc::BGR_4_t * bmp_dst,
	std::unique_ptr< pfc::BGR_4_t[]> &hp_dst,
	int const p_buffer_size,
	pfc::bitmap  &bmpCp,
	dim3 const threads_per_block,
	dim3 const num_blocks,
	int const runs)
{
	float helper_real_min = pfc::config::point_real - pfc::config::real_min;
	float helper_real_max = pfc::config::real_max - pfc::config::point_real;
	float helper_imag_max = pfc::config::imag_max - pfc::config::point_imag;
	float helper_imag_min = pfc::config::point_imag - pfc::config::imag_min;


	pfc::cuda::timer timer(true);
	for (size_t i = 0; i < runs; i++)
	{
		for (int k = 0; k < pfc::config::amount_of_images; k++) {
			float zoomFactor = pow(pfc::config::zoom_factor, k);
			double real_min = pfc::config::point_real - helper_real_min * zoomFactor;
			double real_max = pfc::config::point_real + helper_real_max * zoomFactor;
			double imag_max = pfc::config::point_imag + helper_imag_max * zoomFactor;
			double imag_min = pfc::config::point_imag - helper_imag_min * zoomFactor;
			check(call_kernel_4(num_blocks,
				threads_per_block,
				bmp_dst,
				imag_max,
				imag_min,
				real_max,
				real_min,
				pfc::config::threshold,
				pfc::config::iterations,
				pfc::config::bitmap_width,
				pfc::config::bitmap_height));

			cudaMemcpy(hp_dst.get(), bmp_dst, p_buffer_size, cudaMemcpyDeviceToHost);

			if (pfc::config::print_images && runs < 2) {
				print_images(bmpCp, hp_dst, k);
			}
		}
	}

	return std::move(timer.stop());
}


pfc::cuda::timer cuda_wrapper_version5(
	pfc::BGR_4_t * bmp_dst,
	pfc::BGR_4_t * hp_dst,
	int const p_buffer_size,
	pfc::bitmap  &bmpCp,
	dim3 const threads_per_block,
	dim3 const num_blocks,
	int const runs)
{
	float helper_real_min = pfc::config::point_real - pfc::config::real_min;
	float helper_real_max = pfc::config::real_max - pfc::config::point_real;
	float helper_imag_max = pfc::config::imag_max - pfc::config::point_imag;
	float helper_imag_min = pfc::config::point_imag - pfc::config::imag_min;


	pfc::cuda::timer timer(true);
	for (size_t i = 0; i < runs; i++)
	{
		for (int k = 0; k < pfc::config::amount_of_images; k++) {
			float zoomFactor = pow(pfc::config::zoom_factor, k);
			double real_min = pfc::config::point_real - helper_real_min * zoomFactor;
			double real_max = pfc::config::point_real + helper_real_max * zoomFactor;
			double imag_max = pfc::config::point_imag + helper_imag_max * zoomFactor;
			double imag_min = pfc::config::point_imag - helper_imag_min * zoomFactor;
			check(call_kernel_4(num_blocks,
				threads_per_block,
				bmp_dst,
				imag_max,
				imag_min,
				real_max,
				real_min,
				pfc::config::threshold,
				pfc::config::iterations,
				pfc::config::bitmap_width,
				pfc::config::bitmap_height));

			cudaMemcpy(hp_dst, bmp_dst, p_buffer_size, cudaMemcpyDeviceToHost);

			if (pfc::config::print_images && runs < 2) {
				print_images(bmpCp, hp_dst, k);
			}
		}
	}

	return std::move(timer.stop());
}

pfc::cuda::timer cuda_wrapper_version6(
	pfc::BGR_4_t * bmp_dst,
	pfc::BGR_4_t * hp_dst,
	int const p_buffer_size,
	pfc::bitmap  &bmpCp,
	dim3 const threads_per_block,
	dim3 const num_blocks,
	int const runs)
{
	float helper_real_min = pfc::config::point_real - pfc::config::real_min;
	float helper_real_max = pfc::config::real_max - pfc::config::point_real;
	float helper_imag_max = pfc::config::imag_max - pfc::config::point_imag;
	float helper_imag_min = pfc::config::point_imag - pfc::config::imag_min;

	cudaStream_t s1;
	cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
	cudaStream_t s2;
	cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
	pfc::cuda::timer timer(true);
	for (size_t i = 0; i < runs; i++)
	{
		for (int k = 0; k < pfc::config::amount_of_images; k++) {
			float zoomFactor = pow(pfc::config::zoom_factor, k);
			double real_min = pfc::config::point_real - helper_real_min * zoomFactor;
			double real_max = pfc::config::point_real + helper_real_max * zoomFactor;
			double imag_max = pfc::config::point_imag + helper_imag_max * zoomFactor;
			double imag_min = pfc::config::point_imag - helper_imag_min * zoomFactor;
			check(call_kernel_5(num_blocks,
				threads_per_block,
				bmp_dst,
				imag_max,
				imag_min,
				real_max,
				real_min,
				pfc::config::threshold,
				pfc::config::iterations,
				pfc::config::bitmap_width,
				pfc::config::bitmap_height,
				s1, s2));

			cudaMemcpyAsync(hp_dst, bmp_dst, p_buffer_size, cudaMemcpyDeviceToHost, s2);

			if (pfc::config::print_images && runs < 2) {
				print_images(bmpCp, hp_dst, k);
			}
		}
	}

	return std::move(timer.stop());
}


int main(int argc, char * argv[]) {
	if ((argc >= 2) && (argv != nullptr)) {
		pfc::config::code_version(std::atoi(argv[1]));
	}
	std::cout
		<< std::boolalpha << pfc::config::app_title() << '\n'
		<< std::string(std::string(pfc::config::app_title()).size(), '-') << "\n"
		"Code Version: " << pfc::config::code_version().as_int() << "\n"
		"Description:  " << pfc::config::code_version().as_string() << "\n";

	int count{ -1 };
	check(cudaGetDeviceCount(&count));
	if (count > 0) {
		cudaSetDevice(0);

		cudaDeviceProp prop;
		check(cudaGetDeviceProperties(&prop, 0));
		std::cout << "name: " << prop.name << '\n' << "cc: " << prop.major << " " << prop.minor << std::endl;

		dim3 threads_per_block = pfc::config::block_size_fractal();
		dim3 num_blocks(pfc::config::bitmap_width / threads_per_block.x,
			pfc::config::bitmap_height / threads_per_block.y);

		pfc::bitmap bmp{ pfc::config::bitmap_width, pfc::config::bitmap_height };


		auto & span{ bmp.pixel_span() };
		auto * const p_buffer{ std::data(span) };

		//int p_buffer_size = pfc::config::bitmap_width * pfc::config::bitmap_height * sizeof(span);

		int p_buffer_size = pfc::config::bitmap_width * pfc::config::bitmap_height * sizeof(pfc::BGR_4_t);
		
		 
		//cudaHostAlloc((void**)p_buffer, p_buffer_size, cudaHostAllocDefault);


		pfc::BGR_4_t * bmp_dst{}; cudaMalloc(&bmp_dst, p_buffer_size);
		pfc::BGR_4_t * hp_dst_pinned{}; cudaHostAlloc(&hp_dst_pinned, p_buffer_size, cudaHostAllocDefault);

		std::unique_ptr< pfc::BGR_4_t[]>			hp_dst{ std::make_unique <pfc::BGR_4_t[]>(p_buffer_size) };
		

		pfc::bitmap bmpCp{ pfc::config::bitmap_width, pfc::config::bitmap_height };

		dim3 gridSize(pfc::config::bitmap_height, pfc::config::amount_of_images);

		int count = 0;
		int const runs = 1;

		switch (pfc::config::code_version().as_int()) {
			case 0: print_time("Mandelbrot GPU:   ", cuda_wrapper_version1(bmp_dst, hp_dst, p_buffer_size, bmpCp, threads_per_block, num_blocks, runs), runs) << "\n"; break;
			case 1: print_time("Mandelbrot GPU - bulb checking:   ", cuda_wrapper_version2(bmp_dst, hp_dst, p_buffer_size, bmpCp, threads_per_block, num_blocks, runs), runs) << "\n"; break;
			case 2: print_time("Mandelbrot GPU - block size 32x2:   ", cuda_wrapper_version1(bmp_dst, hp_dst, p_buffer_size, bmpCp, threads_per_block, num_blocks, runs), runs) << "\n"; break;
			case 3: print_time("Mandelbrot GPU - block size 32x4:   ", cuda_wrapper_version1(bmp_dst, hp_dst, p_buffer_size, bmpCp, threads_per_block, num_blocks, runs), runs) << "\n"; break;
			case 4: print_time("Mandelbrot GPU - block size 32x8:   ", cuda_wrapper_version1(bmp_dst, hp_dst, p_buffer_size, bmpCp, threads_per_block, num_blocks, runs), runs) << "\n"; break;
			case 5: print_time("Ma ndelbrot GPU - Prefer Cach L1:   ", cuda_wrapper_version3(bmp_dst, hp_dst, p_buffer_size, bmpCp, threads_per_block, num_blocks, runs), runs) << "\n"; break;
			case 6: print_time("Mandelbrot GPU - Optimizing implementation:   ", cuda_wrapper_version4(bmp_dst, hp_dst, p_buffer_size, bmpCp, threads_per_block, num_blocks, runs), runs) << "\n"; break;
			case 7: print_time("Mandelbrot GPU - Using Pinned memory:   ", cuda_wrapper_version5(bmp_dst, hp_dst_pinned, p_buffer_size, bmpCp, threads_per_block, num_blocks, runs), runs) << "\n"; break;
			case 8: print_time("Mandelbrot GPU - Using Streams:   ", cuda_wrapper_version6(bmp_dst, hp_dst_pinned, p_buffer_size, bmpCp, threads_per_block, num_blocks, runs), runs) << "\n"; break;
		}
			
		std::cout << "Amount of images: " << pfc::config::amount_of_images << std::endl;
		std::cout << "Size of one image - Width: " << pfc::config::bitmap_width << " / Height: " << pfc::config::bitmap_height << std::endl;
		std::cout << "Iterations: " << pfc::config::iterations << std::endl;
		std::cout << "Threshold: " << pfc::config::threshold << std::endl;
		
		check(cudaFree(bmp_dst));
		check(cudaFreeHost(hp_dst_pinned));
	}

	cudaDeviceReset();
	return 0;
}