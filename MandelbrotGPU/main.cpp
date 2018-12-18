#include "./kernel.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <memory>


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

		auto const text{ "Hello world"s };
		auto const size{ std::size(text) + 1 };

		auto const * const hp_src{ std::data(text) };
		// make_unique -> erzeugt einen smart pointer -> kein memory leak
		auto				hp_dst{ std::make_unique <char[]>(size) };

		char * dp_src{}; cudaMalloc(&dp_src, size * sizeof(char));
		char * dp_dst{}; cudaMalloc(&dp_dst, size * sizeof(char));

		check(cudaMemcpy(dp_src, hp_src, size * sizeof(char), cudaMemcpyHostToDevice));

		// kernel
		check(call_kernel(1, 512, dp_dst, dp_src, size));

		check(cudaMemcpy(hp_dst.get(), dp_dst, size * sizeof(char), cudaMemcpyDeviceToHost));

		check(cudaFree(dp_src));
		check(cudaFree(dp_dst));

		std::cout << "copy: [" << hp_dst << "]" << std::endl;

	}

	cudaDeviceReset();
}