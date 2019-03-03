#pragma once
//#include "./pfc_status.h"
#include "./pfc_version.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#undef  PFC_CONFIG
#define PFC_CONFIG \
   pfc::config::instance ()

namespace pfc {

	class config final {
	public:


		static char const * app_title() {
			return "Fractals Mandelbrot";
		}

		static char const * img_filename() {
			return "./images/fractal{0}.bmp";
		}

		static version const & code_version(int const n = 0) {
			static bool    init(false);
			static version ver(n, 0, 8);

			if (!init) {
				ver.register_name(0, "initiale Version");
				ver.register_name(1, "Block size 32x2");
				ver.register_name(2, "Block size 64x2");
				ver.register_name(3, "Block size 32x8");
				ver.register_name(4, "Prefer L1 cache");
				ver.register_name(5, "Optimizing implementation");
				ver.register_name(6, "Using Pinned Memory");
				ver.register_name(7, "Using Streams");
				ver.register_name(8, "Adding bulb checking");

				init = true;
			}

			return ver;
		}

		static dim3 block_size_fractal() {
			switch (code_version().as_int()) {
			case  0: return { 8,  8 };
			case  1: return { 32,  2 };
			case  2: return { 64,  2 };
			case  3: return { 32,  8 };
			case  4: return { 32,  4 };
			case  5: return { 32,  4 };
			case  6: return { 32,  4 };
			case  7: return { 32,  4 };
			case  8: return { 32,  4 };
			};
		}


		config(config const &) = delete;   // no copy construction
		config(config &&) = delete;   // no move construction

		config & operator = (config const &) = delete;   // no copy assignment
		config & operator = (config &&) = delete;   // no move assignment

		static int const amount_of_images = 200; // 200;

		static float constexpr point_real = -0.745289981;
		static float constexpr point_imag = 0.113075003;
		static float constexpr real_max = 1.25470996;
		static float constexpr real_min = -2.74529005;
		static float constexpr imag_max = 1.23807502;
		static float constexpr  imag_min = -1.01192498;

		static float constexpr zoom_factor = 0.95;

		static int const iterations = 127;
		static int const threshold = 4;

		static int const bitmap_width = 8192;
		static int const bitmap_height = 4608;

		static bool const print_images = true;

	private:
		config() {   // singleton
		}
	};
}