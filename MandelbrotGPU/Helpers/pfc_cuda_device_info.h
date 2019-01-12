//           $Id: pfc_cuda_device_info.h 37996 2018-10-28 15:16:00Z p20068 $
//          $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/ESD/SPS3/2015-WS/Ablauf/src/Filters/src/helpers/pfc_cuda_device_info.h $
//     $Revision: 37996 $
//         $Date: 2018-10-28 16:16:00 +0100 (So., 28 Okt 2018) $
//       $Author: p20068 $
//
//       Creator: Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
// Creation Date:
//     Copyright: (c) 2018 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//
//       License: This document contains proprietary information belonging to
//                University of Applied Sciences Upper Austria, Campus
//                Hagenberg. It is distributed under the Boost Software License,
//                Version 1.0 (see http://www.boost.org/LICENSE_1_0.txt).
//
//    Annotation: This file is part of the code snippets handed out during one
//                of my HPC lessons held at the University of Applied Sciences
//                Upper Austria, Campus Hagenberg.

#pragma once

#include "./pfc_cuda_exception.h"

#include <algorithm>
#include <map>
#include <tuple>

namespace pfc { namespace cuda {   // pfc::cuda

// -------------------------------------------------------------------------------------------------

/**
 * Stringizes a CUDA-runtime-version number.
 */
inline auto cudart_version_to_string (int const version = CUDART_VERSION) noexcept {
   return std::to_string (version / 1000) + '.' + std::to_string (version % 100 / 10);
}

inline auto const & get_device_props (int const device = 0) noexcept {
   static cudaDeviceProp props cudaDevicePropDontCare; PFC_CUDA_CHECK (cudaGetDeviceProperties (&props, device)); return props;
}

inline bool can_map_host_memory (int const device = 0) {
   int value {0}; PFC_CUDA_CHECK (cudaDeviceGetAttribute (&value, cudaDevAttrCanMapHostMemory, device)); return value != 0;
}

inline bool have_managed_memory (int const device = 0) {
   int value {0}; PFC_CUDA_CHECK (cudaDeviceGetAttribute (&value, cudaDevAttrManagedMemory, device)); return value != 0;
}

inline bool activate_map_host_memory (int const device = 0) {
   auto const ok {pfc::cuda::can_map_host_memory (device)};

   if (ok) {
      PFC_CUDA_CHECK (cudaSetDevice (device));
      PFC_CUDA_CHECK (cudaSetDeviceFlags (cudaDeviceMapHost));
   }

   return ok;
}

// -------------------------------------------------------------------------------------------------

struct device_info final {
   int          cc_major              {0};    //  0
   int          cc_minor              {0};    //  1
   int          cores_sm              {0};    //  2
   char const * uarch                 {""};   //  4
   char const * chip                  {""};   //  5
   int          ipc                   {0};    //  6
   int          max_act_cores_sm      {0};    //  7
   int          max_regs_thread       {0};    //  8
   int          max_regs_block        {0};    //  9
   int          max_smem_block        {0};    // 10
   int          max_threads_block     {0};    // 11
   int          max_act_blocks_sm     {0};    // 12
   int          max_threads_sm        {0};    // 13
   int          max_warps_sm          {0};    // 14
   int          alloc_gran_regs       {0};    // 15
   int          regs_sm               {0};    // 16 (32-bit registers)
   int          alloc_gran_smem       {0};    // 17
   int          smem_bank_width       {0};    // 18
   int          smem_sm               {0};    // 19 (in bytes)
   int          smem_banks            {0};    // 20
   char const * sm_version            {""};   // 21
   int          warp_size             {0};    // 22 (in threads)
   int          alloc_gran_warps      {0};    // 23
   int          schedulers_sm         {0};    // 24
   int          width_cl1             {0};    // 25
   int          width_cl2             {0};    // 26
   int          load_store_units_sm   {0};    // 27
   int          load_store_throughput {0};    // 28 (per cycle)
   int          texture_units_sm      {0};    // 29
   int          texture_throughput    {0};    // 30 (per cycle)
   int          fp32_units_sm         {0};    // 31
   int          fp32_throughput       {0};    // 32 (per cycle)
   int          sf_units_sm           {0};    // 33 (special function unit, e.g. sin, cosine, square root)
   int          sfu_throughput        {0};    // 34 (per cycle)
};

/**
 * see <http://en.wikipedia.org/wiki/CUDA>
 * see <http://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities>
 * see <https://devblogs.nvidia.com/parallelforall/inside-volta>
 */
inline auto const & get_device_info (int const cc_major, int const cc_minor) {
   static std::map <std::tuple <int, int>, pfc::cuda::device_info> const info {
//              0  1    2  4          5            6   7    8        9       10    11  12    13  14   15        16   17  18         19  20  21       22  23  24   25  26  27  28  29  30   31   32  33  34
      {{0, 0}, {0, 0,   0, "",        "",          0,  0,   0,       0,       0,    0,  0,    0,  0,   0,        0,   0,  0,         0,  0, "",       0,  0,  0,   0,  0,  0,  0,  0,  0,   0,   0,  0,  0}},
      {{1, 0}, {1, 0,   8, "Tesla",   "G80",       1,  1,  -1,      -1,      -1,   -1,  8,   -1, -1,  -1,       -1,  -1, -1,        -1, 16, "sm_10", -1, -1,  1, 128, 32, -1, -1, -1, -1,   2,  -1, -1, -1}},
      {{1, 1}, {1, 1,   8, "Tesla",   "G8x",       1,  1,  -1,      -1,      -1,   -1,  8,   -1, -1,  -1,       -1,  -1, -1,        -1, 16, "sm_11", -1, -1,  1, 128, 32, -1, -1, -1, -1,   2,  -1, -1, -1}},
      {{1, 2}, {1, 2,   8, "Tesla",   "G9x",       1,  1,  -1,      -1,      -1,   -1,  8,   -1, -1,  -1,       -1,  -1, -1,        -1, 16, "sm_12", -1, -1,  1, 128, 32, -1, -1, -1, -1,   2,  -1, -1, -1}},
      {{1, 3}, {1, 3,   8, "Tesla",   "GT20x",     1,  1,  -1,      -1,      -1,   -1,  8,   -1, -1,  -1,       -1,  -1, -1,        -1, 16, "sm_13", -1, -1,  1, 128, 32, -1, -1, -1, -1,   2,  -1, -1, -1}},
      {{2, 0}, {2, 0,  32, "Fermi",   "GF10x",     1, 16,  63, 32*1024, 48*1024, 1024,  8, 1536, 48,  64,  32*1024, 128,  4, 48 * 1024, 32, "sm_20", 32,  2,  2, 128, 32, 16, 16,  4,  4,  32,  64,  4,  8}},
      {{2, 1}, {2, 1,  48, "Fermi",   "GF10x",     2, 16,  63, 32*1024, 48*1024, 1024,  8, 1536, 48,  64,  32*1024, 128,  4, 48 * 1024, 32, "sm_21", 32,  2,  2, 128, 32, 16, 16,  4,  4,  32,  64,  4,  8}},
      {{3, 0}, {3, 0, 192, "Kepler",  "GK10x",     2, 16,  63, 64*1024, 48*1024, 1024, 16, 2048, 64, 256,  64*1024, 256,  4, 48 * 1024, 32, "sm_30", 32,  4,  4, 128, 32, 32, 32, 16, 16, 192, 192, 32, 32}},
      {{3, 2}, {3, 2, 192, "Kepler",  "Tegra K1",  2, 16, 255, 64*1024, 48*1024, 1024, 16, 2048, 64, 256,  64*1024, 256,  4, 48 * 1024, 32, "sm_32", 32,  4,  4, 128, 32, 32, 32, 16, 16, 192, 192, 32, 32}},
      {{3, 5}, {3, 5, 192, "Kepler",  "GK11x",     2, 32, 255, 64*1024, 48*1024, 1024, 16, 2048, 64, 256,  64*1024, 256,  4, 48 * 1024, 32, "sm_35", 32,  4,  4, 128, 32, 32, 32, 16, 16, 192, 192, 32, 32}},
      {{3, 7}, {3, 7, 192, "Kepler",  "GK21x",    -1, -1, 255, 64*1024, 48*1024, 1024, 16, 2048, 64, 256, 128*1024, 256, -1, 96 * 1024, 32, "sm_37", 32,  4, -1,  -1, -1, 32, 32, 16, 16, 192, 192, 32, 32}},
      {{5, 0}, {5, 0, 128, "Maxwell", "GM10x",     2, 32, 255, 32*1024, 48*1024, 1024, 32, 2048, 64, 256,  64*1024, 256,  4, 64 * 1024, 32, "sm_50", 32,  4,  4, 128, 32, -1, -1, -1, -1, 128,  -1, -1, -1}},
      {{5, 2}, {5, 2, 128, "Maxwell", "GM20x",     2, 32, 255, 32*1024, 48*1024, 1024, 32, 2048, 64, 256,  64*1024, 256,  4, 96 * 1024, 32, "sm_52", 32,  4,  4, 128, 32, -1, -1, -1, -1, 128,  -1, -1, -1}},
      {{5, 3}, {5, 3, 256, "Maxwell", "Tegra X1",  2, 32, 255, 32*1024, 48*1024, 1024, 32, 2048, 64, 256,  64*1024, 256,  4, 64 * 1024, 32, "sm_53", 32,  4,  4, 128, 32, -1, -1, -1, -1, 128,  -1, -1, -1}},
      {{6, 0}, {6, 0,  64, "Pascal",  "GP10x",     0,  0, 255, 64*1024,       0, 1024, 32, 2048, 64,   0,  64*1024,   0,  0, 64 * 1024,  0, "sm_60", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0}},
      {{6, 1}, {6, 1, 128, "Pascal",  "GP10x",     0,  0, 255, 64*1024,       0, 1024, 32, 2048, 64,   0,  64*1024,   0,  0, 64 * 1024,  0, "sm_61", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0}},
      {{6, 2}, {6, 2, 128, "Pascal",  "GP10x",     0,  0, 255, 64*1024,       0, 1024, 32, 2048, 64,   0,  64*1024,   0,  0, 64 * 1024,  0, "sm_62", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0}},
      {{7, 0}, {7, 0,   0, "Volta",   "GV10x",     0,  0, 255, 64*1024,       0, 1024, 32, 2048, 64,   0,  64*1024,   0,  0, 96 * 1024,  0, "sm_70", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0}},
      {{7, 1}, {7, 1,   0, "Volta",   "GV10x",     0,  0, 255, 64*1024,       0, 1024, 32, 2048, 64,   0,  64*1024,   0,  0, 96 * 1024,  0, "sm_71", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0}}
   };

   return info.at ({cc_major, cc_minor});
}

inline auto const & get_device_info (cudaDeviceProp const & props) {
   return pfc::cuda::get_device_info (props.major, props.minor);
}

inline auto const & get_device_info (int const device = 0) {
   return pfc::cuda::get_device_info (pfc::cuda::get_device_props (device));
}

// -------------------------------------------------------------------------------------------------

} }   // namespace pfc::cuda
