//           $Id: pfc_cuda_memory.h 37996 2018-10-28 15:16:00Z p20068 $
//          $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/ESD/SPS3/2015-WS/Ablauf/src/Filters/src/helpers/pfc_cuda_memory.h $
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
#include "./pfc_traits.h"

#include <memory>
#include <type_traits>

// -------------------------------------------------------------------------------------------------

#undef  PFC_CUDA_FREE
#define PFC_CUDA_FREE(dp_memory) \
   pfc::cuda::free (dp_memory, __FILE__, __LINE__)

#undef  PFC_CUDA_MALLOC
#define PFC_CUDA_MALLOC(value_t, size) \
   pfc::cuda::malloc <value_t> (size, __FILE__, __LINE__)

#undef  PFC_CUDA_MALLOC_MANAGED
#define PFC_CUDA_MALLOC_MANAGED(value_t, size) \
   pfc::cuda::malloc_managed <value_t> (size, __FILE__, __LINE__)

#undef  PFC_CUDA_MEMCPY
#define PFC_CUDA_MEMCPY(p_dst, p_src, size, kind) \
   pfc::cuda::memcpy (p_dst, p_src, size, kind, __FILE__, __LINE__)

// -------------------------------------------------------------------------------------------------

namespace pfc { namespace cuda {   // pfc::cuda

template <typename value_t> value_t * & free (value_t * & dp_memory, std::string const & file = {}, int const line = {}) {
   if (dp_memory != nullptr) {
      pfc::cuda::check (cudaFree (dp_memory), file, line); dp_memory = nullptr;
   }

   return dp_memory;
}

template <typename value_t> value_t * free (value_t * && dp_memory, std::string const & file = {}, int const line = {}) {
   return pfc::cuda::free (dp_memory, file, line);
}

template <typename value_t, typename size_t> value_t * malloc (size_t const size, std::string const & file = {}, int const line = {}) {
   static_assert (pfc::is_integral_v <size_t>, "size_t must be an integral type");

   value_t * dp_memory {nullptr};

   if (size > size_t {}) {
      pfc::cuda::check (cudaMalloc (&dp_memory, size * sizeof (value_t)), file, line);
   }

   return dp_memory;
}

template <typename value_t, typename size_t> value_t * malloc_managed (size_t const size, std::string const & file = {}, int const line = {}) {
   static_assert (pfc::is_integral_v <size_t>, "size_t must be an integral type");

   value_t * dp_memory {nullptr};

   if (size > size_t {}) {
      pfc::cuda::check (cudaMallocManaged (&dp_memory, size * sizeof (value_t)), file, line);
   }

   return dp_memory;
}

template <typename value_t, typename size_t> value_t * memcpy (value_t * const p_dst, value_t const * const p_src, size_t const size, cudaMemcpyKind const kind, std::string const & file = {}, int const line = {}) {
   static_assert (pfc::is_integral_v <size_t>, "size_t must be an integral type");

   if ((p_dst != nullptr) && (p_src != nullptr) && (size > size_t {})) {
      pfc::cuda::check (cudaMemcpy (p_dst, p_src, size * sizeof (value_t), kind), file, line);
   }

   return p_dst;
}

template <typename value_t, typename size_t> value_t * memcpy (value_t * const p_dst, std::unique_ptr <value_t> const & p_src, size_t const size, cudaMemcpyKind const kind, std::string const & file = {}, int const line = {}) {
   return pfc::cuda::memcpy (p_dst, p_src.get (), size, kind, file, line);
}

} }   // namespace pfc::cuda
