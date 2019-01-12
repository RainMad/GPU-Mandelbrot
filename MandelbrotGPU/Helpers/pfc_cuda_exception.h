//           $Id: pfc_cuda_exception.h 37996 2018-10-28 15:16:00Z p20068 $
//          $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/ESD/SPS3/2015-WS/Ablauf/src/Filters/src/helpers/pfc_cuda_exception.h $
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

#include "./pfc_macros.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <string>

using namespace std::literals;

// -------------------------------------------------------------------------------------------------

#undef  PFC_CUDA_CHECK
#define PFC_CUDA_CHECK(error) \
   pfc::cuda::check (error, __FILE__, __LINE__)

#undef  PFC_CUDA_MAKE_ERROR_MESSAGE
#define PFC_CUDA_MAKE_ERROR_MESSAGE(error) \
   pfc::cuda::make_error_message (error, __FILE__, __LINE__)

// -------------------------------------------------------------------------------------------------

namespace pfc { namespace cuda {   // pfc::cuda

inline auto make_error_message (cudaError_t const error, std::string const & file = {}, int const line = {}) {
   auto message {"CUDA error #"s};

   message += std::to_string (error);
   message += " '";
   message += cudaGetErrorString (error);
   message += "' occurred";

   if (!file.empty () && (line > 0)) {
      message += " in file '";
      message += file;
      message += "' on line ";
      message += std::to_string (line);
   }

   return message;
}

// -------------------------------------------------------------------------------------------------

class exception final : public std::runtime_error {
   using inherited = std::runtime_error;

   public:
      explicit exception (cudaError_t const error, std::string const & file, int const line)
         : inherited {make_error_message (error, file, line)} {
      }
};

// -------------------------------------------------------------------------------------------------

inline void check (cudaError_t const error, std::string const & file = {}, int const line = {}) {
   if (error != cudaSuccess) {
      throw pfc::cuda::exception (error, file, line);
   }
}

// -------------------------------------------------------------------------------------------------

} }   // namespace pfc::cuda
