//           $Id: pfc_cuda_timer.h 37996 2018-10-28 15:16:00Z p20068 $
//          $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/ESD/SPS3/2015-WS/Ablauf/src/Filters/src/helpers/pfc_cuda_timer.h $
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

namespace pfc { namespace cuda {   // pfc::cuda

// -------------------------------------------------------------------------------------------------

class timer final {
   public:
      explicit timer (bool const start_immediately = false) {
         PFC_CUDA_CHECK (cudaEventCreate (&m_start));
         PFC_CUDA_CHECK (cudaEventCreate (&m_stop));

         if (start_immediately) {
            start ();
         }
      }

      timer (timer const &) = delete;   // no copy construction
//    timer (timer &&) = delete;        // no move construction

      timer (timer && tmp) noexcept {
         std::memcpy (this, &tmp, sizeof (timer)); tmp.m_moved_from = true;
      }

     ~timer () {   // must not be called after 'cudaDeviceReset'
         if (!m_moved_from) {
            stop ();

            PFC_CUDA_CHECK (cudaEventDestroy (m_stop));
            PFC_CUDA_CHECK (cudaEventDestroy (m_start));
         }
      }

      timer & operator = (timer const &) = delete;   // no copy assignment
      timer & operator = (timer &&) = delete;        // no move assignment

//    timer & operator = (timer && tmp) noexcept {
//       if (&tmp != this) {
//          PFC_CUDA_CHECK (cudaEventDestroy (m_stop));
//          PFC_CUDA_CHECK (cudaEventDestroy (m_start));

//          std::memcpy (this, &tmp, sizeof (timer)); tmp.m_moved_from = true;
//       }

//       return *this;
//    }

      auto did_run () const {
         return m_did_run;
      }

      auto get_elapsed () const {
         return get_elapsed_s ();
      }

      auto get_elapsed_ms () const {
         return static_cast <int> (m_elapsed);
      }

      double get_elapsed_s () const {
         return m_elapsed / 1000.0;
      }

      bool is_running () const {
         return m_running;
      }

      void reset () {
         stop ();

         m_did_run = false;
         m_elapsed = 0;
      }

      timer & start () {
         if (!m_running) {
            PFC_CUDA_CHECK (cudaEventRecord (m_start, 0));

            m_elapsed = 0;
            m_running = true;
         }

         return *this;
      }

      timer & stop () {
         if (m_running) {
            PFC_CUDA_CHECK (cudaEventRecord (m_stop, 0));
            PFC_CUDA_CHECK (cudaEventSynchronize (m_stop));
            PFC_CUDA_CHECK (cudaEventElapsedTime (&m_elapsed, m_start, m_stop));   // in ms (with a resolution of approximately 1/2 us)

            m_did_run = true;
            m_running = false;
         }

         return *this;   // get_elapsed_s ()
      }

   private:
      bool        m_did_run    {false};     // the timer ran at least once
      float       m_elapsed    {0};         // elapsed time [ms] (see cudaEventElapsedTime)
      bool        m_moved_from {false};     //
      bool        m_running    {false};     // the timer is currently running
      cudaEvent_t m_start      {nullptr};   //
      cudaEvent_t m_stop       {nullptr};   //
};

// -------------------------------------------------------------------------------------------------

} }   // namespace pfc::cuda
