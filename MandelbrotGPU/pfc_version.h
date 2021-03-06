//           $Id: pfc_version.h 37996 2018-10-28 15:16:00Z p20068 $
//          $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/ESD/SPS3/2015-WS/Ablauf/src/Filters/src/host/pfc_version.h $
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

#include "./Helpers/pfc_macros.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace pfc {

	// -------------------------------------------------------------------------------------------------

	class version final {
	public:
		version(int const n = 0, int const min = 0, int const max = 0) {
			if ((n < min) || (n > max) || (min > max)) {
				throw std::runtime_error("Wrong version number");
			}

			m_max = max;
			m_min = min;
			m_version = n;

			m_names.resize(m_max - m_min + 1);
		}

		int const & as_int() const {
			return m_version;
		}

		std::string const & as_string() const {
			return m_names[m_version];
		}

		bool is(int const n) const {
			return m_version == n;
		}

		bool is_not(int const n) const {
			return m_version != n;
		}

		version & register_name(int const n, std::string const & name) {
			if ((n < m_min) || (n > m_max)) {
				throw std::runtime_error("Wrong version number");
			}

			m_names[n - m_min] = name; return *this;
		}

	private:
		int                       m_max = 0;
		int                       m_min = 0;
		std::vector <std::string> m_names = {};
		int                       m_version = 0;
	};

	// -------------------------------------------------------------------------------------------------

}   // namespace pfc
