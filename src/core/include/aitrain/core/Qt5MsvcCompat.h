#pragma once

#if defined(_MSC_VER) && _MSC_VER >= 1950
#ifndef _SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING
#define _SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING
#endif

#include <cstddef>

namespace stdext {
template <typename Iterator>
constexpr Iterator make_checked_array_iterator(Iterator iterator, std::size_t) noexcept
{
    return iterator;
}

template <typename Iterator>
constexpr Iterator make_unchecked_array_iterator(Iterator iterator) noexcept
{
    return iterator;
}
}
#endif
