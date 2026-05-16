#pragma once

#if defined(_MSC_VER) && _MSC_VER >= 1950
#include <cstddef>

namespace stdext {

template <typename Iterator>
Iterator make_checked_array_iterator(Iterator iterator, std::size_t)
{
    return iterator;
}

template <typename Iterator>
Iterator make_unchecked_array_iterator(Iterator iterator)
{
    return iterator;
}

} // namespace stdext
#endif
