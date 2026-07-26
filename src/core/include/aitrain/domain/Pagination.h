#pragma once

#include <QString>
#include <QVector>

namespace aitrain {

struct PageRequest final {
    int pageSize = 50;
    QString after;
};

template<class T>
struct Page final {
    QVector<T> items;
    QString nextCursor;
    bool hasMore = false;
};

} // namespace aitrain
