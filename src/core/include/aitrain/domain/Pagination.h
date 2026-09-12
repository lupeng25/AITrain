#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

namespace aitrain {

struct PageRequest final {
    int pageSize = 50;
    QString after;
};

// 目录条件在 Storage 内应用，先搜索整个项目，再截取返回页。
struct CatalogFilter final {
    QString text;
    QStringList kinds;
    QString state;
};

template<class T>
struct Page final {
    QVector<T> items;
    QString nextCursor;
    bool hasMore = false;
};

} // namespace aitrain
