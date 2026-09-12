#pragma once

#include <QCoreApplication>
#include <QString>

namespace aitrain_app {
// 仅翻译界面文案；身份、用户数据和后端原始输出不经过此入口。
inline QString workbenchText(const QString& source)
{
    const auto utf8 = source.toUtf8();
    return QCoreApplication::translate("Workbench", utf8.constData());
}
}
