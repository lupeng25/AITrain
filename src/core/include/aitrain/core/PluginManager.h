#pragma once

#include "aitrain/core/PluginInterfaces.h"

#include <QObject>
#include <QSharedPointer>
#include <QString>
#include <QStringList>
#include <QVector>

class QPluginLoader;

namespace aitrain {

class PluginManager : public QObject {
    Q_OBJECT

public:
    explicit PluginManager(QObject* parent = nullptr);
    ~PluginManager() override;

    void scan(const QStringList& directories);
    QVector<IModelPlugin*> plugins() const;
    IModelPlugin* pluginById(const QString& id) const;
    QStringList errors() const;

private:
    QVector<QSharedPointer<QPluginLoader>> loaders_;
    QVector<IModelPlugin*> plugins_;
    QStringList errors_;
};

} // namespace aitrain

