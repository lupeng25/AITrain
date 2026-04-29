#include "aitrain/core/PluginManager.h"

#include <QCoreApplication>
#include <QDir>
#include <QPluginLoader>

namespace aitrain {

QJsonObject PluginManifest::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("id"), id);
    object.insert(QStringLiteral("name"), name);
    object.insert(QStringLiteral("version"), version);
    object.insert(QStringLiteral("description"), description);
    object.insert(QStringLiteral("taskTypes"), QJsonArray::fromStringList(taskTypes));
    object.insert(QStringLiteral("datasetFormats"), QJsonArray::fromStringList(datasetFormats));
    object.insert(QStringLiteral("exportFormats"), QJsonArray::fromStringList(exportFormats));
    object.insert(QStringLiteral("parameterSchema"), parameterSchema);
    object.insert(QStringLiteral("requiresGpu"), requiresGpu);
    return object;
}

PluginManifest PluginManifest::fromJson(const QJsonObject& object)
{
    auto readStringList = [](const QJsonArray& array) {
        QStringList values;
        for (const QJsonValue& value : array) {
            values.append(value.toString());
        }
        return values;
    };

    PluginManifest manifest;
    manifest.id = object.value(QStringLiteral("id")).toString();
    manifest.name = object.value(QStringLiteral("name")).toString();
    manifest.version = object.value(QStringLiteral("version")).toString();
    manifest.description = object.value(QStringLiteral("description")).toString();
    manifest.taskTypes = readStringList(object.value(QStringLiteral("taskTypes")).toArray());
    manifest.datasetFormats = readStringList(object.value(QStringLiteral("datasetFormats")).toArray());
    manifest.exportFormats = readStringList(object.value(QStringLiteral("exportFormats")).toArray());
    manifest.parameterSchema = object.value(QStringLiteral("parameterSchema")).toArray();
    manifest.requiresGpu = object.value(QStringLiteral("requiresGpu")).toBool();
    return manifest;
}

QJsonObject DatasetValidationResult::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("ok"), ok);
    object.insert(QStringLiteral("sampleCount"), sampleCount);
    object.insert(QStringLiteral("errors"), QJsonArray::fromStringList(errors));
    object.insert(QStringLiteral("warnings"), QJsonArray::fromStringList(warnings));
    return object;
}

PluginManager::PluginManager(QObject* parent)
    : QObject(parent)
{
}

PluginManager::~PluginManager() = default;

void PluginManager::scan(const QStringList& directories)
{
    loaders_.clear();
    plugins_.clear();
    errors_.clear();

    QStringList suffixes;
#if defined(Q_OS_WIN)
    suffixes << QStringLiteral("*.dll");
#elif defined(Q_OS_MAC)
    suffixes << QStringLiteral("*.dylib") << QStringLiteral("*.so");
#else
    suffixes << QStringLiteral("*.so");
#endif

    for (const QString& directoryPath : directories) {
        QDir directory(directoryPath);
        if (!directory.exists()) {
            continue;
        }

        const QFileInfoList files = directory.entryInfoList(suffixes, QDir::Files);
        for (const QFileInfo& file : files) {
            auto loader = QSharedPointer<QPluginLoader>::create(file.absoluteFilePath());
            QObject* instance = loader->instance();
            if (!instance) {
                errors_.append(file.fileName() + QStringLiteral(": ") + loader->errorString());
                continue;
            }

            auto* plugin = qobject_cast<IModelPlugin*>(instance);
            if (!plugin) {
                errors_.append(file.fileName() + QStringLiteral(": not an AITrain model plugin"));
                loader->unload();
                continue;
            }

            plugins_.append(plugin);
            loaders_.append(loader);
        }
    }
}

QVector<IModelPlugin*> PluginManager::plugins() const
{
    return plugins_;
}

IModelPlugin* PluginManager::pluginById(const QString& id) const
{
    for (IModelPlugin* plugin : plugins_) {
        if (plugin && plugin->manifest().id == id) {
            return plugin;
        }
    }
    return nullptr;
}

QStringList PluginManager::errors() const
{
    return errors_;
}

} // namespace aitrain

