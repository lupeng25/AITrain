#pragma once

#include <QMap>
#include <QString>
#include <QStringList>

namespace aitrain {

struct YoloDataYaml {
    bool exists = false;
    QString rootPath;
    QString yamlPath;
    QString basePath;
    int classCount = -1;
    QStringList classNames;
    QMap<QString, QString> splitImagePaths;
};

struct YoloSplitPaths {
    QString imageDir;
    QString labelDir;
};

YoloDataYaml parseYoloDataYaml(const QString& datasetPath, QString* error = nullptr);
YoloSplitPaths yoloSplitPaths(const YoloDataYaml& layout, const QString& split);
bool writeNormalizedYoloDataYaml(
    const QString& outputPath,
    const YoloDataYaml& sourceLayout,
    bool includeTest,
    QStringList* errors = nullptr);

} // namespace aitrain
