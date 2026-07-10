#pragma once

#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QVector>

namespace aitrain {

struct DatasetValidationResult {
    struct Issue {
        QString severity;
        QString code;
        QString filePath;
        int line = 0;
        QString message;

        QJsonObject toJson() const;
    };

    bool ok = true;
    int sampleCount = 0;
    QStringList errors;
    QStringList warnings;
    QStringList previewSamples;
    QVector<Issue> issues;

    QJsonObject toJson() const;
};

} // namespace aitrain
