#include "YoloDatasetLayout.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QTextStream>

namespace aitrain {
namespace {

QString stripYamlComment(QString line)
{
    const int commentIndex = line.indexOf(QLatin1Char('#'));
    if (commentIndex >= 0) {
        line = line.left(commentIndex);
    }
    return line;
}

QString unquoteYamlScalar(QString value)
{
    value = value.trimmed();
    if ((value.startsWith(QLatin1Char('\'')) && value.endsWith(QLatin1Char('\'')))
        || (value.startsWith(QLatin1Char('"')) && value.endsWith(QLatin1Char('"')))) {
        value = value.mid(1, value.size() - 2);
    }
    return value.trimmed();
}

QString yamlScalar(const QString& value)
{
    QString escaped = value;
    escaped.replace(QLatin1Char('\\'), QStringLiteral("\\\\"));
    escaped.replace(QLatin1Char('"'), QStringLiteral("\\\""));
    return QStringLiteral("\"%1\"").arg(escaped);
}

QString resolveYoloPath(const QString& basePath, const QString& path)
{
    const QString trimmed = unquoteYamlScalar(path);
    if (trimmed.isEmpty()) {
        return QDir::cleanPath(basePath);
    }
    if (QDir::isAbsolutePath(trimmed)) {
        return QDir::cleanPath(trimmed);
    }
    return QDir::cleanPath(QDir(basePath).filePath(trimmed));
}

QStringList parseInlineNames(QString value)
{
    value = value.trimmed();
    if (!value.startsWith(QLatin1Char('[')) || !value.endsWith(QLatin1Char(']'))) {
        return {};
    }

    QStringList names;
    const QString inner = value.mid(1, value.size() - 2);
    for (QString name : inner.split(QLatin1Char(','),
#if QT_VERSION < QT_VERSION_CHECK(5, 15, 0)
             QString::SkipEmptyParts
#else
             Qt::SkipEmptyParts
#endif
             )) {
        name = unquoteYamlScalar(name);
        if (!name.isEmpty()) {
            names.append(name);
        }
    }
    return names;
}

QString labelPathForImagePath(const QString& imagePath, const QString& split)
{
    QString normalized = imagePath;
    normalized.replace(QLatin1Char('\\'), QLatin1Char('/'));
    QStringList parts = normalized.split(QLatin1Char('/'),
#if QT_VERSION < QT_VERSION_CHECK(5, 15, 0)
        QString::KeepEmptyParts
#else
        Qt::KeepEmptyParts
#endif
    );
    for (int index = 0; index < parts.size(); ++index) {
        if (parts.at(index) == QStringLiteral("images")) {
            parts[index] = QStringLiteral("labels");
            return parts.join(QLatin1Char('/'));
        }
    }
    return QStringLiteral("labels/%1").arg(split);
}

QString scalarValueFromLine(const QString& line, const QString& key)
{
    const QString trimmed = line.trimmed();
    const QRegularExpression pattern(QStringLiteral("^%1\\s*:\\s*(.*)$").arg(QRegularExpression::escape(key)));
    const QRegularExpressionMatch match = pattern.match(trimmed);
    if (!match.hasMatch()) {
        return {};
    }
    QString value = match.captured(1).trimmed();
    if (value.startsWith(QLatin1Char('[')) || value.startsWith(QLatin1Char('{'))) {
        return {};
    }
    return unquoteYamlScalar(value);
}

bool yamlKeyValue(const QString& line, const QString& key, QString* value)
{
    const QString trimmed = line.trimmed();
    const QRegularExpression pattern(QStringLiteral("^%1\\s*:\\s*(.*)$").arg(QRegularExpression::escape(key)));
    const QRegularExpressionMatch match = pattern.match(trimmed);
    if (!match.hasMatch()) {
        return false;
    }
    if (value) {
        *value = match.captured(1).trimmed();
    }
    return true;
}

} // namespace

YoloDataYaml parseYoloDataYaml(const QString& datasetPath, QString* error)
{
    YoloDataYaml layout;
    layout.rootPath = QDir::cleanPath(datasetPath);
    layout.basePath = layout.rootPath;
    layout.yamlPath = QDir(layout.rootPath).filePath(QStringLiteral("data.yaml"));
    if (error) {
        error->clear();
    }

    QFile file(layout.yamlPath);
    if (!file.exists()) {
        return layout;
    }
    layout.exists = true;
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) {
            *error = QStringLiteral("Cannot open data.yaml: %1").arg(layout.yamlPath);
        }
        return layout;
    }

    QMap<int, QString> indexedNames;
    QStringList listNames;
    bool inNamesBlock = false;
    int lineNumber = 0;
    while (!file.atEnd()) {
        ++lineNumber;
        const QString rawLine = QString::fromUtf8(file.readLine());
        const QString uncommented = stripYamlComment(rawLine);
        const QString trimmed = uncommented.trimmed();
        if (trimmed.isEmpty()) {
            continue;
        }

        if (inNamesBlock) {
            const bool indented = uncommented.startsWith(QLatin1Char(' '))
                || uncommented.startsWith(QLatin1Char('\t'));
            if (indented) {
                if (trimmed.startsWith(QStringLiteral("-"))) {
                    const QString name = unquoteYamlScalar(trimmed.mid(1));
                    if (!name.isEmpty()) {
                        listNames.append(name);
                    }
                    continue;
                }

                const QRegularExpressionMatch itemMatch =
                    QRegularExpression(QStringLiteral("^(\\d+)\\s*:\\s*(.+)$")).match(trimmed);
                if (itemMatch.hasMatch()) {
                    bool indexOk = false;
                    const int index = itemMatch.captured(1).toInt(&indexOk);
                    const QString name = unquoteYamlScalar(itemMatch.captured(2));
                    if (indexOk && !name.isEmpty()) {
                        indexedNames.insert(index, name);
                    }
                    continue;
                }
            }
            inNamesBlock = false;
        }

        QString yamlValue;
        if (yamlKeyValue(trimmed, QStringLiteral("nc"), &yamlValue)) {
            bool ok = false;
            const int classCount = yamlValue.toInt(&ok);
            if (!ok || classCount <= 0) {
                if (error) {
                    *error = QStringLiteral("data.yaml:%1 nc must be a positive integer").arg(lineNumber);
                }
                layout.classCount = -1;
            } else {
                layout.classCount = classCount;
            }
            continue;
        }

        if (yamlKeyValue(trimmed, QStringLiteral("names"), &yamlValue)) {
            if (yamlValue.isEmpty()) {
                inNamesBlock = true;
            } else {
                layout.classNames = parseInlineNames(yamlValue);
            }
            continue;
        }

        const QString yamlPath = scalarValueFromLine(uncommented, QStringLiteral("path"));
        if (!yamlPath.isEmpty()) {
            layout.basePath = resolveYoloPath(layout.rootPath, yamlPath);
            continue;
        }
        for (const QString& split : {QStringLiteral("train"), QStringLiteral("val"), QStringLiteral("test")}) {
            const QString splitPath = scalarValueFromLine(uncommented, split);
            if (!splitPath.isEmpty()) {
                layout.splitImagePaths.insert(split, splitPath);
            }
        }
    }

    if (!indexedNames.isEmpty()) {
        layout.classNames.clear();
        for (auto it = indexedNames.cbegin(); it != indexedNames.cend(); ++it) {
            layout.classNames.append(it.value());
        }
    } else if (!listNames.isEmpty()) {
        layout.classNames = listNames;
    }

    if (layout.classCount < 0 && !layout.classNames.isEmpty()) {
        layout.classCount = layout.classNames.size();
    }
    if (layout.classCount > 0 && !layout.classNames.isEmpty() && layout.classNames.size() != layout.classCount) {
        if (error && error->isEmpty()) {
            *error = QStringLiteral("data.yaml names count does not match nc");
        }
    }

    return layout;
}

YoloSplitPaths yoloSplitPaths(const YoloDataYaml& layout, const QString& split)
{
    const QString imagePath = layout.splitImagePaths.value(split, QStringLiteral("images/%1").arg(split));
    const QString labelPath = labelPathForImagePath(imagePath, split);

    YoloSplitPaths paths;
    paths.imageDir = resolveYoloPath(layout.basePath, imagePath);
    paths.labelDir = resolveYoloPath(layout.basePath, labelPath);
    return paths;
}

bool writeNormalizedYoloDataYaml(
    const QString& outputPath,
    const YoloDataYaml& sourceLayout,
    bool includeTest,
    QStringList* errors)
{
    QStringList classNames = sourceLayout.classNames;
    int classCount = sourceLayout.classCount;
    if (classCount <= 0 && !classNames.isEmpty()) {
        classCount = classNames.size();
    }
    if (classCount <= 0) {
        classCount = 1;
    }
    while (classNames.size() < classCount) {
        classNames.append(QStringLiteral("class_%1").arg(classNames.size()));
    }
    if (classNames.size() > classCount) {
        classNames = classNames.mid(0, classCount);
    }

    const QString dataYamlPath = QDir(outputPath).filePath(QStringLiteral("data.yaml"));
    QDir().mkpath(QFileInfo(dataYamlPath).absolutePath());
    QFile file(dataYamlPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        if (errors) {
            errors->append(QStringLiteral("Cannot write data.yaml: %1").arg(dataYamlPath));
        }
        return false;
    }

    QTextStream stream(&file);
    stream.setCodec("UTF-8");
    stream << QStringLiteral("path: .\n");
    stream << QStringLiteral("train: images/train\n");
    stream << QStringLiteral("val: images/val\n");
    if (includeTest) {
        stream << QStringLiteral("test: images/test\n");
    }
    stream << QStringLiteral("nc: ") << classNames.size() << QLatin1Char('\n');
    stream << QStringLiteral("names:\n");
    for (int index = 0; index < classNames.size(); ++index) {
        stream << QStringLiteral("  ") << index << QStringLiteral(": ") << yamlScalar(classNames.at(index)) << QLatin1Char('\n');
    }
    return true;
}

} // namespace aitrain
