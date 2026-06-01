#include "DetectionTrainerInternal.h"

#include <QtMath>

namespace aitrain {
namespace detection_detail {

double sigmoid(double value)
{
    return 1.0 / (1.0 + qExp(-value));
}

QJsonObject boxObject(const DetectionBox& box)
{
    QJsonObject object;
    object.insert(QStringLiteral("classId"), box.classId);
    object.insert(QStringLiteral("xCenter"), box.xCenter);
    object.insert(QStringLiteral("yCenter"), box.yCenter);
    object.insert(QStringLiteral("width"), box.width);
    object.insert(QStringLiteral("height"), box.height);
    return object;
}

double boxIou(const DetectionBox& left, const DetectionBox& right)
{
    const double leftX1 = left.xCenter - left.width / 2.0;
    const double leftY1 = left.yCenter - left.height / 2.0;
    const double leftX2 = left.xCenter + left.width / 2.0;
    const double leftY2 = left.yCenter + left.height / 2.0;

    const double rightX1 = right.xCenter - right.width / 2.0;
    const double rightY1 = right.yCenter - right.height / 2.0;
    const double rightX2 = right.xCenter + right.width / 2.0;
    const double rightY2 = right.yCenter + right.height / 2.0;

    const double intersectionWidth = qMax(0.0, qMin(leftX2, rightX2) - qMax(leftX1, rightX1));
    const double intersectionHeight = qMax(0.0, qMin(leftY2, rightY2) - qMax(leftY1, rightY1));
    const double intersection = intersectionWidth * intersectionHeight;
    const double leftArea = qMax(0.0, left.width) * qMax(0.0, left.height);
    const double rightArea = qMax(0.0, right.width) * qMax(0.0, right.height);
    const double unionArea = leftArea + rightArea - intersection;
    if (unionArea <= 0.0) {
        return 0.0;
    }
    return intersection / unionArea;
}

QStringList stringListFromArray(const QJsonArray& array)
{
    QStringList values;
    for (const QJsonValue& value : array) {
        values.append(value.toString());
    }
    return values;
}

} // namespace detection_detail

QJsonObject detectionTrainingBackendStatus()
{
    QJsonObject status;
    status.insert(QStringLiteral("status"), QStringLiteral("official_only"));
    status.insert(QStringLiteral("nativeTrainingAvailable"), false);
    status.insert(QStringLiteral("message"),
        QStringLiteral("Production detection training uses the official Ultralytics Worker adapter. Legacy C++ diagnostic training has been removed."));
    status.insert(QStringLiteral("productionBackends"), QJsonArray{
        QStringLiteral("ultralytics_yolo_detect"),
        QStringLiteral("ultralytics_yolo_segment")});
    return status;
}

} // namespace aitrain
