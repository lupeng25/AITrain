#include "DetectionTrainerInternal.h"

#include "aitrain/core/Deployment.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QPainter>
#include <QProcess>
#include <QQueue>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QtEndian>
#include <QtMath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
namespace aitrain {

using namespace detection_detail;

QJsonObject detectionPredictionToJson(const DetectionPrediction& prediction)
{
    QJsonObject box;
    box.insert(QStringLiteral("classId"), prediction.box.classId);
    box.insert(QStringLiteral("xCenter"), prediction.box.xCenter);
    box.insert(QStringLiteral("yCenter"), prediction.box.yCenter);
    box.insert(QStringLiteral("width"), prediction.box.width);
    box.insert(QStringLiteral("height"), prediction.box.height);

    QJsonObject object;
    object.insert(QStringLiteral("classId"), prediction.box.classId);
    object.insert(QStringLiteral("className"), prediction.className);
    object.insert(QStringLiteral("objectness"), prediction.objectness);
    object.insert(QStringLiteral("confidence"), prediction.confidence);
    object.insert(QStringLiteral("box"), box);
    return object;
}

QJsonObject segmentationPredictionToJson(const SegmentationPrediction& prediction)
{
    QJsonObject object = detectionPredictionToJson(prediction.detection);
    object.insert(QStringLiteral("taskType"), QStringLiteral("segmentation"));
    object.insert(QStringLiteral("maskArea"), prediction.maskArea);
    object.insert(QStringLiteral("maskThreshold"), prediction.maskThreshold);
    object.insert(QStringLiteral("hasMask"), !prediction.mask.isNull());
    return object;
}

QJsonObject ocrRecPredictionToJson(const OcrRecPrediction& prediction)
{
    QJsonArray tokens;
    for (int token : prediction.tokens) {
        tokens.append(token);
    }
    return QJsonObject{
        {QStringLiteral("taskType"), QStringLiteral("ocr_recognition")},
        {QStringLiteral("text"), prediction.text},
        {QStringLiteral("confidence"), prediction.confidence},
        {QStringLiteral("blankIndex"), prediction.blankIndex},
        {QStringLiteral("tokens"), tokens}
    };
}

QJsonObject ocrDetPredictionToJson(const OcrDetPrediction& prediction)
{
    QJsonArray points;
    for (const QPointF& point : prediction.polygon) {
        QJsonArray item;
        item.append(point.x());
        item.append(point.y());
        points.append(item);
    }

    QJsonObject object;
    object.insert(QStringLiteral("taskType"), QStringLiteral("ocr_detection"));
    object.insert(QStringLiteral("confidence"), prediction.confidence);
    object.insert(QStringLiteral("pixelArea"), prediction.pixelArea);
    object.insert(QStringLiteral("points"), points);
    object.insert(QStringLiteral("box"), boxObject(prediction.box));
    return object;
}

QImage renderDetectionPredictions(
    const QString& imagePath,
    const QVector<DetectionPrediction>& predictions,
    QString* error)
{
    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot render prediction image: %1").arg(imagePath);
        }
        return {};
    }

    QImage output = image.convertToFormat(QImage::Format_RGB888);
    const QColor color(220, 40, 40);
    const int thickness = qMax(2, output.width() / 240);
    auto drawPoint = [&output, &color](int x, int y) {
        if (x >= 0 && x < output.width() && y >= 0 && y < output.height()) {
            output.setPixelColor(x, y, color);
        }
    };
    for (const DetectionPrediction& prediction : predictions) {
        const double imageWidth = static_cast<double>(output.width());
        const double imageHeight = static_cast<double>(output.height());
        const int x = qRound((prediction.box.xCenter - prediction.box.width / 2.0) * imageWidth);
        const int y = qRound((prediction.box.yCenter - prediction.box.height / 2.0) * imageHeight);
        const int width = qRound(prediction.box.width * imageWidth);
        const int height = qRound(prediction.box.height * imageHeight);
        QRect rect(x, y, width, height);
        rect = rect.intersected(output.rect());
        if (rect.isEmpty()) {
            continue;
        }

        for (int offset = 0; offset < thickness; ++offset) {
            for (int px = rect.left(); px <= rect.right(); ++px) {
                drawPoint(px, rect.top() + offset);
                drawPoint(px, rect.bottom() - offset);
            }
            for (int py = rect.top(); py <= rect.bottom(); ++py) {
                drawPoint(rect.left() + offset, py);
                drawPoint(rect.right() - offset, py);
            }
        }
    }
    return output;
}

QImage renderSegmentationPredictions(
    const QString& imagePath,
    const QVector<SegmentationPrediction>& predictions,
    QString* error)
{
    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot render segmentation prediction image: %1").arg(imagePath);
        }
        return {};
    }

    QImage output = image.convertToFormat(QImage::Format_ARGB32);
    QPainter painter(&output);
    painter.setRenderHint(QPainter::Antialiasing, true);
    for (const SegmentationPrediction& prediction : predictions) {
        if (!prediction.mask.isNull()) {
            const QColor fillColor = overlayColorForClass(prediction.detection.box.classId, 95);
            const int maskHeight = qMin(output.height(), prediction.mask.height());
            const int maskWidth = qMin(output.width(), prediction.mask.width());
            for (int y = 0; y < maskHeight; ++y) {
                const QRgb* maskLine = reinterpret_cast<const QRgb*>(prediction.mask.constScanLine(y));
                for (int x = 0; x < maskWidth; ++x) {
                    if (qAlpha(maskLine[x]) > 0) {
                        painter.fillRect(QRect(x, y, 1, 1), fillColor);
                    }
                }
            }
        }

        const double imageWidth = static_cast<double>(output.width());
        const double imageHeight = static_cast<double>(output.height());
        QRect rect(
            qRound((prediction.detection.box.xCenter - prediction.detection.box.width / 2.0) * imageWidth),
            qRound((prediction.detection.box.yCenter - prediction.detection.box.height / 2.0) * imageHeight),
            qRound(prediction.detection.box.width * imageWidth),
            qRound(prediction.detection.box.height * imageHeight));
        rect = rect.intersected(output.rect());
        if (rect.isEmpty()) {
            continue;
        }
        painter.setPen(QPen(overlayColorForClass(prediction.detection.box.classId, 230), qMax(2, output.width() / 240)));
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(rect);
    }
    painter.end();
    return output;
}

QString pixelGlyph(QChar ch)
{
    ch = ch.toLower();
    switch (ch.toLatin1()) {
    case 'a': return QStringLiteral("01110""10001""10001""11111""10001""10001""10001");
    case 'b': return QStringLiteral("11110""10001""10001""11110""10001""10001""11110");
    case 'c': return QStringLiteral("01111""10000""10000""10000""10000""10000""01111");
    case 'd': return QStringLiteral("11110""10001""10001""10001""10001""10001""11110");
    case 'e': return QStringLiteral("11111""10000""10000""11110""10000""10000""11111");
    case 'f': return QStringLiteral("11111""10000""10000""11110""10000""10000""10000");
    case 'g': return QStringLiteral("01111""10000""10000""10111""10001""10001""01111");
    case 'h': return QStringLiteral("10001""10001""10001""11111""10001""10001""10001");
    case 'i': return QStringLiteral("11111""00100""00100""00100""00100""00100""11111");
    case 'j': return QStringLiteral("00111""00010""00010""00010""00010""10010""01100");
    case 'k': return QStringLiteral("10001""10010""10100""11000""10100""10010""10001");
    case 'l': return QStringLiteral("10000""10000""10000""10000""10000""10000""11111");
    case 'm': return QStringLiteral("10001""11011""10101""10101""10001""10001""10001");
    case 'n': return QStringLiteral("10001""11001""10101""10011""10001""10001""10001");
    case 'o': return QStringLiteral("01110""10001""10001""10001""10001""10001""01110");
    case 'p': return QStringLiteral("11110""10001""10001""11110""10000""10000""10000");
    case 'q': return QStringLiteral("01110""10001""10001""10001""10101""10010""01101");
    case 'r': return QStringLiteral("11110""10001""10001""11110""10100""10010""10001");
    case 's': return QStringLiteral("01111""10000""10000""01110""00001""00001""11110");
    case 't': return QStringLiteral("11111""00100""00100""00100""00100""00100""00100");
    case 'u': return QStringLiteral("10001""10001""10001""10001""10001""10001""01110");
    case 'v': return QStringLiteral("10001""10001""10001""10001""10001""01010""00100");
    case 'w': return QStringLiteral("10001""10001""10001""10101""10101""10101""01010");
    case 'x': return QStringLiteral("10001""10001""01010""00100""01010""10001""10001");
    case 'y': return QStringLiteral("10001""10001""01010""00100""00100""00100""00100");
    case 'z': return QStringLiteral("11111""00001""00010""00100""01000""10000""11111");
    case '0': return QStringLiteral("01110""10001""10011""10101""11001""10001""01110");
    case '1': return QStringLiteral("00100""01100""00100""00100""00100""00100""01110");
    case '2': return QStringLiteral("01110""10001""00001""00010""00100""01000""11111");
    case '3': return QStringLiteral("11110""00001""00001""01110""00001""00001""11110");
    case '4': return QStringLiteral("00010""00110""01010""10010""11111""00010""00010");
    case '5': return QStringLiteral("11111""10000""10000""11110""00001""00001""11110");
    case '6': return QStringLiteral("01110""10000""10000""11110""10001""10001""01110");
    case '7': return QStringLiteral("11111""00001""00010""00100""01000""01000""01000");
    case '8': return QStringLiteral("01110""10001""10001""01110""10001""10001""01110");
    case '9': return QStringLiteral("01110""10001""10001""01111""00001""00001""01110");
    default: return QStringLiteral("11111""10001""00010""00100""00000""00100""00100");
    }
}

void drawPixelText(QImage* image, QPoint origin, const QString& text, const QColor& color, int scale)
{
    if (!image || image->isNull()) {
        return;
    }
    int cursorX = origin.x();
    for (const QChar ch : text.left(32)) {
        if (ch.isSpace()) {
            cursorX += 4 * scale;
            continue;
        }
        const QString glyph = pixelGlyph(ch);
        for (int row = 0; row < 7; ++row) {
            for (int col = 0; col < 5; ++col) {
                if (glyph.at(row * 5 + col) != QLatin1Char('1')) {
                    continue;
                }
                for (int dy = 0; dy < scale; ++dy) {
                    for (int dx = 0; dx < scale; ++dx) {
                        const int x = cursorX + col * scale + dx;
                        const int y = origin.y() + row * scale + dy;
                        if (image->rect().contains(x, y)) {
                            image->setPixelColor(x, y, color);
                        }
                    }
                }
            }
        }
        cursorX += 6 * scale;
        if (cursorX >= image->width()) {
            break;
        }
    }
}

QImage renderOcrRecPrediction(
    const QString& imagePath,
    const OcrRecPrediction& prediction,
    QString* error)
{
    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot render OCR Rec prediction image: %1").arg(imagePath);
        }
        return {};
    }

    QImage output(image.width(), image.height() + 40, QImage::Format_ARGB32);
    output.fill(Qt::white);
    QPainter painter(&output);
    painter.drawImage(QPoint(0, 0), image.convertToFormat(QImage::Format_ARGB32));
    painter.fillRect(QRect(0, image.height(), output.width(), 40), QColor(20, 20, 20));
    painter.end();
    const QString text = prediction.text.isEmpty() ? QStringLiteral("empty") : prediction.text;
    drawPixelText(&output, QPoint(8, image.height() + 8), text, Qt::white, 3);
    return output;
}

QImage renderOcrDetPredictions(
    const QString& imagePath,
    const QVector<OcrDetPrediction>& predictions,
    QString* error)
{
    QImage image(imagePath);
    if (image.isNull()) {
        if (error) {
            *error = QStringLiteral("Cannot render OCR Det prediction image: %1").arg(imagePath);
        }
        return {};
    }

    QImage output = image.convertToFormat(QImage::Format_ARGB32);
    QPainter painter(&output);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(QColor(245, 158, 11), qMax(2, output.width() / 240)));
    painter.setBrush(QColor(245, 158, 11, 45));
    for (const OcrDetPrediction& prediction : predictions) {
        if (prediction.polygon.size() < 4) {
            continue;
        }
        QPolygonF polygon;
        for (const QPointF& point : prediction.polygon) {
            polygon << point;
        }
        painter.drawPolygon(polygon);
    }
    painter.end();
    return output;
}


} // namespace aitrain

