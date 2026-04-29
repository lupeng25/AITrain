#include "MetricsWidget.h"

#include <QPainter>
#include <QtMath>

MetricsWidget::MetricsWidget(QWidget* parent)
    : QWidget(parent)
{
    setMinimumHeight(220);
}

void MetricsWidget::clear()
{
    series_.clear();
    update();
}

void MetricsWidget::addMetric(const QString& name, double value)
{
    series_[name].append(value);
    update();
}

void MetricsWidget::paintEvent(QPaintEvent* event)
{
    Q_UNUSED(event)

    QPainter painter(this);
    painter.fillRect(rect(), QColor(250, 251, 252));
    painter.setRenderHint(QPainter::Antialiasing);

    const QRect plot = rect().adjusted(48, 18, -18, -36);
    painter.setPen(QColor(210, 216, 222));
    painter.drawRect(plot);
    painter.setPen(QColor(80, 86, 92));
    painter.drawText(12, 24, QStringLiteral("metrics"));

    if (series_.isEmpty()) {
        painter.setPen(QColor(130, 136, 142));
        painter.drawText(plot, Qt::AlignCenter, QStringLiteral("No metric data"));
        return;
    }

    double minValue = 0.0;
    double maxValue = 1.0;
    int maxCount = 1;
    for (auto it = series_.cbegin(); it != series_.cend(); ++it) {
        maxCount = qMax(maxCount, it.value().size());
        for (double value : it.value()) {
            minValue = qMin(minValue, value);
            maxValue = qMax(maxValue, value);
        }
    }
    if (qFuzzyCompare(minValue, maxValue)) {
        maxValue += 1.0;
    }

    const QList<QColor> colors = {
        QColor(37, 99, 235),
        QColor(22, 163, 74),
        QColor(220, 38, 38),
        QColor(147, 51, 234),
        QColor(234, 88, 12)
    };

    int colorIndex = 0;
    int legendY = plot.bottom() + 22;
    for (auto it = series_.cbegin(); it != series_.cend(); ++it) {
        const QColor color = colors.at(colorIndex % colors.size());
        painter.setPen(QPen(color, 2.0));

        QPainterPath path;
        const QVector<double>& values = it.value();
        for (int i = 0; i < values.size(); ++i) {
            const double xRatio = maxCount <= 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(maxCount - 1);
            const double yRatio = (values.at(i) - minValue) / (maxValue - minValue);
            const QPointF point(plot.left() + xRatio * plot.width(), plot.bottom() - yRatio * plot.height());
            if (i == 0) {
                path.moveTo(point);
            } else {
                path.lineTo(point);
            }
        }
        painter.drawPath(path);

        painter.setPen(color);
        painter.drawText(plot.left() + colorIndex * 140, legendY, it.key());
        ++colorIndex;
    }
}

