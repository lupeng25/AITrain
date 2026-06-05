#include "MetricsWidget.h"

#include <QPainter>
#include <QPainterPath>
#include <QStringList>
#include <QtMath>

MetricsWidget::MetricsWidget(QWidget* parent)
    : QWidget(parent)
{
    setMinimumHeight(130);
    setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Expanding);
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

QString MetricsWidget::groupForMetric(const QString& name) const
{
    const QString lower = name.toLower();
    if (lower.contains(QStringLiteral("loss")) || lower == QStringLiteral("cer") || lower == QStringLiteral("wer")) {
        return QStringLiteral("loss");
    }
    return QStringLiteral("score");
}

QString MetricsWidget::labelForGroup(const QString& group) const
{
    if (group == QStringLiteral("loss")) {
        return QStringLiteral("Loss / Error");
    }
    return QStringLiteral("Score / Quality");
}

void MetricsWidget::paintEvent(QPaintEvent* event)
{
    Q_UNUSED(event)

    QPainter painter(this);
    painter.fillRect(rect(), QColor(250, 251, 252));
    painter.setRenderHint(QPainter::Antialiasing);

    if (series_.isEmpty()) {
        const QRect plot = rect().adjusted(48, 18, -18, -18);
        painter.setPen(QColor(210, 216, 222));
        painter.drawRect(plot);
        painter.setPen(QColor(130, 136, 142));
        painter.drawText(plot, Qt::AlignCenter, QStringLiteral("No metric data"));
        return;
    }

    QMap<QString, QStringList> groups;
    for (auto it = series_.cbegin(); it != series_.cend(); ++it) {
        groups[groupForMetric(it.key())].append(it.key());
    }

    const QList<QString> orderedGroups = groups.contains(QStringLiteral("loss")) && groups.contains(QStringLiteral("score"))
        ? QList<QString>{QStringLiteral("loss"), QStringLiteral("score")}
        : groups.keys();
    const int panelGap = 12;
    const QRect content = rect().adjusted(44, 18, -18, -18);
    const int panelHeight = qMax(80, (content.height() - panelGap * qMax(0, orderedGroups.size() - 1)) / qMax(1, orderedGroups.size()));

    const QList<QColor> colors = {
        QColor(37, 99, 235),
        QColor(22, 163, 74),
        QColor(220, 38, 38),
        QColor(147, 51, 234),
        QColor(234, 88, 12),
        QColor(118, 185, 0)
    };

    int globalColorIndex = 0;
    for (int groupIndex = 0; groupIndex < orderedGroups.size(); ++groupIndex) {
        const QString group = orderedGroups.at(groupIndex);
        const QStringList names = groups.value(group);
        const QRect plot(content.left(), content.top() + groupIndex * (panelHeight + panelGap), content.width(), panelHeight);

        painter.setPen(QColor(210, 216, 222));
        painter.drawRect(plot);
        painter.setPen(QColor(80, 86, 92));
        painter.drawText(8, plot.top() + 18, labelForGroup(group));

        double minValue = group == QStringLiteral("score") ? 0.0 : 0.0;
        double maxValue = group == QStringLiteral("score") ? 1.0 : 1.0;
        int maxCount = 1;
        for (const QString& name : names) {
            const QVector<double>& values = series_.value(name);
            maxCount = qMax(maxCount, values.size());
            for (double value : values) {
                minValue = qMin(minValue, value);
                maxValue = qMax(maxValue, value);
            }
        }
        if (qFuzzyCompare(minValue, maxValue)) {
            maxValue += 1.0;
        }

        int localIndex = 0;
        for (const QString& name : names) {
            const QColor color = colors.at(globalColorIndex % colors.size());
            painter.setPen(QPen(color, 2.0));

            QPainterPath path;
            const QVector<double>& values = series_.value(name);
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
            const double latest = values.isEmpty() ? 0.0 : values.last();
            const int legendX = plot.left() + localIndex * 150;
            const int legendY = plot.top() + 18;
            painter.drawText(legendX, legendY, QStringLiteral("%1 %2").arg(name).arg(latest, 0, 'f', 4));
            ++localIndex;
            ++globalColorIndex;
        }
    }
}

