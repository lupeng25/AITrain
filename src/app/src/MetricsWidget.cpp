#include "MetricsWidget.h"

#include <QFont>
#include <QFontMetrics>
#include <QPainter>
#include <QPainterPath>
#include <QStringList>
#include <QtMath>

namespace {
QString displayMetricName(const QString& name)
{
    if (name == QStringLiteral("classLoss")) {
        return QStringLiteral("clsLoss");
    }
    if (name == QStringLiteral("precision")) {
        return QStringLiteral("P");
    }
    if (name == QStringLiteral("recall")) {
        return QStringLiteral("R");
    }
    if (name == QStringLiteral("mAP50_95")) {
        return QStringLiteral("mAP50-95");
    }
    if (name == QStringLiteral("maskMap50_95")) {
        return QStringLiteral("mask mAP50-95");
    }
    return name;
}

double clampRatio(double value)
{
    return qBound(0.0, value, 1.0);
}
} // namespace

MetricsWidget::MetricsWidget(QWidget* parent)
    : QWidget(parent)
{
    setMinimumHeight(220);
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
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect(), palette().color(QPalette::Window));

    if (series_.isEmpty()) {
        const QRect plot = rect().adjusted(12, 12, -12, -12);
        painter.setPen(palette().color(QPalette::Mid));
        painter.drawRect(plot);
        painter.setPen(palette().color(QPalette::Disabled, QPalette::Text));
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
    const QRect content = rect().adjusted(8, 8, -8, -8);
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
        const QRect panel(content.left(), content.top() + groupIndex * (panelHeight + panelGap), content.width(), panelHeight);

        painter.setPen(palette().color(QPalette::Mid));
        painter.setBrush(palette().color(QPalette::Base));
        painter.drawRect(panel);

        const QRect inner = panel.adjusted(10, 8, -10, -8);
        const int titleWidth = qBound(74, inner.width() / 5, 118);
        const int legendLeft = inner.left() + titleWidth + 10;
        const int legendWidth = qMax(80, inner.right() - legendLeft + 1);
        const int legendItemMinWidth = 118;
        const int legendColumns = qMax(1, qMin(names.size(), legendWidth / legendItemMinWidth));
        const int legendRows = qMax(1, (names.size() + legendColumns - 1) / legendColumns);
        const int legendItemWidth = qMax(legendItemMinWidth, legendWidth / qMax(1, legendColumns));
        const int legendItemHeight = 16;
        const int headerHeight = qMin(inner.height() / 2, qMax(24, legendRows * legendItemHeight + 2));

        QFont titleFont = painter.font();
        titleFont.setBold(true);
        painter.setFont(titleFont);
        painter.setPen(palette().color(QPalette::Text));
        painter.drawText(QRect(inner.left(), inner.top(), titleWidth, 18),
            Qt::AlignLeft | Qt::AlignVCenter,
            labelForGroup(group));

        QFont legendFont = painter.font();
        legendFont.setBold(false);
        legendFont.setPointSizeF(qMax(7.0, legendFont.pointSizeF() - 1.0));
        painter.setFont(legendFont);
        const QFontMetrics legendMetrics(legendFont);
        for (int localIndex = 0; localIndex < names.size(); ++localIndex) {
            const QString& name = names.at(localIndex);
            const QColor color = colors.at((globalColorIndex + localIndex) % colors.size());
            const QVector<double>& values = series_.value(name);
            const double latest = values.isEmpty() ? 0.0 : values.last();
            const int column = localIndex % legendColumns;
            const int row = localIndex / legendColumns;
            const QRect itemRect(
                legendLeft + column * legendItemWidth,
                inner.top() + row * legendItemHeight,
                qMax(24, legendItemWidth - 8),
                legendItemHeight);
            const int markerY = itemRect.center().y();
            painter.setPen(QPen(color, 2.0));
            painter.drawLine(itemRect.left(), markerY, itemRect.left() + 12, markerY);
            painter.setPen(color);
            const QString legendText = legendMetrics.elidedText(
                QStringLiteral("%1 %2").arg(displayMetricName(name)).arg(latest, 0, 'f', 4),
                Qt::ElideRight,
                qMax(20, itemRect.width() - 18));
            painter.drawText(itemRect.adjusted(18, 0, 0, 0), Qt::AlignLeft | Qt::AlignVCenter, legendText);
        }

        const int plotTop = inner.top() + headerHeight + 8;
        QRect plot(inner.left() + 42, plotTop, inner.width() - 48, inner.bottom() - plotTop - 16);
        if (plot.height() < 32) {
            plot = QRect(inner.left() + 36, inner.top() + 30, inner.width() - 42, qMax(32, inner.height() - 44));
        }

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
        if (group == QStringLiteral("loss")) {
            maxValue = qMax(1.0, maxValue * 1.08);
        }

        painter.setBrush(Qt::NoBrush);
        painter.setPen(palette().color(QPalette::Mid));
        painter.drawRect(plot);
        for (int gridLine = 1; gridLine < 3; ++gridLine) {
            const int y = plot.top() + gridLine * plot.height() / 3;
            painter.setPen(palette().color(QPalette::AlternateBase));
            painter.drawLine(plot.left(), y, plot.right(), y);
        }

        painter.setFont(legendFont);
        painter.setPen(palette().color(QPalette::Disabled, QPalette::Text));
        painter.drawText(QRect(inner.left(), plot.top() - 2, 36, 14),
            Qt::AlignRight | Qt::AlignVCenter,
            QString::number(maxValue, 'g', 3));
        painter.drawText(QRect(inner.left(), plot.bottom() - 12, 36, 14),
            Qt::AlignRight | Qt::AlignVCenter,
            QString::number(minValue, 'g', 3));

        for (const QString& name : names) {
            const QColor color = colors.at(globalColorIndex % colors.size());
            painter.setPen(QPen(color, 2.0));

            QPainterPath path;
            QVector<QPointF> points;
            const QVector<double>& values = series_.value(name);
            for (int i = 0; i < values.size(); ++i) {
                const double xRatio = maxCount <= 1 ? 0.5 : static_cast<double>(i) / static_cast<double>(maxCount - 1);
                const double yRatio = clampRatio((values.at(i) - minValue) / (maxValue - minValue));
                const QPointF point(plot.left() + xRatio * plot.width(), plot.bottom() - yRatio * plot.height());
                points.append(point);
                if (i == 0) {
                    path.moveTo(point);
                } else {
                    path.lineTo(point);
                }
            }
            painter.drawPath(path);
            painter.setBrush(color);
            painter.setPen(QPen(palette().color(QPalette::Base), 1.0));
            for (const QPointF& point : points) {
                painter.drawEllipse(point, 3.5, 3.5);
            }
            painter.setBrush(Qt::NoBrush);
            ++globalColorIndex;
        }
    }
}

