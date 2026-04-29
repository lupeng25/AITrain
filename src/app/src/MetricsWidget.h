#pragma once

#include <QMap>
#include <QVector>
#include <QWidget>

class MetricsWidget : public QWidget {
    Q_OBJECT

public:
    explicit MetricsWidget(QWidget* parent = nullptr);

    void clear();
    void addMetric(const QString& name, double value);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QMap<QString, QVector<double>> series_;
};

