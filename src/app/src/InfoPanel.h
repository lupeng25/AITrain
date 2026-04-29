#pragma once

#include <QFrame>

class QVBoxLayout;

class InfoPanel : public QFrame {
    Q_OBJECT

public:
    explicit InfoPanel(const QString& title, QWidget* parent = nullptr);

    QVBoxLayout* bodyLayout() const;

private:
    QVBoxLayout* bodyLayout_ = nullptr;
};

