#pragma once

#include <QButtonGroup>
#include <QFrame>

class QVBoxLayout;

class Sidebar : public QFrame {
    Q_OBJECT

public:
    explicit Sidebar(QWidget* parent = nullptr);
    void addSection(const QString& text);
    void addItem(const QString& text, int pageIndex);
    void setCurrentIndex(int pageIndex);

signals:
    void pageRequested(int pageIndex, const QString& title);

private:
    QButtonGroup buttons_;
    QVBoxLayout* itemsLayout_ = nullptr;
};
