#pragma once

#include <QButtonGroup>
#include <QFrame>
#include <QList>

class QLabel;
class QPushButton;
class QVBoxLayout;

class Sidebar : public QFrame {
    Q_OBJECT

public:
    explicit Sidebar(QWidget* parent = nullptr);
    void addSection(const QString& text);
    void addItem(const QString& text, int pageIndex);
    void addToolItem(const QString& text, int pageIndex);
    void setCurrentIndex(int pageIndex);
    void setCompact(bool compact);

signals:
    void pageRequested(int pageIndex, const QString& title);

private:
    void resizeEvent(QResizeEvent* event) override;
    bool compact_ = false;
    QButtonGroup buttons_;
    QVBoxLayout* itemsLayout_ = nullptr;
    QVBoxLayout* toolsLayout_ = nullptr;
    QLabel* brandTitle_ = nullptr;
    QLabel* brandSubtitle_ = nullptr;
    QLabel* userText_ = nullptr;
    QList<QLabel*> sectionLabels_;
    QList<QPushButton*> navigationButtons_;
};
