#pragma once

#include <QObject>
#include <QString>

// 工作台唯一页面路由。Sidebar 和页面动作只发出导航意图；测试与 Presenter
// 通过只读状态观察当前页面，不访问 MainWindow 控件内部结构。
class WorkspaceRouter final : public QObject {
    Q_OBJECT

public:
    explicit WorkspaceRouter(int pageCount, QObject* parent = nullptr);

    int currentPageIndex() const;
    QString currentTitle() const;

public slots:
    void navigate(int pageIndex, const QString& title);
    void synchronize(int pageIndex, const QString& title);

signals:
    void pageRequested(int pageIndex, const QString& title);
    void currentPageChanged(int pageIndex, const QString& title);

private:
    bool accepts(int pageIndex) const;

    int pageCount_ = 0;
    int currentPageIndex_ = -1;
    QString currentTitle_;
};
