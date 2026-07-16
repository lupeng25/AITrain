#include "WorkspaceRouter.h"

WorkspaceRouter::WorkspaceRouter(int pageCount, QObject* parent)
    : QObject(parent)
    , pageCount_(pageCount)
{
    setObjectName(QStringLiteral("WorkspaceRouter"));
}

int WorkspaceRouter::currentPageIndex() const
{
    return currentPageIndex_;
}

QString WorkspaceRouter::currentTitle() const
{
    return currentTitle_;
}

void WorkspaceRouter::navigate(int pageIndex, const QString& title)
{
    if (accepts(pageIndex)) {
        emit pageRequested(pageIndex, title);
    }
}

void WorkspaceRouter::synchronize(int pageIndex, const QString& title)
{
    if (!accepts(pageIndex)) {
        return;
    }
    const QString normalizedTitle = title.trimmed();
    if (currentPageIndex_ == pageIndex && currentTitle_ == normalizedTitle) {
        return;
    }
    currentPageIndex_ = pageIndex;
    currentTitle_ = normalizedTitle;
    emit currentPageChanged(currentPageIndex_, currentTitle_);
}

bool WorkspaceRouter::accepts(int pageIndex) const
{
    return pageIndex >= 0 && pageIndex < pageCount_;
}
