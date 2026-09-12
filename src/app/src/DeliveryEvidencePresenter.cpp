#include "WorkbenchTranslation.h"
#include "DeliveryEvidencePresenter.h"

DeliveryEvidencePresenter::DeliveryEvidencePresenter(
    const aitrain::ProjectQueryService* queryService, QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
{
    setObjectName(QStringLiteral("DeliveryEvidencePresenter"));
}

bool DeliveryEvidencePresenter::refresh(const aitrain::PageRequest& request)
{
    ++refreshGeneration_;
    QString error;
    if (!queryService_) {
        error = aitrain_app::workbenchText(QStringLiteral("交付证据查询服务未初始化。"));
    }
    DeliveryEvidenceViewModel next;
    if (error.isEmpty()) {
        const auto page = queryService_->deliveryEvidence(request, &error, filter_);
        next.records = request.after.isEmpty() ? page.items : viewModel_.records + page.items;
        nextCursor_ = page.nextCursor;
        hasMore_ = page.hasMore;
        next.available = error.isEmpty();
        for (const auto& record : next.records) {
            if (record.verified) ++next.verifiedCount;
            else ++next.unverifiedCount;
        }
    }
    if (!error.isEmpty()) {
        clear();
        lastError_ = error;
        emit queryFailed(error);
        return false;
    }
    viewModel_ = next;
    lastError_.clear();
    emit changed();
    return true;
}

bool DeliveryEvidencePresenter::refreshAsync(const aitrain::PageRequest& request)
{
    const quint64 generation = ++refreshGeneration_;
    QString error;
    if (!queryService_) {
        error = aitrain_app::workbenchText(QStringLiteral("交付证据查询服务未初始化。"));
    }
    if (!error.isEmpty()) {
        clear();
        lastError_ = error;
        emit queryFailed(error);
        return false;
    }

    const bool append = !request.after.isEmpty();
    const bool scheduled = queryService_->deliveryEvidenceAsync(request, this,
        [this, generation, append](bool success,
            aitrain::Page<aitrain::DeliveryEvidenceReadModel> page, QString callbackError) {
            if (generation != refreshGeneration_) return;
            if (!success) {
                clear();
                lastError_ = callbackError;
                emit queryFailed(lastError_);
                return;
            }
            DeliveryEvidenceViewModel next;
            next.records = append ? viewModel_.records + page.items : std::move(page.items);
            next.available = true;
            for (const auto& record : next.records) {
                if (record.verified) ++next.verifiedCount;
                else ++next.unverifiedCount;
            }
            viewModel_ = std::move(next);
            nextCursor_ = page.nextCursor;
            hasMore_ = page.hasMore;
            lastError_.clear();
            emit changed();
        }, &error, filter_);
    if (!scheduled) {
        clear();
        lastError_ = error;
        emit queryFailed(error);
        return false;
    }
    return true;
}

bool DeliveryEvidencePresenter::loadMoreAsync()
{
    return hasMore_ && refreshAsync({50, nextCursor_});
}

bool DeliveryEvidencePresenter::hasMore() const { return hasMore_; }

void DeliveryEvidencePresenter::clear()
{
    ++refreshGeneration_;
    viewModel_ = {};
    lastError_.clear();
    nextCursor_.clear();
    hasMore_ = false;
    emit changed();
}

const DeliveryEvidenceViewModel& DeliveryEvidencePresenter::viewModel() const
{
    return viewModel_;
}

QString DeliveryEvidencePresenter::lastError() const
{
    return lastError_;
}
