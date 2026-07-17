#include "DeliveryEvidencePresenter.h"

DeliveryEvidencePresenter::DeliveryEvidencePresenter(
    const aitrain::ProjectQueryService* queryService, QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
{
    setObjectName(QStringLiteral("DeliveryEvidencePresenter"));
}

bool DeliveryEvidencePresenter::refresh(int limit)
{
    QString error;
    if (!queryService_) {
        error = QStringLiteral("交付证据查询服务未初始化。");
    }
    DeliveryEvidenceViewModel next;
    if (error.isEmpty()) {
        next.records = queryService_->deliveryEvidence(limit, &error);
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

void DeliveryEvidencePresenter::clear()
{
    viewModel_ = {};
    lastError_.clear();
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
