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
    ++refreshGeneration_;
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

bool DeliveryEvidencePresenter::refreshAsync(int limit)
{
    const quint64 generation = ++refreshGeneration_;
    QString error;
    if (!queryService_) {
        error = QStringLiteral("交付证据查询服务未初始化。");
    }
    if (!error.isEmpty()) {
        clear();
        lastError_ = error;
        emit queryFailed(error);
        return false;
    }

    const bool scheduled = queryService_->deliveryEvidenceAsync(limit, this,
        [this, generation](bool success,
            QVector<aitrain::DeliveryEvidenceReadModel> records, QString callbackError) {
            if (generation != refreshGeneration_) return;
            if (!success) {
                clear();
                lastError_ = callbackError;
                emit queryFailed(lastError_);
                return;
            }
            DeliveryEvidenceViewModel next;
            next.records = std::move(records);
            next.available = true;
            for (const auto& record : next.records) {
                if (record.verified) ++next.verifiedCount;
                else ++next.unverifiedCount;
            }
            viewModel_ = std::move(next);
            lastError_.clear();
            emit changed();
        }, &error);
    if (!scheduled) {
        clear();
        lastError_ = error;
        emit queryFailed(error);
        return false;
    }
    return true;
}

void DeliveryEvidencePresenter::clear()
{
    ++refreshGeneration_;
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
