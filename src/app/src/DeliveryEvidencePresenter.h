#pragma once

#include "aitrain/workflow/ProjectQueryService.h"

#include <QObject>

struct DeliveryEvidenceViewModel final {
    bool available = false;
    QVector<aitrain::DeliveryEvidenceReadModel> records;
    int verifiedCount = 0;
    int unverifiedCount = 0;
};

class DeliveryEvidencePresenter final : public QObject {
    Q_OBJECT

public:
    explicit DeliveryEvidencePresenter(
        const aitrain::ProjectQueryService* queryService,
        QObject* parent = nullptr);

    bool refresh(int limit = 64);
    bool refreshAsync(int limit = 64);
    void clear();
    const DeliveryEvidenceViewModel& viewModel() const;
    QString lastError() const;

signals:
    void changed();
    void queryFailed(const QString& error);

private:
    const aitrain::ProjectQueryService* queryService_ = nullptr;
    DeliveryEvidenceViewModel viewModel_;
    QString lastError_;
    quint64 refreshGeneration_ = 0;
};
