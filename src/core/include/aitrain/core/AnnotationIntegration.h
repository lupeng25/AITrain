#pragma once

#include "aitrain/core/Cancellation.h"
#include "aitrain/core/DatasetConversion.h"
#include "aitrain/core/WorkflowResult.h"

#include <QJsonObject>
#include <QString>
#include <QStringList>

namespace aitrain {

QStringList xAnyLabelingExecutableCandidates(const QJsonObject& options = {});
QString resolveXAnyLabelingExecutable(const QJsonObject& options = {});

WorkflowResult inspectXAnyLabelingEnvironment(
    const QString& outputPath,
    const QJsonObject& options = {},
    const CancellationCallback& shouldCancel = {});

DatasetConversionResult convertDatasetWithXAnyLabelingCli(
    const DatasetConversionRequest& request,
    const CancellationCallback& shouldCancel = {});

} // namespace aitrain
