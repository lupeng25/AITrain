#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

namespace wp = aitrain::worker_protocol;

QVector<WorkerSession::CommandBinding> WorkerSession::commandBindings()
{
    // New Worker commands must define protocol constants, register a binding here,
    // and add/extend focused tests before relying on GUI entry points.
    return {
        {wp::command::runEnvironmentCheckWorkflowV2(), &WorkerSession::runEnvironmentCheckWorkflowV2Command},
        {wp::command::runDatasetSplitWorkflowV2(), &WorkerSession::runDatasetSplitWorkflowV2Command},
        {wp::command::runDatasetConversionWorkflowV2(), &WorkerSession::runDatasetConversionWorkflowV2Command},
        {wp::command::runDataQualityWorkflowV2(), &WorkerSession::runDataQualityWorkflowV2Command},
        {wp::command::runDiagnosticsWorkflowV2(), &WorkerSession::runDiagnosticsWorkflowV2Command},
        {wp::command::createAnnotationSessionV2(), &WorkerSession::createAnnotationSessionV2Command},
        {wp::command::syncAnnotationSessionV2(), &WorkerSession::syncAnnotationSessionV2Command},
        {wp::command::runDatasetSnapshotImportWorkflowV2(), &WorkerSession::runDatasetSnapshotImportWorkflowV2Command},
        {wp::command::importOcrOfficialReportsV2(), &WorkerSession::importOcrOfficialReportsV2Command},
        {wp::command::runOcrAcceptanceWorkflowV2(), &WorkerSession::runOcrAcceptanceWorkflowV2Command},
        {wp::command::runRuntimeDeliveryWorkflowV2(), &WorkerSession::runRuntimeDeliveryWorkflowV2Command},
        {wp::command::importModelV2(), &WorkerSession::importModelV2Command},
        {wp::command::runTrainingWorkflowV2(), &WorkerSession::runTrainingWorkflowV2Command},
    };
}
