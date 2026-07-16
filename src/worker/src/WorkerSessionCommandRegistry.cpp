#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

namespace wp = aitrain::worker_protocol;

QVector<WorkerSession::CommandBinding> WorkerSession::commandBindings()
{
    // New Worker commands must define protocol constants, register a binding here,
    // and add/extend focused tests before relying on GUI entry points.
    return {
        {wp::command::runEnvironmentCheckWorkflow(), &WorkerSession::runEnvironmentCheckWorkflowCommand},
        {wp::command::runDatasetSplitWorkflow(), &WorkerSession::runDatasetSplitWorkflowCommand},
        {wp::command::runDatasetConversionWorkflow(), &WorkerSession::runDatasetConversionWorkflowCommand},
        {wp::command::runDataQualityWorkflow(), &WorkerSession::runDataQualityWorkflowCommand},
        {wp::command::runDiagnosticsWorkflow(), &WorkerSession::runDiagnosticsWorkflowCommand},
        {wp::command::createAnnotationSession(), &WorkerSession::createAnnotationSessionCommand},
        {wp::command::syncAnnotationSession(), &WorkerSession::syncAnnotationSessionCommand},
        {wp::command::runDatasetSnapshotImportWorkflow(), &WorkerSession::runDatasetSnapshotImportWorkflowCommand},
        {wp::command::importOcrOfficialReports(), &WorkerSession::importOcrOfficialReportsCommand},
        {wp::command::runOcrAcceptanceWorkflow(), &WorkerSession::runOcrAcceptanceWorkflowCommand},
        {wp::command::runRuntimeDeliveryWorkflow(), &WorkerSession::runRuntimeDeliveryWorkflowCommand},
        {wp::command::importModel(), &WorkerSession::importModelCommand},
        {wp::command::runTrainingWorkflow(), &WorkerSession::runTrainingWorkflowCommand},
    };
}
