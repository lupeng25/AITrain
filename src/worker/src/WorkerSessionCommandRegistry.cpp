#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

namespace wp = aitrain::worker_protocol;

QVector<WorkerSession::CommandBinding> WorkerSession::commandBindings()
{
    // New Worker commands must define protocol constants, register a binding here,
    // and add/extend focused tests before relying on GUI entry points.
    return {
        {wp::command::startTrain(), &WorkerSession::startTrainingCommand},
        {wp::command::pause(), &WorkerSession::pauseCommand},
        {wp::command::resume(), &WorkerSession::resumeCommand},
        {wp::command::heartbeat(), &WorkerSession::heartbeatCommand},
        {wp::command::environmentCheck(), &WorkerSession::environmentCheckCommand},
        {wp::command::validateDataset(), &WorkerSession::validateDatasetCommand},
        {wp::command::splitDataset(), &WorkerSession::splitDatasetCommand},
        {wp::command::convertDataset(), &WorkerSession::convertDatasetCommand},
        {wp::command::curateDataset(), &WorkerSession::curateDatasetCommand},
        {wp::command::prepareAnnotationSession(), &WorkerSession::prepareAnnotationSessionCommand},
        {wp::command::syncAnnotationSession(), &WorkerSession::syncAnnotationSessionCommand},
        {wp::command::createDatasetSnapshot(), &WorkerSession::createDatasetSnapshotCommand},
        {wp::command::evaluateModel(), &WorkerSession::evaluateModelCommand},
        {wp::command::benchmarkModel(), &WorkerSession::benchmarkModelCommand},
        {wp::command::runLocalPipeline(), &WorkerSession::runLocalPipelineCommand},
        {wp::command::generateDeliveryReport(), &WorkerSession::generateDeliveryReportCommand},
        {wp::command::runCustomerOcrAcceptance(), &WorkerSession::runCustomerOcrAcceptanceCommand},
        {wp::command::collectDiagnostics(), &WorkerSession::collectDiagnosticsCommand},
        {wp::command::validateDeploymentArtifact(), &WorkerSession::validateDeploymentArtifactCommand},
        {wp::command::exportModel(), &WorkerSession::exportModelCommand},
        {wp::command::infer(), &WorkerSession::inferCommand},
        {wp::command::cancel(), &WorkerSession::cancelCommand},
    };
}
