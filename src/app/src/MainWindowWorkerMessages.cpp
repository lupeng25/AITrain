#include "MainWindow.h"

#include "ApplicationEventRouter.h"
#include "EnvironmentCheckPresenter.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "MetricsWidget.h"

#include <QFileInfo>

using namespace aitrain_app;

void MainWindow::handleTaskViewStateChanged(const TaskViewState& state)
{
    if (activeTaskId_.isEmpty() || state.taskId != activeTaskId_) {
        return;
    }

    if (activeWorkflowKind_ == QStringLiteral("training") && progressBar_) {
        progressBar_->setValue(state.progress);
        if (trainingPhaseLabel_ && !state.terminal) {
            trainingPhaseLabel_->setText(uiText(
                "阶段：校验快照 -> 训练 -> 评估 -> 导出 -> 部署验证 -> 登记模型 -> 交付报告 | 当前：Worker 运行中（%1%）")
                .arg(state.progress));
        }
        const qint64 pendingMetricCount = qMax<qint64>(0, state.metricSequence - liveMetricSequence_);
        const int firstMetricIndex = qMax(0,
            state.metrics.size() - static_cast<int>(qMin<qint64>(pendingMetricCount, state.metrics.size())));
        for (int metricIndex = firstMetricIndex; metricIndex < state.metrics.size(); ++metricIndex) {
            const TaskMetricView& metric = state.metrics.at(metricIndex);
            if (metricsWidget_ && !metric.name.isEmpty()) {
                metricsWidget_->addMetric(metric.name, metric.value);
            }
            if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingEpochValue"));
                label && metric.details.contains(QStringLiteral("epoch"))) {
                label->setText(QString::number(metric.details.value(QStringLiteral("epoch")).toInt()));
            }
            if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingBatchValue"));
                label && metric.details.contains(QStringLiteral("step"))) {
                label->setText(QString::number(metric.details.value(QStringLiteral("step")).toInt()));
            }
            if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingDeviceValue"));
                label && metric.details.value(QStringLiteral("device")).isString()) {
                label->setText(metric.details.value(QStringLiteral("device")).toString());
            }
            const QString lowerName = metric.name.toLower();
            if (lowerName.contains(QStringLiteral("loss"))
                || lowerName == QStringLiteral("cer")
                || lowerName == QStringLiteral("wer")) {
                if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingLossValue"))) {
                    label->setText(QStringLiteral("%1: %2").arg(metric.name).arg(metric.value, 0, 'g', 6));
                }
            } else if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingMapValue"))) {
                label->setText(QStringLiteral("%1: %2").arg(metric.name).arg(metric.value, 0, 'g', 6));
            }
        }
        liveMetricSequence_ = state.metricSequence;
        const qint64 pendingArtifactCount = qMax<qint64>(0, state.artifactSequence - liveArtifactSequence_);
        const int firstArtifactIndex = qMax(0,
            state.artifacts.size() - static_cast<int>(qMin<qint64>(pendingArtifactCount, state.artifacts.size())));
        for (int artifactIndex = firstArtifactIndex; artifactIndex < state.artifacts.size(); ++artifactIndex) {
            const TaskArtifactView& artifact = state.artifacts.at(artifactIndex);
            const QString kind = artifact.kind.toLower();
            const QString suffix = QFileInfo(artifact.relativePath).suffix().toLower();
            const QString value = QStringLiteral("%1 · %2")
                .arg(artifact.artifactId.left(8), artifact.relativePath);
            if (latestCheckpointLabel_ && kind.contains(QStringLiteral("checkpoint"))) {
                latestCheckpointLabel_->setText(uiText("最新 checkpoint：%1").arg(value));
            }
            if (latestOnnxLabel_ && (suffix == QStringLiteral("onnx")
                    || kind.contains(QStringLiteral("onnx")))) {
                latestOnnxLabel_->setText(uiText("最新 ONNX：%1").arg(value));
            }
            if (latestReportLabel_ && kind.contains(QStringLiteral("report"))) {
                latestReportLabel_->setText(uiText("训练报告：%1").arg(value));
            }
            if (latestPreviewLabel_ && (kind.contains(QStringLiteral("preview"))
                    || kind.contains(QStringLiteral("overlay")))) {
                latestPreviewLabel_->setText(uiText("最新预览：%1").arg(value));
            }
        }
        liveArtifactSequence_ = state.artifactSequence;
    }
    if (activeWorkflowKind_ == QStringLiteral("dataset_conversion")
        && datasetConversionProgressBar_) {
        datasetConversionProgressBar_->setValue(state.progress);
    }

    if (!state.terminal) {
        updateTaskCancelButton();
        return;
    }

    if (activeWorkflowKind_ == QStringLiteral("dataset_conversion")) {
        setDatasetConversionFormRunning(false);
    }

    if (activeWorkflowKind_ == QStringLiteral("training")) {
        if (trainingPhaseLabel_) {
            trainingPhaseLabel_->setText(state.status == QStringLiteral("succeeded")
                ? uiText("训练 Workflow 已完成；持久化事实已刷新。")
                : uiText("训练 Workflow 已终止：%1").arg(state.status));
        }
        if (state.status == QStringLiteral("succeeded")) {
            if (auto* label = trainingLiveValueLabel(QStringLiteral("TrainingEtaValue"))) {
                label->setText(QStringLiteral("0s"));
            }
        }
    }

    if (activeWorkflowKind_ == QStringLiteral("environment_check")
        && environmentCheckPresenter_) {
        environmentCheckPresenter_->selectTask(activeTaskId_);
        refreshEnvironmentReportView();
    }

    QString workflowName = uiText("当前任务");
    if (activeWorkflowKind_ == QStringLiteral("training")) {
        workflowName = uiText("训练 Workflow");
    } else if (activeWorkflowKind_ == QStringLiteral("data_quality")) {
        workflowName = uiText("Data Quality");
    } else if (activeWorkflowKind_ == QStringLiteral("dataset_conversion")) {
        workflowName = uiText("Dataset Conversion");
    } else if (activeWorkflowKind_ == QStringLiteral("dataset_split")) {
        workflowName = uiText("Dataset Split");
    } else if (activeWorkflowKind_ == QStringLiteral("environment_check")) {
        workflowName = uiText("Environment Check");
    } else if (activeWorkflowKind_ == QStringLiteral("diagnostics")) {
        workflowName = uiText("Diagnostics Bundle");
    } else if (activeWorkflowKind_ == QStringLiteral("model_import")) {
        workflowName = uiText("模型导入");
    }

    appendLog(uiText("%1：%2：%3").arg(
        workflowName, state.status, state.terminalMessage));
    activeTaskId_.clear();
    activeWorkflowKind_.clear();
    updateTaskCancelButton();
    updateRecentTasks();
    updateSelectedTaskDetails();
    updateProjectSummary();
    updateDashboardSummary();
    updateDeliveryAcceptanceSummary();
    updateModelRegistry();
}
