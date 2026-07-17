#include "aitrain/workflow/TrainingWorkflowProfile.h"

#include <QDir>
#include <QSet>
#include <QTest>

class TrainingWorkflowProfileTests : public QObject {
    Q_OBJECT

private slots:
    void resolvesRegisteredYoloProfiles_data();
    void resolvesRegisteredYoloProfiles();
    void resolvesSmpProfileWithStepContracts();
    void resolvesAnomalibProfiles_data();
    void resolvesAnomalibProfiles();
    void resolvesPaddleOcrProfiles_data();
    void resolvesPaddleOcrProfiles();
    void normalizesBackendTextAndRejectsAliases();
    void rejectsUnknownOrMissingBackend();
    void registeredProfilesAreCompleteAndUnique();
};

void TrainingWorkflowProfileTests::resolvesRegisteredYoloProfiles_data()
{
    QTest::addColumn<QString>("backend");
    QTest::addColumn<QString>("capabilityTaskType");
    QTest::addColumn<QString>("adapterTaskType");
    QTest::addColumn<QString>("datasetFormat");
    QTest::addColumn<QString>("modelFamily");
    QTest::addColumn<QString>("decoder");
    QTest::addColumn<QString>("trainScript");

    QTest::newRow("detection")
        << QStringLiteral("ultralytics_yolo_detect")
        << QStringLiteral("detection")
        << QStringLiteral("detection")
        << QStringLiteral("yolo_detection")
        << QStringLiteral("yolo_detection")
        << QStringLiteral("yolo_detection_v8")
        << QStringLiteral("detection/ultralytics_trainer.py");
    QTest::newRow("instance-segmentation")
        << QStringLiteral("ultralytics_yolo_segment")
        << QStringLiteral("segmentation")
        << QStringLiteral("segmentation")
        << QStringLiteral("yolo_segmentation")
        << QStringLiteral("yolo_segmentation")
        << QStringLiteral("yolo_segmentation_v8")
        << QStringLiteral("segmentation/ultralytics_trainer.py");
    QTest::newRow("obb")
        << QStringLiteral("ultralytics_yolo_obb")
        << QStringLiteral("obb_detection")
        << QStringLiteral("obb_detection")
        << QStringLiteral("yolo_obb")
        << QStringLiteral("yolo_obb")
        << QStringLiteral("yolo_obb_v8")
        << QStringLiteral("obb/ultralytics_trainer.py");
}

void TrainingWorkflowProfileTests::resolvesRegisteredYoloProfiles()
{
    QFETCH(QString, backend);
    QFETCH(QString, capabilityTaskType);
    QFETCH(QString, adapterTaskType);
    QFETCH(QString, datasetFormat);
    QFETCH(QString, modelFamily);
    QFETCH(QString, decoder);
    QFETCH(QString, trainScript);

    aitrain::TrainingWorkflowProfile profile;
    QString error;
    QVERIFY2(aitrain::resolveTrainingWorkflowProfile(backend, &profile, &error), qPrintable(error));
    QCOMPARE(profile.trainingBackend, backend);
    QCOMPARE(profile.templateId, QStringLiteral("official_yolo_training_delivery"));
    QCOMPARE(profile.capabilityTaskType, capabilityTaskType);
    QCOMPARE(profile.adapterTaskType, adapterTaskType);
    QCOMPARE(profile.datasetFormat, datasetFormat);
    QCOMPARE(profile.modelFamily, modelFamily);
    QCOMPARE(profile.decoder, decoder);
    QCOMPARE(profile.artifactFormat, QStringLiteral("onnx"));
    QCOMPARE(profile.trainScript, trainScript);
    QCOMPARE(profile.evaluationBackend, QStringLiteral("ultralytics_yolo_eval"));
    QCOMPARE(profile.evaluationScript, QStringLiteral("yolo/ultralytics_evaluator.py"));
    QCOMPARE(profile.exportBackend, QStringLiteral("ultralytics_yolo_export"));
    QCOMPARE(profile.exportScript, QStringLiteral("yolo/ultralytics_exporter.py"));
    QCOMPARE(profile.deploymentBackend, QStringLiteral("aitrain_onnxruntime"));
    QCOMPARE(profile.runtimeRoutes, QStringList{QStringLiteral("aitrain_onnxruntime")});
    QVERIFY(!profile.limitations.isEmpty());
    QCOMPARE(profile.steps.size(), 8);
    QCOMPARE(profile.steps.at(2).kind, QStringLiteral("Train"));
    QCOMPARE(profile.steps.at(2).backend, profile.trainingBackend);
    QCOMPARE(profile.steps.at(2).script, profile.trainScript);
    QCOMPARE(profile.steps.at(3).backend, profile.evaluationBackend);
    QCOMPARE(profile.steps.at(4).backend, profile.exportBackend);
    QCOMPARE(profile.steps.at(5).backend, profile.deploymentBackend);
}

void TrainingWorkflowProfileTests::resolvesSmpProfileWithStepContracts()
{
    aitrain::TrainingWorkflowProfile profile;
    QString error;
    QVERIFY2(aitrain::resolveTrainingWorkflowProfile(
        QStringLiteral("  SMP_SEMANTIC_SEGMENTATION "), &profile, &error), qPrintable(error));
    QCOMPARE(profile.trainingBackend, QStringLiteral("smp_semantic_segmentation"));
    QCOMPARE(profile.templateId, QStringLiteral("smp_semantic_segmentation_delivery"));
    QCOMPARE(profile.capabilityTaskType, QStringLiteral("semantic_segmentation"));
    QCOMPARE(profile.adapterTaskType, QStringLiteral("semantic_segmentation"));
    QCOMPARE(profile.datasetFormat, QStringLiteral("semantic_segmentation_mask"));
    QCOMPARE(profile.modelFamily, QStringLiteral("semantic_segmentation"));
    QCOMPARE(profile.decoder, QStringLiteral("smp_semantic_segmentation"));
    QCOMPARE(profile.artifactFormat, QStringLiteral("onnx"));
    QCOMPARE(profile.runtimeRoutes, QStringList{QStringLiteral("aitrain_onnxruntime")});
    QCOMPARE(profile.steps.size(), 8);

    const auto findStep = [&profile](const QString& kind) -> const aitrain::TrainingWorkflowStepProfile* {
        for (const aitrain::TrainingWorkflowStepProfile& candidate : profile.steps) {
            if (candidate.kind == kind) {
                return &candidate;
            }
        }
        return nullptr;
    };
    const auto* train = findStep(QStringLiteral("Train"));
    const auto* evaluate = findStep(QStringLiteral("Evaluate"));
    const auto* exportStep = findStep(QStringLiteral("Export"));
    const auto* deployment = findStep(QStringLiteral("DeploymentValidate"));
    const auto* registration = findStep(QStringLiteral("RegisterModel"));
    QVERIFY(train);
    QVERIFY(evaluate);
    QVERIFY(exportStep);
    QVERIFY(deployment);
    QVERIFY(registration);
    QCOMPARE(train->backend, QStringLiteral("smp_semantic_segmentation"));
    QCOMPARE(train->script, QStringLiteral("semantic_segmentation/smp_trainer.py"));
    QVERIFY(train->artifactCandidates.isEmpty());
    QCOMPARE(evaluate->backend, QStringLiteral("smp_semantic_segmentation_eval"));
    QCOMPARE(evaluate->script, QStringLiteral("semantic_segmentation/smp_evaluator.py"));
    QCOMPARE(evaluate->artifactCandidates, QStringList({QStringLiteral("onnx_model/best.onnx"),
        QStringLiteral("model_sidecar/semantic_segmentation_sidecar.json"),
        QStringLiteral("checkpoint/best.pt")}));
    QCOMPARE(exportStep->backend, QStringLiteral("smp_semantic_segmentation_export"));
    QCOMPARE(exportStep->script, QStringLiteral("semantic_segmentation/smp_exporter.py"));
    QCOMPARE(exportStep->artifactCandidates, QStringList({QStringLiteral("onnx_model/best.onnx"),
        QStringLiteral("model_sidecar/semantic_segmentation_sidecar.json"),
        QStringLiteral("evaluation_report/evaluation_report.json")}));
    const QStringList exportedModelCandidates{QStringLiteral("export/model.onnx"),
        QStringLiteral("export_sidecar/model.aitrain-export.json")};
    QCOMPARE(deployment->artifactCandidates, exportedModelCandidates);
    QCOMPARE(registration->artifactCandidates, exportedModelCandidates);
    QVERIFY(profile.limitations.join(QLatin1Char(' ')).contains(QStringLiteral("不支持 NCNN 或 TensorRT")));
}

void TrainingWorkflowProfileTests::resolvesAnomalibProfiles_data()
{
    QTest::addColumn<QString>("backend");
    QTest::addColumn<bool>("efficientAd");
    QTest::newRow("patchcore") << QStringLiteral("anomalib_patchcore") << false;
    QTest::newRow("efficientad") << QStringLiteral("anomalib_efficientad") << true;
}

void TrainingWorkflowProfileTests::resolvesAnomalibProfiles()
{
    QFETCH(QString, backend);
    QFETCH(bool, efficientAd);
    aitrain::TrainingWorkflowProfile profile;
    QString error;
    QVERIFY2(aitrain::resolveTrainingWorkflowProfile(backend, &profile, &error), qPrintable(error));
    QCOMPARE(profile.templateId, QStringLiteral("anomalib_training_delivery"));
    QCOMPARE(profile.capabilityTaskType, QStringLiteral("anomaly_detection"));
    QCOMPARE(profile.datasetFormat, QStringLiteral("anomaly_folder"));
    QCOMPARE(profile.modelFamily, QStringLiteral("anomaly_detection"));
    QCOMPARE(profile.decoder, QStringLiteral("anomalib_python_sidecar_v1"));
    QCOMPARE(profile.artifactFormat, QStringLiteral("anomalib_bundle"));
    QCOMPARE(profile.runtimeRoutes, QStringList{QStringLiteral("anomalib_python")});
    QCOMPARE(profile.steps.size(), 8);
    QCOMPARE(profile.steps.at(2).backend, backend);
    QCOMPARE(profile.steps.at(3).backend, QStringLiteral("anomalib_python_eval"));
    QCOMPARE(profile.steps.at(4).backend, QStringLiteral("anomalib_artifact_export"));
    QCOMPARE(profile.steps.at(5).backend, QStringLiteral("anomalib_python"));
    QVERIFY(!profile.steps.at(5).script.isEmpty());
    QVERIFY(profile.limitations.join(QLatin1Char(' ')).contains(QStringLiteral("不声明 AITrain C++ ONNX")));
    QCOMPARE(profile.limitations.join(QLatin1Char(' ')).contains(QStringLiteral("batchSize=1")), efficientAd);
}

void TrainingWorkflowProfileTests::resolvesPaddleOcrProfiles_data()
{
    QTest::addColumn<QString>("backend");
    QTest::addColumn<QString>("component");
    QTest::addColumn<QString>("taskType");
    QTest::addColumn<QString>("datasetFormat");
    QTest::addColumn<QString>("decoder");
    QTest::addColumn<bool>("recognition");

    QTest::newRow("det")
        << QStringLiteral("paddleocr_det_official") << QStringLiteral("det")
        << QStringLiteral("ocr_detection") << QStringLiteral("paddleocr_det")
        << QStringLiteral("paddleocr_official_det_v1") << false;
    QTest::newRow("rec")
        << QStringLiteral("paddleocr_rec_official") << QStringLiteral("rec")
        << QStringLiteral("ocr_recognition") << QStringLiteral("paddleocr_rec")
        << QStringLiteral("paddleocr_official_rec_v1") << true;
}

void TrainingWorkflowProfileTests::resolvesPaddleOcrProfiles()
{
    QFETCH(QString, backend);
    QFETCH(QString, component);
    QFETCH(QString, taskType);
    QFETCH(QString, datasetFormat);
    QFETCH(QString, decoder);
    QFETCH(bool, recognition);
    aitrain::TrainingWorkflowProfile profile;
    QString error;
    QVERIFY2(aitrain::resolveTrainingWorkflowProfile(backend, &profile, &error), qPrintable(error));
    QCOMPARE(profile.templateId, QStringLiteral("paddleocr_%1_training_delivery").arg(component));
    QCOMPARE(profile.capabilityTaskType, taskType);
    QCOMPARE(profile.adapterTaskType, taskType);
    QCOMPARE(profile.datasetFormat, datasetFormat);
    QCOMPARE(profile.modelFamily, taskType);
    QCOMPARE(profile.decoder, decoder);
    QCOMPARE(profile.artifactFormat, QStringLiteral("paddleocr_inference_bundle"));
    QCOMPARE(profile.trainScript, QStringLiteral("ocr_%1/paddleocr_%1_trainer.py").arg(component));
    QCOMPARE(profile.evaluationBackend, QStringLiteral("paddleocr_%1_official_eval").arg(component));
    QCOMPARE(profile.evaluationScript, QStringLiteral("ocr_%1/paddleocr_%1_evaluator.py").arg(component));
    QCOMPARE(profile.exportBackend, QStringLiteral("paddleocr_%1_official_export").arg(component));
    QCOMPARE(profile.exportScript, QStringLiteral("ocr_%1/paddleocr_%1_exporter.py").arg(component));
    QCOMPARE(profile.deploymentBackend, QStringLiteral("paddleocr_%1_official_runtime").arg(component));
    QCOMPARE(profile.runtimeRoutes, QStringList{QStringLiteral("paddleocr_official")});
    QCOMPARE(profile.steps.size(), 8);
    QCOMPARE(profile.steps.at(3).artifactCandidates,
        recognition
            ? QStringList({QStringLiteral("checkpoint/model.zip"), QStringLiteral("config/train.yml"),
                  QStringLiteral("dictionary/dict.txt")})
            : QStringList({QStringLiteral("checkpoint/model.zip"), QStringLiteral("config/train.yml")}));
    QStringList expectedExport = profile.steps.at(3).artifactCandidates;
    expectedExport.append(QStringLiteral("evaluation_report/evaluation_report.json"));
    QCOMPARE(profile.steps.at(4).artifactCandidates, expectedExport);
    const QStringList expectedBundle{QStringLiteral("export/paddleocr_bundle.json"),
        QStringLiteral("export/paddleocr_inference.zip")};
    QCOMPARE(profile.steps.at(5).script,
        QStringLiteral("ocr_%1/paddleocr_%1_predictor.py").arg(component));
    QCOMPARE(profile.steps.at(5).artifactCandidates, expectedBundle);
    QCOMPARE(profile.steps.at(6).artifactCandidates, expectedBundle);
    QVERIFY(profile.limitations.join(QLatin1Char(' ')).contains(QStringLiteral("官方工具链")));
}

void TrainingWorkflowProfileTests::normalizesBackendTextAndRejectsAliases()
{
    aitrain::TrainingWorkflowProfile profile;
    QString error = QStringLiteral("stale");
    QVERIFY2(aitrain::resolveTrainingWorkflowProfile(
        QStringLiteral("  ULTRALYTICS_YOLO_DETECT  "), &profile, &error), qPrintable(error));
    QCOMPARE(profile.trainingBackend, QStringLiteral("ultralytics_yolo_detect"));
    QVERIFY(error.isEmpty());
    QVERIFY(aitrain::hasTrainingWorkflowProfile(QStringLiteral("ULTRALYTICS_YOLO_OBB")));
    QVERIFY(!aitrain::resolveTrainingWorkflowProfile(
        QStringLiteral(" PADDLEOCR_PPOCRV4_REC "), &profile, &error));
    QVERIFY(error.contains(QStringLiteral("未注册")));
    QVERIFY(!aitrain::resolveTrainingWorkflowProfile(
        QStringLiteral(" ULTRALYTICS_YOLO "), &profile, &error));
}

void TrainingWorkflowProfileTests::rejectsUnknownOrMissingBackend()
{
    aitrain::TrainingWorkflowProfile profile;
    QString error;
    QVERIFY(!aitrain::resolveTrainingWorkflowProfile(QString(), &profile, &error));
    QVERIFY(error.contains(QStringLiteral("不能为空")));
    QVERIFY(!aitrain::resolveTrainingWorkflowProfile(QStringLiteral("unknown_backend"), &profile, &error));
    QVERIFY(error.contains(QStringLiteral("未注册")));
    QVERIFY(!aitrain::resolveTrainingWorkflowProfile(QStringLiteral("ultralytics_yolo_detect"), nullptr, &error));
    QVERIFY(error.contains(QStringLiteral("输出对象")));
    QVERIFY(!aitrain::hasTrainingWorkflowProfile(QStringLiteral("unknown")));
}

void TrainingWorkflowProfileTests::registeredProfilesAreCompleteAndUnique()
{
    const QVector<aitrain::TrainingWorkflowProfile>& profiles = aitrain::trainingWorkflowProfiles();
    QCOMPARE(profiles.size(), 8);
    QSet<QString> backends;
    for (const aitrain::TrainingWorkflowProfile& profile : profiles) {
        QVERIFY(!profile.trainingBackend.isEmpty());
        QVERIFY(!backends.contains(profile.trainingBackend));
        backends.insert(profile.trainingBackend);
        QVERIFY(!profile.templateId.isEmpty());
        QVERIFY(!profile.capabilityTaskType.isEmpty());
        QVERIFY(!profile.adapterTaskType.isEmpty());
        QVERIFY(!profile.datasetFormat.isEmpty());
        QVERIFY(!profile.modelFamily.isEmpty());
        QVERIFY(!profile.decoder.isEmpty());
        QVERIFY(profile.artifactFormat == QStringLiteral("onnx")
            || profile.artifactFormat == QStringLiteral("anomalib_bundle")
            || profile.artifactFormat == QStringLiteral("paddleocr_inference_bundle"));
        QVERIFY(!profile.evaluationBackend.isEmpty());
        QVERIFY(!profile.exportBackend.isEmpty());
        QVERIFY(!profile.deploymentBackend.isEmpty());
        QVERIFY(!profile.runtimeRoutes.isEmpty());
        QVERIFY(!profile.limitations.isEmpty());
        QCOMPARE(profile.steps.size(), 8);
        QSet<QString> stepKinds;
        for (const aitrain::TrainingWorkflowStepProfile& step : profile.steps) {
            QVERIFY(!step.kind.isEmpty());
            QVERIFY(!step.backend.isEmpty());
            QVERIFY(!stepKinds.contains(step.kind));
            stepKinds.insert(step.kind);
            if (!step.script.isEmpty()) {
                QVERIFY(!QDir::isAbsolutePath(step.script));
                QVERIFY(!QDir::cleanPath(step.script).startsWith(QStringLiteral("..")));
                QVERIFY(step.script.endsWith(QStringLiteral(".py")));
            }
            for (const QString& candidate : step.artifactCandidates) {
                QVERIFY(!candidate.isEmpty());
                QVERIFY(!QDir::isAbsolutePath(candidate));
                QVERIFY(!QDir::cleanPath(candidate).startsWith(QStringLiteral("..")));
            }
        }
        for (const QString& script : {profile.trainScript, profile.evaluationScript, profile.exportScript}) {
            QVERIFY(!script.isEmpty());
            QVERIFY(!QDir::isAbsolutePath(script));
            QVERIFY(!QDir::cleanPath(script).startsWith(QStringLiteral("..")));
            QVERIFY(script.endsWith(QStringLiteral(".py")));
        }
    }
}

QTEST_MAIN(TrainingWorkflowProfileTests)
#include "tst_training_workflow_profile.moc"
