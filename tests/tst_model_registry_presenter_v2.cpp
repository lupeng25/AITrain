#include "ModelRegistryPresenterV2.h"

#include "aitrain/v2/StorageV2.h"

#include <QDir>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTest>

namespace {

QString databasePath(const QString& projectRoot)
{
    return QDir(projectRoot).filePath(QStringLiteral(".aitrain-v2/project-v2.sqlite"));
}

aitrain::v2::TaskSnapshot createTask(aitrain::v2::StorageV2* storage, QString* error)
{
    aitrain::v2::TaskSnapshot task;
    task.id = aitrain::v2::TaskId::create();
    task.requestId = aitrain::v2::RequestId::create();
    task.capabilityId = QStringLiteral("model.registry.presenter.test");
    task.taskType = QStringLiteral("detection");
    if (!storage || !storage->createTask(task, error)) return {};
    return task;
}

aitrain::v2::ModelPackageSnapshotV2 registerPackage(
    aitrain::v2::StorageV2* storage,
    const aitrain::v2::TaskSnapshot& task,
    const QDateTime& createdAt,
    const QString& family,
    const QStringList& runtimeRoutes,
    const QStringList& limitations,
    const QString& hiddenEntryPath,
    QString* error)
{
    using namespace aitrain::v2;
    const QString sha256(64, family == QStringLiteral("newer_family")
            ? QLatin1Char('b') : QLatin1Char('a'));
    const ArtifactId artifactId = ArtifactId::create();
    if (!storage->recordArtifactWithFiles(artifactId, task.id, QStringLiteral("model_export_v2"),
            {{hiddenEntryPath, sha256, 42}}, createdAt, error)) {
        return {};
    }

    ModelManifestV2 manifest;
    manifest.modelPackageId = ModelPackageId::create();
    manifest.modelFamily = family;
    manifest.taskType = QStringLiteral("detection");
    manifest.sourceBackend = QStringLiteral("ultralytics_yolo_detect");
    manifest.sourceTaskId = task.id;
    manifest.sourceSnapshotId = SnapshotId::create();
    manifest.sourceArtifactSha256 = sha256;
    manifest.artifactEntryPath = hiddenEntryPath;
    manifest.inputs = {{QStringLiteral("images"), QStringLiteral("NCHW"), {1, 3, 640, 640}}};
    manifest.outputs = {{QStringLiteral("output0"), QStringLiteral("NCN"), {1, 84, -1}}};
    manifest.preprocessing = {{QStringLiteral("id"), QStringLiteral("letterbox_rgb_0_1")}};
    manifest.postprocessing = {{QStringLiteral("id"), QStringLiteral("yolo_detection_nms")}};
    manifest.decoder = QStringLiteral("yolo_detection_v8");
    manifest.classNames = QStringList{QStringLiteral("part")};
    manifest.opset = 17;
    manifest.exporterVersion = QStringLiteral("test-exporter-1");
    manifest.runtimeRoutes = runtimeRoutes;
    manifest.limitations = limitations;
    manifest.verified = true;

    ModelPackageSnapshotV2 package{manifest, artifactId, createdAt};
    if (!storage->registerModelPackage(package, error)) return {};
    return package;
}

QStringList allVisibleStrings(const ModelPackageListItemV2& row)
{
    QStringList values{
        row.modelPackageId, row.sourceTaskId, row.sourceSnapshotId,
        row.sourceArtifactId, row.sourceArtifactSha256, row.modelFamily,
        row.taskType, row.sourceBackend, row.artifactFormat, row.decoder,
        row.exporterVersion, row.createdAt
    };
    values.append(row.runtimeRoutes);
    values.append(row.limitations);
    return values;
}

} // namespace

class ModelRegistryPresenterV2Tests final : public QObject {
    Q_OBJECT

private slots:
    void emptyProjectHasEmptyModelRegistry();
    void exposesNewestFirstIdentityLineageRuntimeAndLimitationsWithoutPaths();
    void missingQueryServiceFailsAndClearsRows();
};

void ModelRegistryPresenterV2Tests::emptyProjectHasEmptyModelRegistry()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    aitrain::v2::ProjectQueryServiceV2 query(&workspace);
    ModelRegistryPresenterV2 presenter(&query);
    QSignalSpy changed(&presenter, &ModelRegistryPresenterV2::modelPackagesChanged);

    QVERIFY2(presenter.refresh(), qPrintable(presenter.lastError()));
    QCOMPARE(presenter.objectName(), QStringLiteral("ModelRegistryPresenterV2"));
    QCOMPARE(presenter.modelPackageCount(), 0);
    QCOMPARE(presenter.modelPackages().size(), 0);
    QCOMPARE(changed.count(), 1);
}

void ModelRegistryPresenterV2Tests::exposesNewestFirstIdentityLineageRuntimeAndLimitationsWithoutPaths()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString projectRoot = directory.filePath(QStringLiteral("带 空格 project"));
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(projectRoot, &error), qPrintable(error));
    aitrain::v2::StorageV2 storage;
    QVERIFY2(storage.open(databasePath(projectRoot), &error), qPrintable(error));
    const auto task = createTask(&storage, &error);
    QVERIFY2(task.id.isValid(), qPrintable(error));

    const QString olderHiddenPath = QStringLiteral("private/older/model.onnx");
    const QString newerHiddenPath = QStringLiteral("private/newer/model.onnx");
    const auto older = registerPackage(&storage, task,
        QDateTime::currentDateTimeUtc().addSecs(-10), QStringLiteral("older_family"),
        {QStringLiteral("aitrain_onnxruntime")},
        {QStringLiteral("仅用于回归验证。")}, olderHiddenPath, &error);
    QVERIFY2(older.manifest.modelPackageId.isValid(), qPrintable(error));
    const auto newer = registerPackage(&storage, task,
        QDateTime::currentDateTimeUtc(), QStringLiteral("newer_family"),
        {QStringLiteral("aitrain_onnxruntime"), QStringLiteral("aitrain_ncnn")},
        {QStringLiteral("NCNN 需要显式 blob 合同。"), QStringLiteral("客户域证据待补充。")},
        newerHiddenPath, &error);
    QVERIFY2(newer.manifest.modelPackageId.isValid(), qPrintable(error));

    aitrain::v2::ProjectQueryServiceV2 query(&workspace);
    ModelRegistryPresenterV2 presenter(&query);
    QVERIFY2(presenter.refresh(20), qPrintable(presenter.lastError()));
    QCOMPARE(presenter.modelPackageCount(), 2);
    const QVector<ModelPackageListItemV2>& rows = presenter.modelPackages();
    QCOMPARE(rows.first().modelPackageId, newer.manifest.modelPackageId.toString());
    QCOMPARE(rows.last().modelPackageId, older.manifest.modelPackageId.toString());

    const ModelPackageListItemV2& row = rows.first();
    QCOMPARE(row.sourceTaskId, task.id.toString());
    QCOMPARE(row.sourceSnapshotId, newer.manifest.sourceSnapshotId.toString());
    QCOMPARE(row.sourceArtifactId, newer.sourceArtifactId.toString());
    QCOMPARE(row.sourceArtifactSha256, newer.manifest.sourceArtifactSha256);
    QCOMPARE(row.modelFamily, QStringLiteral("newer_family"));
    QCOMPARE(row.taskType, QStringLiteral("detection"));
    QCOMPARE(row.sourceBackend, QStringLiteral("ultralytics_yolo_detect"));
    QCOMPARE(row.artifactFormat, QStringLiteral("onnx"));
    QCOMPARE(row.decoder, QStringLiteral("yolo_detection_v8"));
    QCOMPARE(row.runtimeRoutes,
        QStringList({QStringLiteral("aitrain_onnxruntime"), QStringLiteral("aitrain_ncnn")}));
    QCOMPARE(row.limitations.size(), 2);
    QVERIFY(row.verified);

    const QString visible = allVisibleStrings(row).join(QLatin1Char('\n'));
    QVERIFY(!visible.contains(QDir::toNativeSeparators(projectRoot)));
    QVERIFY(!visible.contains(projectRoot));
    QVERIFY(!visible.contains(olderHiddenPath));
    QVERIFY(!visible.contains(newerHiddenPath));
    QVERIFY(!visible.contains(QStringLiteral("artifactPath")));
    QVERIFY(!visible.contains(QStringLiteral("modelPath")));
    QVERIFY(!visible.contains(QStringLiteral("checkpointPath")));
    QVERIFY(!visible.contains(QStringLiteral("onnxPath")));
    QVERIFY(!visible.contains(QStringLiteral("reportPath")));
}

void ModelRegistryPresenterV2Tests::missingQueryServiceFailsAndClearsRows()
{
    ModelRegistryPresenterV2 presenter(nullptr);
    QSignalSpy failed(&presenter, &ModelRegistryPresenterV2::queryFailed);
    QVERIFY(!presenter.refresh());
    QCOMPARE(presenter.modelPackageCount(), 0);
    QVERIFY(!presenter.lastError().isEmpty());
    QCOMPARE(failed.count(), 1);
}

QTEST_GUILESS_MAIN(ModelRegistryPresenterV2Tests)

#include "tst_model_registry_presenter_v2.moc"
