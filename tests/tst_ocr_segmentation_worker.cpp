#include "TestSupport.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/protocol/Protocol.h"
#include "aitrain/workflow/ProjectWorkspace.h"
#include "aitrain/storage/ProjectStore.h"

#include <QCoreApplication>
#include <QEventLoop>
#include <QLocalServer>
#include <QLocalSocket>
#include <QProcess>
#include <QTimer>
#include <QUuid>

#include <algorithm>
#include <utility>

namespace {

bool startTaskFromPayload(
    WorkerClient& client,
    const QString& workerProgram,
    const QString& commandType,
    const QJsonObject& payload,
    QString* error)
{
    aitrain::worker_protocol::TaskCommand command;
    if (!aitrain::worker_protocol::taskCommandFromPayload(commandType, payload, &command, error)) {
        return false;
    }
    return client.startTask(workerProgram, command, error);
}

template <typename Receiver, typename Callback>
QMetaObject::Connection connectWorkerEvents(
    WorkerClient* client, Receiver* receiver, Callback&& callback)
{
    return QObject::connect(client, &WorkerClient::taskEventReceived, receiver,
        [callback = std::forward<Callback>(callback)](
            const aitrain::worker_protocol::TaskEvent& event) mutable {
            callback(aitrain::worker_protocol::taskEventType(event), event.details);
        });
}

bool writeFakeSmpWorkflowAdapters(const QString& root);
bool writeFakeAnomalibWorkflowAdapters(const QString& root);

bool initializeWorkerProject(const QString& projectRoot, QString* error)
{
    aitrain::ProjectWorkspace workspace;
    if (!workspace.createProject(projectRoot, error)) return false;
    workspace.close();
    return true;
}

class ScopedEnvironment final {
public:
    ~ScopedEnvironment()
    {
        for (auto it = entries_.crbegin(); it != entries_.crend(); ++it) {
            if (it->existed) qputenv(it->name.constData(), it->value);
            else qunsetenv(it->name.constData());
        }
    }

    void set(const QByteArray& name, const QString& value)
    {
        const auto existing = std::find_if(entries_.cbegin(), entries_.cend(),
            [&name](const Entry& entry) { return entry.name == name; });
        if (existing == entries_.cend()) {
            entries_.append({name, qEnvironmentVariableIsSet(name.constData()), qgetenv(name.constData())});
        }
        if (value.isEmpty()) qunsetenv(name.constData());
        else qputenv(name.constData(), value.toLocal8Bit());
    }

private:
    struct Entry final {
        QByteArray name;
        bool existed = false;
        QByteArray value;
    };
    QVector<Entry> entries_;
};

aitrain::DatasetSnapshotArtifactBundle commitTrainingSnapshotFixture(
    const QString& projectRoot,
    const QString& datasetRoot,
    const QString& datasetFormat,
    QString* error)
{
    aitrain::ProjectWorkspace workspace;
    aitrain::DatasetSnapshotArtifactBundle snapshot;
    aitrain::TaskSnapshot task;
    const aitrain::TaskId producerTaskId = aitrain::TaskId::create();
    aitrain::DatasetSnapshotCommitRequest request;
    request.datasetRoot = datasetRoot;
    request.datasetFormat = datasetFormat;
    request.driverId = QStringLiteral("worker-training-fixture.%1").arg(datasetFormat);
    request.driverVersion = QStringLiteral("2");
    const QString databasePath = QDir(projectRoot)
        .filePath(QStringLiteral(".aitrain/project.sqlite"));
    const bool opened = QFileInfo::exists(databasePath)
        ? workspace.open(projectRoot, error)
        : workspace.createProject(projectRoot, error);
    if (!opened
        || !workspace.startTask(producerTaskId, QStringLiteral("dataset.snapshot"),
            QStringLiteral("dataset_snapshot"), &task, error)
        || !workspace.commitDatasetSnapshot(producerTaskId, request, &snapshot, error)
        || !workspace.finalizeTask(producerTaskId, aitrain::TaskState::Succeeded, {}, error)) {
        return {};
    }
    return snapshot;
}

QJsonObject trainingSnapshotIdentity(const aitrain::DatasetSnapshotArtifactBundle& snapshot)
{
    return {
        {QStringLiteral("datasetId"), snapshot.snapshot.datasetId.toString()},
        {QStringLiteral("datasetVersionId"), snapshot.snapshot.datasetVersionId.toString()},
        {QStringLiteral("snapshotId"), snapshot.snapshot.id.toString()},
        {QStringLiteral("snapshotArtifactId"), snapshot.snapshot.artifactId.toString()}
    };
}

QString workerWithLocalTrainersFixture(const QString& root, bool anomaly, QString* error)
{
    const QString sourceWorker = workerExecutablePath();
    const QString runtimeRoot = QDir(root).filePath(QStringLiteral("worker-runtime"));
    const QString worker = QDir(runtimeRoot).filePath(QFileInfo(sourceWorker).fileName());
    const QString trainers = QDir(runtimeRoot).filePath(QStringLiteral("python_trainers"));
    if (!QDir().mkpath(runtimeRoot) || !QFile::copy(sourceWorker, worker)) {
        if (error) *error = QStringLiteral("无法创建隔离 Worker 可执行文件。");
        return {};
    }
    const QDir sourceDirectory = QFileInfo(sourceWorker).absoluteDir();
    for (const QString& runtimeLibrary : {QStringLiteral("onnxruntime.dll"),
             QStringLiteral("onnxruntime_providers_shared.dll")}) {
        const QString source = sourceDirectory.filePath(runtimeLibrary);
        if (QFileInfo::exists(source)
            && !QFile::copy(source, QDir(runtimeRoot).filePath(runtimeLibrary))) {
            if (error) *error = QStringLiteral("无法复制隔离 Worker 的 %1。").arg(runtimeLibrary);
            return {};
        }
    }
    const QDir sourceSqlDrivers(sourceDirectory.filePath(QStringLiteral("sqldrivers")));
    const QStringList sqlPlugins = sourceSqlDrivers.entryList(
        {QStringLiteral("qsqlite*.dll")}, QDir::Files);
    if (!sqlPlugins.isEmpty()) {
        const QDir targetSqlDrivers(QDir(runtimeRoot).filePath(QStringLiteral("sqldrivers")));
        if (!targetSqlDrivers.mkpath(QStringLiteral("."))) {
            if (error) *error = QStringLiteral("无法创建隔离 Worker 的 SQLite Driver 目录。");
            return {};
        }
        for (const QString& plugin : sqlPlugins) {
            if (!QFile::copy(sourceSqlDrivers.filePath(plugin), targetSqlDrivers.filePath(plugin))) {
                if (error) *error = QStringLiteral("无法复制隔离 Worker 的 SQLite Driver：%1").arg(plugin);
                return {};
            }
        }
    }
    const bool written = anomaly
        ? writeFakeAnomalibWorkflowAdapters(trainers)
        : writeFakeSmpWorkflowAdapters(trainers);
    if (!written) {
        if (error) *error = QStringLiteral("无法创建 Worker 本地 Trainer fixture。");
        return {};
    }
    return worker;
}

QString committedArtifactEventPath(const QString& projectRoot, const QJsonObject& payload)
{
    const QString artifactId = payload.value(QStringLiteral("artifactId")).toString();
    const QString relativePath = payload.value(QStringLiteral("relativePath")).toString();
    if (artifactId.isEmpty() || relativePath.isEmpty()) return {};
    return QDir(projectRoot).filePath(QStringLiteral(".aitrain/artifacts/committed/%1/%2")
        .arg(artifactId, relativePath));
}

QString writeFakePaddleOcrRepo(const QString& root)
{
    QDir repo(root);
    if (!repo.mkpath(QStringLiteral("tools/infer"))
        || !repo.mkpath(QStringLiteral("configs/rec/PP-OCRv5/multi_language"))
        || !repo.mkpath(QStringLiteral("configs/rec/PP-OCRv6"))
        || !repo.mkpath(QStringLiteral("configs/det/PP-OCRv5"))
        || !repo.mkpath(QStringLiteral("configs/det/PP-OCRv6"))
        || !repo.mkpath(QStringLiteral("ppocr/utils/dict"))) {
        return {};
    }
    writeTextFile(repo.filePath(QStringLiteral("tools/train.py")), QStringLiteral(
        "import sys\n"
        "from pathlib import Path\n"
        "value=next((a.split('=',1)[1] for a in sys.argv if a.startswith('Global.save_model_dir=')), '')\n"
        "out=Path(value); out.mkdir(parents=True,exist_ok=True)\n"
        "(out/'best_accuracy.pdparams').write_bytes(b'fake-paddle-checkpoint')\n"
        "(out/'best_accuracy.pdopt').write_bytes(b'fake-paddle-optimizer')\n"
        "print('loss: 0.1, accuracy: 0.95, hmean: 0.90')\n"));
    writeTextFile(repo.filePath(QStringLiteral("tools/eval.py")), QStringLiteral(
        "print('precision: 0.91, recall: 0.92, hmean: 0.915, accuracy: 0.95, norm_edit_dis: 0.96')\n"));
    writeTextFile(repo.filePath(QStringLiteral("tools/export_model.py")), QStringLiteral(
        "import sys\n"
        "from pathlib import Path\n"
        "value=next((a.split('=',1)[1] for a in sys.argv if a.startswith('Global.save_inference_dir=')), '')\n"
        "out=Path(value); out.mkdir(parents=True,exist_ok=True)\n"
        "(out/'inference.pdmodel').write_bytes(b'fake-paddle-graph')\n"
        "(out/'inference.pdiparams').write_bytes(b'fake-paddle-params')\n"
        "(out/'inference.yml').write_text('Global:\\n  model_name: fake\\n',encoding='utf-8')\n"
        "print('official export complete')\n"));
    writeTextFile(repo.filePath(QStringLiteral("tools/infer/predict_det.py")),
        QStringLiteral("print('det prediction: [[0,0,16,16]]')\n"));
    writeTextFile(repo.filePath(QStringLiteral("tools/infer/predict_rec.py")),
        QStringLiteral("print('rec prediction: text, 0.99')\n"));
    writeTextFile(repo.filePath(QStringLiteral("tools/infer/predict_system.py")), QStringLiteral("# fake system predict\n"));
    writeTextFile(repo.filePath(QStringLiteral("ppocr/utils/dict/ppocrv5_dict.txt")), QStringLiteral("a\nb\n1\n2\nz\n"));
    writeTextFile(repo.filePath(QStringLiteral("ppocr/utils/dict/ppocrv5_en_dict.txt")), QStringLiteral("a\nb\n1\n2\nz\n"));
    writeTextFile(repo.filePath(QStringLiteral("ppocr/utils/dict/ppocrv6_tiny_dict.txt")), QStringLiteral("a\nb\n1\n2\nz\n"));
    writeTextFile(repo.filePath(QStringLiteral("ppocr/utils/dict/ppocrv6_dict.txt")), QStringLiteral("a\nb\n1\n2\nz\n"));
    writeTextFile(repo.filePath(QStringLiteral("configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml")),
        QStringLiteral(
            "Global:\n"
            "  model_name: PP-OCRv5_mobile_rec\n"
            "Architecture:\n"
            "  model_type: rec\n"
            "  algorithm: SVTR_LCNet\n"
            "PostProcess:\n"
            "  name: CTCLabelDecode\n"
            "Metric:\n"
            "  name: RecMetric\n"
            "Train:\n"
            "  dataset:\n"
            "    transforms:\n"
            "    - RecConAug:\n"
            "        image_shape: [48, 320, 3]\n"
            "        max_text_length: 25\n"
            "  loader: {}\n"
            "  sampler: {}\n"
            "Eval:\n"
            "  dataset:\n"
            "    transforms:\n"
            "    - RecResizeImg:\n"
            "        image_shape: [3, 48, 320]\n"
            "  loader: {}\n"));
    writeTextFile(repo.filePath(QStringLiteral("configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml")),
        QStringLiteral(
            "Global:\n"
            "  model_name: PP-OCRv5_server_rec\n"
            "Architecture:\n"
            "  model_type: rec\n"
            "  algorithm: SVTR_HGNet\n"
            "PostProcess:\n"
            "  name: CTCLabelDecode\n"
            "Metric:\n"
            "  name: RecMetric\n"
            "Train:\n"
            "  dataset: {}\n"
            "  loader: {}\n"
            "  sampler: {}\n"
            "Eval:\n"
            "  dataset: {}\n"
            "  loader: {}\n"));
    writeTextFile(repo.filePath(QStringLiteral("configs/rec/PP-OCRv5/multi_language/en_PP-OCRv5_mobile_rec.yaml")),
        QStringLiteral(
            "Global:\n"
            "  model_name: en_PP-OCRv5_mobile_rec\n"
            "Architecture:\n"
            "  model_type: rec\n"
            "  algorithm: SVTR_LCNet\n"
            "PostProcess:\n"
            "  name: CTCLabelDecode\n"
            "Metric:\n"
            "  name: RecMetric\n"
            "Train:\n"
            "  dataset: {}\n"
            "  loader: {}\n"
            "  sampler: {}\n"
            "Eval:\n"
            "  dataset: {}\n"
            "  loader: {}\n"));
    writeTextFile(repo.filePath(QStringLiteral("configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml")),
        QStringLiteral(
            "Global:\n"
            "  model_name: PP-OCRv5_mobile_det\n"
            "Architecture:\n"
            "  model_type: det\n"
            "  algorithm: DB\n"
            "PostProcess:\n"
            "  name: DBPostProcess\n"
            "Metric:\n"
            "  name: DetMetric\n"
            "Train:\n"
            "  dataset:\n"
            "    transforms: []\n"
            "  loader: {}\n"
            "Eval:\n"
            "  dataset:\n"
            "    transforms: []\n"
            "  loader: {}\n"));
    writeTextFile(repo.filePath(QStringLiteral("configs/det/PP-OCRv5/PP-OCRv5_server_det.yml")),
        QStringLiteral(
            "Global:\n"
            "  model_name: PP-OCRv5_server_det\n"
            "Architecture:\n"
            "  model_type: det\n"
            "  algorithm: DB\n"
            "PostProcess:\n"
            "  name: DBPostProcess\n"
            "Metric:\n"
            "  name: DetMetric\n"
            "Train:\n"
            "  dataset:\n"
            "    transforms: []\n"
            "  loader: {}\n"
            "Eval:\n"
            "  dataset:\n"
            "    transforms: []\n"
            "  loader: {}\n"));
    const auto writeV6RecConfig = [&repo](const QString& tier, const QString& dictPath) {
        writeTextFile(repo.filePath(QStringLiteral("configs/rec/PP-OCRv6/PP-OCRv6_%1_rec.yml").arg(tier)),
            QStringLiteral(
                "Global:\n"
                "  model_name: PP-OCRv6_%1_rec\n"
                "  character_dict_path: %2\n"
                "Architecture:\n"
                "  model_type: rec\n"
                "  algorithm: SVTR_LCNet\n"
                "PostProcess:\n"
                "  name: CTCLabelDecode\n"
                "Metric:\n"
                "  name: RecMetric\n"
                "Train:\n"
                "  dataset:\n"
                "    transforms:\n"
                "    - RecConAug:\n"
                "        image_shape: [48, 320, 3]\n"
                "        max_text_length: 25\n"
                "  loader: {}\n"
                "  sampler: {}\n"
                "Eval:\n"
                "  dataset:\n"
                "    transforms:\n"
                "    - RecResizeImg:\n"
                "        image_shape: [3, 48, 320]\n"
                "  loader: {}\n").arg(tier, dictPath));
    };
    writeV6RecConfig(QStringLiteral("tiny"), QStringLiteral("ppocr/utils/dict/ppocrv6_tiny_dict.txt"));
    writeV6RecConfig(QStringLiteral("small"), QStringLiteral("ppocr/utils/dict/ppocrv6_dict.txt"));
    writeV6RecConfig(QStringLiteral("medium"), QStringLiteral("ppocr/utils/dict/ppocrv6_dict.txt"));
    const auto writeV6DetConfig = [&repo](const QString& tier) {
        writeTextFile(repo.filePath(QStringLiteral("configs/det/PP-OCRv6/PP-OCRv6_%1_det.yml").arg(tier)),
            QStringLiteral(
                "Global:\n"
                "  model_name: PP-OCRv6_%1_det\n"
                "Architecture:\n"
                "  model_type: det\n"
                "  algorithm: DB\n"
                "PostProcess:\n"
                "  name: DBPostProcess\n"
                "Metric:\n"
                "  name: DetMetric\n"
                "Train:\n"
                "  dataset:\n"
                "    transforms: []\n"
                "  loader: {}\n"
                "Eval:\n"
                "  dataset:\n"
                "    transforms: []\n"
                "  loader: {}\n").arg(tier));
    };
    writeV6DetConfig(QStringLiteral("tiny"));
    writeV6DetConfig(QStringLiteral("small"));
    writeV6DetConfig(QStringLiteral("medium"));
    return repo.absolutePath();
}

bool pythonCanImportModule(const QString& python, const QString& module)
{
    QProcess process;
    process.start(python, QStringList() << QStringLiteral("-c") << QStringLiteral("import %1").arg(module));
    return process.waitForStarted(1000)
        && process.waitForFinished(5000)
        && process.exitStatus() == QProcess::NormalExit
        && process.exitCode() == 0;
}

void writeFakeUltralyticsOnnxExportPackage(const QString& root)
{
    writeTextFile(
        QDir(root).filePath(QStringLiteral("ultralytics/__init__.py")),
        QStringLiteral(
            "from pathlib import Path\n"
            "__version__ = 'test-fixture'\n"
            "\n"
            "class YOLO:\n"
            "    def __init__(self, model):\n"
            "        self.model = str(model)\n"
            "        self.task = 'detect'\n"
            "\n"
            "    def export(self, **kwargs):\n"
            "        import onnx\n"
            "        from onnx import TensorProto, helper\n"
            "        model_path = Path(self.model)\n"
            "        output_path = model_path.with_suffix('.onnx')\n"
            "        output_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "        input_info = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 32, 32])\n"
            "        output_info = helper.make_tensor_value_info('output0', TensorProto.FLOAT, [1, 5, 1])\n"
            "        values = helper.make_tensor('values', TensorProto.FLOAT, [1, 5, 1], [16.0, 16.0, 8.0, 8.0, 0.10])\n"
            "        node = helper.make_node('Constant', inputs=[], outputs=['output0'], value=values)\n"
            "        graph = helper.make_graph([node], 'aitrain_fake_yolo', [input_info], [output_info])\n"
            "        model = helper.make_model(graph, producer_name='aitrain-test', opset_imports=[helper.make_opsetid('', 13)])\n"
            "        model.ir_version = 8\n"
            "        onnx.save(model, output_path)\n"
            "        return str(output_path)\n"));
}

QString repoRelativeFilePath(const QString& relative)
{
    const QString applicationDir = QCoreApplication::applicationDirPath();
    const QStringList candidates = {
        QDir::current().absoluteFilePath(relative),
        QDir(applicationDir).absoluteFilePath(relative),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../%1").arg(relative)),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../%1").arg(relative)),
        QDir(applicationDir).absoluteFilePath(QStringLiteral("../../../%1").arg(relative)),
    };
    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return QFileInfo(candidate).absoluteFilePath();
        }
    }
    return QDir::current().absoluteFilePath(relative);
}

bool writeFakeSmpWorkflowAdapters(const QString& root)
{
    QDir directory(root);
    if (!directory.mkpath(QStringLiteral("semantic_segmentation"))) return false;
    for (const QString& module : {QStringLiteral("adapter_event_channel.py"),
             QStringLiteral("adapter_sdk.py"), QStringLiteral("trainer_protocol.py")}) {
        if (!QFile::copy(repoRelativeFilePath(QStringLiteral("python_trainers/%1").arg(module)),
                directory.filePath(module))) return false;
    }
    if (!QFile::copy(repoRelativeFilePath(QStringLiteral("python_trainers/semantic_segmentation/smp_exporter.py")),
            directory.filePath(QStringLiteral("semantic_segmentation/smp_exporter.py")))) return false;

    const QString common = QStringLiteral(
        "import json, os\n"
        "from pathlib import Path\n"
        "os.sys.path.insert(0,str(Path(__file__).resolve().parents[1]))\n"
        "from adapter_event_channel import event_channel_from_environment\n"
        "from adapter_sdk import AdapterSdk\n"
        "request=json.loads(Path(os.sys.argv[os.sys.argv.index('--request')+1]).read_text(encoding='utf-8'))\n"
        "out=Path(request['outputPath']); out.mkdir(parents=True, exist_ok=True)\n");
    const QString train = common + QStringLiteral(
        "import onnx\n"
        "from onnx import TensorProto, helper\n"
        "assert Path(request['datasetSnapshotManifest']).is_file()\n"
        "checkpoint=out/'best.pt'; checkpoint.write_text('fake smp checkpoint', encoding='utf-8')\n"
        "model_path=out/'best.onnx'\n"
        "inp=helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,32,32])\n"
        "output=helper.make_tensor_value_info('logits',TensorProto.FLOAT,[1,3,32,32])\n"
        "values=[0.0]*(3*32*32)\n"
        "for i in range(32*32): values[2*32*32+i]=1.0\n"
        "tensor=helper.make_tensor('values',TensorProto.FLOAT,[1,3,32,32],values)\n"
        "node=helper.make_node('Constant',inputs=[],outputs=['logits'],value=tensor)\n"
        "model=helper.make_model(helper.make_graph([node],'fake_smp',[inp],[output]),producer_name='aitrain-test',opset_imports=[helper.make_opsetid('',13)])\n"
        "model.ir_version=8; onnx.save(model,model_path)\n"
        "sidecar=out/'semantic_segmentation_sidecar.json'\n"
        "sidecar.write_text(json.dumps({'backend':'smp_semantic_segmentation','modelFamily':'semantic_segmentation','taskType':'semantic_segmentation','datasetFormat':'semantic_segmentation_mask','classNames':['background','part','scratch'],'inputWidth':32,'inputHeight':32,'normalization':{'mean':[0,0,0],'std':[1,1,1],'scale':0.003921568627},'decoder':'smp_semantic_segmentation'}),encoding='utf-8')\n"
        "report=out/'smp_training_report.json'; report.write_text(json.dumps({'ok':True,'backend':'smp_semantic_segmentation','metrics':{'mIoU':1.0}}),encoding='utf-8')\n"
        "channel=event_channel_from_environment(); channel.connect(); sdk=AdapterSdk('smp_semantic_segmentation',event_sink=channel.emit_event)\n"
        "sdk.emit_artifact_candidate('checkpoint',str(checkpoint)); sdk.emit_artifact_candidate('onnx_model',str(model_path)); sdk.emit_artifact_candidate('model_sidecar',str(sidecar)); sdk.emit_artifact_candidate('training_report',str(report)); sdk.emit_completed('fake SMP train completed'); channel.close()\n");
    const QString evaluate = common + QStringLiteral(
        "model=Path(request['modelPath']); sidecar=Path(request['sidecarPath']); checkpoint=Path(request.get('checkpointPath',''))\n"
        "assert model.is_file() and sidecar.is_file() and Path(request['datasetSnapshotManifest']).is_file()\n"
        "report=out/'evaluation_report.json'; report.write_text(json.dumps({'ok':True,'taskType':'semantic_segmentation','datasetFormat':'semantic_segmentation_mask','classNames':['background','part','scratch'],'metrics':{'mIoU':1.0,'meanDice':1.0,'pixelAccuracy':1.0}}),encoding='utf-8')\n"
        "channel=event_channel_from_environment(); channel.connect(); sdk=AdapterSdk('smp_semantic_segmentation_eval',event_sink=channel.emit_event)\n"
        "sdk.emit_artifact_candidate('onnx_model',str(model)); sdk.emit_artifact_candidate('model_sidecar',str(sidecar));\n"
        "if checkpoint.is_file(): sdk.emit_artifact_candidate('checkpoint',str(checkpoint))\n"
        "sdk.emit_artifact_candidate('evaluation_report',str(report)); sdk.emit_completed('fake SMP evaluation completed'); channel.close()\n");
    writeTextFile(directory.filePath(QStringLiteral("semantic_segmentation/smp_trainer.py")), train);
    writeTextFile(directory.filePath(QStringLiteral("semantic_segmentation/smp_evaluator.py")), evaluate);
    return true;
}

bool writeFakeAnomalibWorkflowAdapters(const QString& root)
{
    QDir directory(root);
    if (!directory.mkpath(QStringLiteral("anomaly"))) return false;
    for (const QString& module : {QStringLiteral("adapter_event_channel.py"),
             QStringLiteral("adapter_sdk.py"), QStringLiteral("trainer_protocol.py")}) {
        if (!QFile::copy(repoRelativeFilePath(QStringLiteral("python_trainers/%1").arg(module)),
                directory.filePath(module))) return false;
    }
    if (!QFile::copy(repoRelativeFilePath(QStringLiteral("python_trainers/anomaly/anomalib_exporter.py")),
            directory.filePath(QStringLiteral("anomaly/anomalib_exporter.py")))) return false;
    const QString script = QStringLiteral(
        "import json, os, shutil\n"
        "from pathlib import Path\n"
        "os.sys.path.insert(0,str(Path(__file__).resolve().parents[1]))\n"
        "from adapter_event_channel import event_channel_from_environment\n"
        "from adapter_sdk import AdapterSdk\n"
        "request=json.loads(Path(os.sys.argv[os.sys.argv.index('--request')+1]).read_text(encoding='utf-8'))\n"
        "out=Path(request['outputPath']); out.mkdir(parents=True,exist_ok=True)\n"
        "backend=request['backend']; mode=request.get('mode','train')\n"
        "channel=event_channel_from_environment(); channel.connect(); sdk=AdapterSdk(backend,event_sink=channel.emit_event)\n"
        "if mode=='train':\n"
        " assert Path(request['datasetSnapshotManifest']).is_file()\n"
        " checkpoint=out/'model.ckpt'; checkpoint.write_bytes(b'fake anomalib checkpoint')\n"
        " report=out/'anomalib_training_report.json'; report.write_text(json.dumps({'ok':True,'taskType':'anomaly_detection','runtime':'anomalib_python'}),encoding='utf-8')\n"
        " sidecar=out/'anomaly_sidecar.json'; sidecar.write_text(json.dumps({'schemaVersion':1,'kind':'anomaly_sidecar','trainingBackend':backend,'taskType':'anomaly_detection','modelFamily':'anomaly_detection','runtime':'anomalib_python','checkpointPath':str(checkpoint),'threshold':0.5,'parameters':{}}),encoding='utf-8')\n"
        " sdk.emit_artifact_candidate('training_report',str(report)); sdk.emit_artifact_candidate('anomaly_sidecar',str(sidecar)); sdk.emit_artifact_candidate('checkpoint',str(checkpoint)); sdk.emit_completed('fake Anomalib train completed')\n"
        "elif mode=='evaluate':\n"
        " sidecar=Path(request['modelPath']); checkpoint=Path(request.get('checkpointPath') or json.loads(sidecar.read_text(encoding='utf-8'))['checkpointPath'])\n"
        " assert sidecar.is_file() and checkpoint.is_file() and Path(request['datasetSnapshotManifest']).is_file()\n"
        " report=out/'evaluation_report.json'; report.write_text(json.dumps({'ok':True,'status':'completed','taskType':'anomaly_detection','runtime':'anomalib_python','metrics':{'imageAUROC':1.0}}),encoding='utf-8')\n"
        " sdk.emit_artifact_candidate('evaluation_report',str(report)); sdk.emit_artifact_candidate('anomaly_sidecar',str(sidecar)); sdk.emit_artifact_candidate('checkpoint',str(checkpoint)); sdk.emit_completed('fake Anomalib evaluation completed')\n"
        "elif mode=='infer':\n"
        " sidecar=Path(request['modelPath']); checkpoint=Path(request['checkpointPath']); image=Path(request['imagePath']); assert sidecar.is_file() and checkpoint.is_file() and image.is_file()\n"
        " report=out/'deployment_validation_report.json'; report.write_text(json.dumps({'schemaVersion':2,'status':'passed','taskType':'anomaly_detection','runtime':'anomalib_python','predictions':[{'anomalyScore':0.1,'threshold':0.5,'decision':'ok'}]}),encoding='utf-8')\n"
        " predictions=out/'deployment_predictions.json'; shutil.copy2(report,predictions)\n"
        " heatmap=out/'anomaly_heatmap.png'; overlay=out/'anomaly_overlay.png'; mask=out/'anomaly_mask.png'\n"
        " shutil.copy2(image,heatmap); shutil.copy2(image,overlay); shutil.copy2(image,mask)\n"
        " sdk.emit_artifact_candidate('deployment_validation_report',str(report)); sdk.emit_artifact_candidate('deployment_predictions',str(predictions)); sdk.emit_artifact_candidate('deployment_heatmap',str(heatmap)); sdk.emit_artifact_candidate('deployment_overlay',str(overlay)); sdk.emit_artifact_candidate('deployment_mask',str(mask)); sdk.emit_completed('fake Anomalib deployment validation completed')\n"
        "else: raise RuntimeError('unsupported fake mode: '+mode)\n"
        "channel.close()\n");
    writeTextFile(directory.filePath(QStringLiteral("anomaly/anomalib_adapter.py")), script);
    return QFileInfo::exists(directory.filePath(QStringLiteral("anomaly/anomalib_adapter.py")));
}

QString createRuntimeDeliveryModel(const QString& projectRoot, const QString& fixtureRoot, QString* error)
{
    const QString python = pythonExecutablePath();
    if (python.isEmpty()) {
        if (error) *error = QStringLiteral("Python 不可用，无法生成 ONNX Runtime 测试模型。");
        return {};
    }
    const QString modelPath = QDir(fixtureRoot).filePath(QStringLiteral("runtime-delivery.onnx"));
    const QString script = QStringLiteral(
        "import sys, onnx\n"
        "from onnx import TensorProto, helper\n"
        "inp=helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,32,32])\n"
        "out=helper.make_tensor_value_info('output0',TensorProto.FLOAT,[1,5,1])\n"
        "tensor=helper.make_tensor('values',TensorProto.FLOAT,[1,5,1],[16,16,8,8,0.9])\n"
        "node=helper.make_node('Constant',[],['output0'],value=tensor)\n"
        "graph=helper.make_graph([node],'runtime_delivery_fixture',[inp],[out])\n"
        "model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)])\n"
        "model.ir_version=8\n"
        "onnx.save(model,sys.argv[1])\n");
    QProcess generator;
    generator.start(python, {QStringLiteral("-c"), script, modelPath});
    if (!generator.waitForFinished(30000) || generator.exitCode() != 0 || !QFileInfo::exists(modelPath)) {
        if (error) *error = QStringLiteral("生成 ONNX Runtime 测试模型失败：%1")
            .arg(QString::fromUtf8(generator.readAllStandardError()));
        return {};
    }

    aitrain::ProjectWorkspace workspace;
    if (!workspace.createProject(projectRoot, error)) return {};
    aitrain::ModelImportRequest request;
    request.taskId = aitrain::TaskId::create();
    request.sourceFilePath = modelPath;
    request.manifest.modelPackageId = aitrain::ModelPackageId::create();
    request.manifest.modelFamily = QStringLiteral("yolo_detection");
    request.manifest.taskType = QStringLiteral("detection");
    request.manifest.sourceBackend = QStringLiteral("runtime_delivery_worker_fixture");
    request.manifest.sourceSnapshotId = aitrain::SnapshotId::create();
    request.manifest.artifactEntryPath = QStringLiteral("model/model.onnx");
    request.manifest.inputs.append({QStringLiteral("images"), QStringLiteral("NCHW"), {1, 3, 32, 32}});
    request.manifest.outputs.append({QStringLiteral("output0"), QStringLiteral("NCN"), {1, 5, 1}});
    request.manifest.preprocessing.insert(QStringLiteral("id"), QStringLiteral("letterbox_rgb_0_1"));
    request.manifest.postprocessing.insert(QStringLiteral("id"), QStringLiteral("yolo_detection_nms"));
    request.manifest.decoder = QStringLiteral("yolo_detection_v8");
    request.manifest.classNames.append(QStringLiteral("part"));
    request.manifest.opset = 13;
    request.manifest.exporterVersion = QStringLiteral("runtime-delivery-worker-fixture");
    request.manifest.runtimeRoutes.append(QStringLiteral("aitrain_onnxruntime"));
    request.manifest.runtimeRoutes.append(QStringLiteral("aitrain_ncnn"));
    request.manifest.verified = true;
    aitrain::ModelImportResult imported;
    if (!workspace.importModel(request, &imported, error)) return {};
    return imported.modelPackage.manifest.modelPackageId.toString();
}

QString createWorkerCocoConversionFixture(const QString& root)
{
    const QString imagePath = QDir(root).filePath(QStringLiteral("images/sample.png"));
    writeTinyPng(imagePath);
    const QJsonObject coco{
        {QStringLiteral("images"), QJsonArray{QJsonObject{{QStringLiteral("id"), 1},
             {QStringLiteral("file_name"), QStringLiteral("images/sample.png")},
             {QStringLiteral("width"), 1}, {QStringLiteral("height"), 1}}}},
        {QStringLiteral("categories"), QJsonArray{QJsonObject{{QStringLiteral("id"), 1},
             {QStringLiteral("name"), QStringLiteral("item")}}}},
        {QStringLiteral("annotations"), QJsonArray{QJsonObject{{QStringLiteral("id"), 1},
             {QStringLiteral("image_id"), 1}, {QStringLiteral("category_id"), 1},
             {QStringLiteral("bbox"), QJsonArray{0.0, 0.0, 1.0, 1.0}}}}}};
    const QString path = QDir(root).filePath(QStringLiteral("annotations.json"));
    writeTextFile(path, QString::fromUtf8(QJsonDocument(coco).toJson(QJsonDocument::Compact)));
    return path;
}

QString createWorkerYoloSnapshotImportFixture(const QString& root)
{
    QDir directory(root);
    directory.mkpath(QStringLiteral("images/train"));
    directory.mkpath(QStringLiteral("images/val"));
    directory.mkpath(QStringLiteral("labels/train"));
    directory.mkpath(QStringLiteral("labels/val"));
    writeTextFile(directory.filePath(QStringLiteral("data.yaml")), QStringLiteral(
        "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [item]\n"));
    writeTinyPng(directory.filePath(QStringLiteral("images/train/a.png")));
    writeTinyPng(directory.filePath(QStringLiteral("images/val/b.png")));
    writeTextFile(directory.filePath(QStringLiteral("labels/train/a.txt")),
        QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
    writeTextFile(directory.filePath(QStringLiteral("labels/val/b.txt")),
        QStringLiteral("0 0.5 0.5 0.25 0.25\n"));
    return root;
}

QString createAnnotationRepairFixture(const QString& projectRoot, const QString& datasetRoot, QString* error,
    aitrain::DatasetSnapshotRecord* snapshotRecord = nullptr)
{
    QDir root(datasetRoot);
    if (!root.mkpath(QStringLiteral("images/train"))
        || !root.mkpath(QStringLiteral("images/val"))
        || !root.mkpath(QStringLiteral("labels/train"))
        || !root.mkpath(QStringLiteral("labels/val"))) {
        if (error) *error = QStringLiteral("无法创建标注 Worker 测试数据目录。");
        return {};
    }
    writeTextFile(root.filePath(QStringLiteral("data.yaml")), QStringLiteral(
        "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [item]\n"));
    writeTinyPng(root.filePath(QStringLiteral("images/train/a.png")));
    writeTinyPng(root.filePath(QStringLiteral("images/val/b.png")));
    writeTextFile(root.filePath(QStringLiteral("labels/train/a.txt")),
        QStringLiteral("0 0.5 0.5 0.05 0.05\n"));
    writeTextFile(root.filePath(QStringLiteral("labels/val/b.txt")),
        QStringLiteral("0 0.5 0.5 0.5 0.5\n"));

    aitrain::ProjectWorkspace workspace;
    if (!workspace.createProject(projectRoot, error)) return {};
    aitrain::TaskSnapshot task;
    const aitrain::TaskId snapshotTaskId = aitrain::TaskId::create();
    if (!workspace.startTask(snapshotTaskId, QStringLiteral("dataset.snapshot"),
            QStringLiteral("dataset_snapshot"), &task, error)) return {};
    aitrain::DatasetSnapshotCommitRequest snapshotRequest;
    snapshotRequest.datasetRoot = datasetRoot;
    snapshotRequest.datasetFormat = QStringLiteral("yolo_detection");
    snapshotRequest.driverId = QStringLiteral("yolo_detection");
    snapshotRequest.driverVersion = QStringLiteral("2.0");
    snapshotRequest.options.classDefinitions = QJsonArray{
        QJsonObject{{QStringLiteral("id"), 0}, {QStringLiteral("name"), QStringLiteral("item")}}};
    aitrain::DatasetSnapshotArtifactBundle snapshot;
    if (!workspace.commitDatasetSnapshot(snapshotTaskId, snapshotRequest, &snapshot, error)
        || !workspace.finalizeTask(snapshotTaskId, aitrain::TaskState::Succeeded, {}, error)) return {};
    if (snapshotRecord) *snapshotRecord = snapshot.snapshot;

    const aitrain::TaskId qualityTaskId = aitrain::TaskId::create();
    if (!workspace.startTask(qualityTaskId, QStringLiteral("dataset.quality"),
            QStringLiteral("dataset_quality"), &task, error)) return {};
    aitrain::DataQualityWorkflowRequest qualityRequest;
    qualityRequest.snapshotId = snapshot.snapshot.id;
    qualityRequest.datasetId = snapshot.snapshot.datasetId;
    qualityRequest.datasetVersionId = snapshot.snapshot.datasetVersionId;
    qualityRequest.snapshotArtifactId = snapshot.snapshot.artifactId;
    aitrain::DataQualityWorkflowResult quality;
    if (!workspace.runDataQualityWorkflow(qualityTaskId, qualityRequest,
            &quality, error)) return {};
    return quality.repairManifestArtifactId.toString();
}

struct OcrAcceptanceWorkerFixture final {
    QString detReportPath;
    QString recReportPath;
    QString systemReportPath;
    QString detSnapshotId;
    QString recSnapshotId;
    QString systemSnapshotId;
};

OcrAcceptanceWorkerFixture createOcrAcceptanceWorkerFixture(const QString& projectRoot,
    const QString& fixtureRoot, bool includeSystemAccuracy, QString* error)
{
    const QString detRoot = QDir(fixtureRoot).filePath(QStringLiteral("ocr-det"));
    const QString recRoot = QDir(fixtureRoot).filePath(QStringLiteral("ocr-rec"));
    writeTinyPng(QDir(detRoot).filePath(QStringLiteral("images/a.png")));
    writeTinyPng(QDir(detRoot).filePath(QStringLiteral("images/b.png")));
    writeTextFile(QDir(detRoot).filePath(QStringLiteral("det_gt_train.txt")), QStringLiteral(
        "images/a.png\t[{\"transcription\":\"A\",\"points\":[[0,0],[4,0],[4,4],[0,4]]}]\n"
        "images/b.png\t[{\"transcription\":\"B\",\"points\":[[0,0],[4,0],[4,4],[0,4]]}]\n"));
    writeTinyPng(QDir(recRoot).filePath(QStringLiteral("images/a.png")));
    writeTinyPng(QDir(recRoot).filePath(QStringLiteral("images/b.png")));
    writeTextFile(QDir(recRoot).filePath(QStringLiteral("rec_gt_train.txt")),
        QStringLiteral("images/a.png\tA\nimages/b.png\tB\n"));

    aitrain::ProjectWorkspace workspace;
    if (!workspace.createProject(projectRoot, error)) return {};
    const auto commitSnapshot = [&](const QString& root, const QString& format,
                                    aitrain::DatasetSnapshotArtifactBundle* snapshot) -> bool {
        const aitrain::TaskId taskId = aitrain::TaskId::create();
        aitrain::TaskSnapshot task;
        aitrain::DatasetSnapshotCommitRequest request;
        request.datasetRoot = root;
        request.datasetFormat = format;
        request.driverId = QStringLiteral("worker.%1").arg(format);
        request.driverVersion = QStringLiteral("2");
        return workspace.startTask(taskId, QStringLiteral("dataset.snapshot"),
                   QStringLiteral("dataset_snapshot"), &task, error)
            && workspace.commitDatasetSnapshot(taskId, request, snapshot, error)
            && workspace.finalizeTask(taskId, aitrain::TaskState::Succeeded, {}, error);
    };
    aitrain::DatasetSnapshotArtifactBundle detSnapshot;
    aitrain::DatasetSnapshotArtifactBundle recSnapshot;
    if (!commitSnapshot(detRoot, QStringLiteral("paddleocr_det"), &detSnapshot)
        || !commitSnapshot(recRoot, QStringLiteral("paddleocr_rec"), &recSnapshot)) return {};

    OcrAcceptanceWorkerFixture fixture;
    fixture.detSnapshotId = detSnapshot.snapshot.id.toString();
    fixture.recSnapshotId = recSnapshot.snapshot.id.toString();
    fixture.systemSnapshotId = detSnapshot.snapshot.id.toString();
    fixture.detReportPath = QDir(fixtureRoot).filePath(QStringLiteral("reports/det.json"));
    fixture.recReportPath = QDir(fixtureRoot).filePath(QStringLiteral("reports/rec.json"));
    fixture.systemReportPath = QDir(fixtureRoot).filePath(QStringLiteral("reports/system.json"));
    writeTextFile(fixture.detReportPath, QString::fromUtf8(QJsonDocument(QJsonObject{
        {QStringLiteral("schemaVersion"), 2},
        {QStringLiteral("backend"), QStringLiteral("paddleocr_det_official_eval")},
        {QStringLiteral("taskType"), QStringLiteral("ocr_detection")},
        {QStringLiteral("component"), QStringLiteral("det")},
        {QStringLiteral("datasetSnapshotManifest"), detSnapshot.manifestPath},
        {QStringLiteral("metrics"), QJsonObject{{QStringLiteral("hmean"), 0.88}}}}
        ).toJson(QJsonDocument::Indented)));
    writeTextFile(fixture.recReportPath, QString::fromUtf8(QJsonDocument(QJsonObject{
        {QStringLiteral("schemaVersion"), 2},
        {QStringLiteral("backend"), QStringLiteral("paddleocr_rec_official_eval")},
        {QStringLiteral("taskType"), QStringLiteral("ocr_recognition")},
        {QStringLiteral("component"), QStringLiteral("rec")},
        {QStringLiteral("datasetSnapshotManifest"), recSnapshot.manifestPath},
        {QStringLiteral("metrics"), QJsonObject{{QStringLiteral("accuracy"), 0.92},
            {QStringLiteral("cer"), 0.08}}}}
        ).toJson(QJsonDocument::Indented)));
    QJsonObject systemMetrics;
    if (includeSystemAccuracy) systemMetrics.insert(QStringLiteral("accuracy"), 0.86);
    writeTextFile(fixture.systemReportPath, QString::fromUtf8(QJsonDocument(QJsonObject{
        {QStringLiteral("ok"), true},
        {QStringLiteral("backend"), QStringLiteral("paddleocr_system_official")},
        {QStringLiteral("framework"), QStringLiteral("PaddleOCR official tools")},
        {QStringLiteral("mode"), QStringLiteral("officialSystemPredict")},
        {QStringLiteral("datasetSnapshotManifest"), detSnapshot.manifestPath},
        {QStringLiteral("predictionCount"), 2},
        {QStringLiteral("metrics"), systemMetrics}}
        ).toJson(QJsonDocument::Indented)));
    return fixture;
}

aitrain::OcrOfficialReportImportResult importOcrFixtureDirect(
    const QString& projectRoot, const OcrAcceptanceWorkerFixture& fixture,
    const QString& evidenceClass, QString* error)
{
    aitrain::ProjectWorkspace workspace;
    aitrain::OcrOfficialReportImportResult result;
    if (!workspace.open(projectRoot, error)) return result;
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    if (!workspace.startTask(taskId, QStringLiteral("paddleocr.official.report.import"),
            QStringLiteral("ocr_official_report_import"), &task, error)) return result;
    aitrain::OcrOfficialReportImportRequest request;
    request.det.reportPath = fixture.detReportPath;
    request.rec.reportPath = fixture.recReportPath;
    request.system.reportPath = fixture.systemReportPath;
    if (!aitrain::SnapshotId::parse(fixture.detSnapshotId, &request.det.datasetSnapshotId, error)
        || !aitrain::SnapshotId::parse(fixture.recSnapshotId, &request.rec.datasetSnapshotId, error)
        || !aitrain::SnapshotId::parse(fixture.systemSnapshotId, &request.system.datasetSnapshotId, error)) return {};
    request.acceptanceCohortId = QStringLiteral("customer-batch-a");
    request.customerDomainId = QStringLiteral("line-a");
    request.evidenceClass = evidenceClass;
    if (!workspace.importOcrOfficialReports(taskId, request, &result, error)
        || !workspace.finalizeTask(taskId, aitrain::TaskState::Succeeded, {}, error)) return {};
    return result;
}

} // namespace

class OcrSegmentationWorkerTests : public QObject {
    Q_OBJECT

private slots:
    void initTestCase()
    {
        qputenv("AITRAIN_ENABLE_DIAGNOSTIC_BACKENDS", "1");
    }

    void removedLegacyModelCommandsAreUnsupported_data()
    {
        QTest::addColumn<QString>("command");
        QTest::newRow("infer") << QStringLiteral("infer");
        QTest::newRow("export-model") << QStringLiteral("exportModel");
        QTest::newRow("evaluate-model") << QStringLiteral("evaluateModel");
        QTest::newRow("benchmark-model") << QStringLiteral("benchmarkModel");
        QTest::newRow("validate-deployment-artifact") << QStringLiteral("validateDeploymentArtifact");
        QTest::newRow("validate-dataset") << QStringLiteral("validateDataset");
        QTest::newRow("curate-dataset") << QStringLiteral("curateDataset");
        QTest::newRow("convert-dataset") << QStringLiteral("convertDataset");
        QTest::newRow("create-dataset-snapshot") << QStringLiteral("createDatasetSnapshot");
        QTest::newRow("split-dataset") << QStringLiteral("splitDataset");
        QTest::newRow("collect-diagnostics") << QStringLiteral("collectDiagnostics");
        QTest::newRow("generate-delivery-report") << QStringLiteral("generateDeliveryReport");
        QTest::newRow("environment-check") << QStringLiteral("environmentCheck");
    }

    void removedLegacyModelCommandsAreUnsupported()
    {
        QFETCH(QString, command);
        namespace wp = aitrain::worker_protocol;

        const QString serverName = QStringLiteral("aitrain_removed_command_%1")
            .arg(QUuid::createUuid().toString(QUuid::Id128));
        QLocalServer::removeServer(serverName);
        QLocalServer server;
        QVERIFY2(server.listen(serverName), qPrintable(server.errorString()));

        QProcess process;
        const aitrain::RequestId controlRequestId = aitrain::RequestId::create();
        const aitrain::TaskId controlTaskId = aitrain::TaskId::create();
        const QString controlToken = QUuid::createUuid().toString(QUuid::Id128);
        aitrain::ProtocolSequenceTracker eventTracker;
        QLocalSocket* socket = nullptr;
        QByteArray buffer;
        QStringList terminalTypes;
        QString terminalMessage;
        QEventLoop loop;
        QTimer timeout;
        timeout.setSingleShot(true);
        connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
        connect(&process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            &loop, &QEventLoop::quit);
        connect(&server, &QLocalServer::newConnection, this, [&]() {
            socket = server.nextPendingConnection();
            connect(socket, &QLocalSocket::readyRead, this, [&]() {
                buffer.append(socket->readAll());
                int newline = buffer.indexOf('\n');
                while (newline >= 0) {
                    const QByteArray line = buffer.left(newline + 1);
                    buffer.remove(0, newline + 1);
                    aitrain::ProtocolEnvelope envelope;
                    QString error;
                    QVERIFY2(aitrain::decodeProtocolMessage(line, &envelope, &error), qPrintable(error));
                    QVERIFY2(eventTracker.observe(envelope, controlRequestId, controlTaskId, &error), qPrintable(error));
                    wp::TaskEvent decodedEvent;
                    QVERIFY2(wp::control::unpackTaskEvent(envelope, &decodedEvent, &error), qPrintable(error));
                    const QString type = wp::taskEventType(decodedEvent);
                    const QJsonObject payload = decodedEvent.details;
                    if (type == wp::event::ready()) {
                        aitrain::ProtocolEnvelope start;
                        start.messageId = aitrain::MessageId::create();
                        start.requestId = controlRequestId;
                        start.taskId = controlTaskId;
                        start.controlToken = controlToken;
                        start.sequence = 1;
                        start.kind = QStringLiteral("command.start_task");
                        start.timestamp = QDateTime::currentDateTimeUtc();
                        start.payload = QJsonObject{
                            {QStringLiteral("schema"), QStringLiteral("aitrain.task-command.typed")},
                            {QStringLiteral("type"), command},
                            {QStringLiteral("taskId"), controlTaskId.toString()},
                            {QStringLiteral("projectRoot"), QDir::tempPath()}};
                        socket->write(aitrain::encodeProtocolMessage(start, &error));
                        socket->flush();
                    }
                    if (wp::isTerminalEvent(type)) {
                        terminalTypes.append(type);
                        terminalMessage = payload.value(wp::field::message()).toString();
                    }
                    newline = buffer.indexOf('\n');
                }
            });
        });

        process.setProgram(workerExecutablePath());
        process.setArguments(QStringList()
            << QStringLiteral("--server") << serverName
            << QStringLiteral("--request-id") << controlRequestId.toString()
            << QStringLiteral("--task-id") << controlTaskId.toString()
            << QStringLiteral("--control-token") << controlToken);
        process.start();
        timeout.start(10000);
        loop.exec();
        if (process.state() != QProcess::NotRunning) {
            process.kill();
            QVERIFY(process.waitForFinished(5000));
        }
        QCOMPARE(terminalTypes.size(), 1);
        QCOMPARE(terminalTypes.constFirst(), wp::event::failed());
        QVERIFY2(terminalMessage.contains(QStringLiteral("Unsupported task command type: %1").arg(command)),
            qPrintable(terminalMessage));
    }

    void workerControlRejectsInvalidEnvelope_data()
    {
        QTest::addColumn<QString>("scenario");
        QTest::newRow("duplicate-message") << QStringLiteral("duplicate");
        QTest::newRow("out-of-order-sequence") << QStringLiteral("out-of-order");
        QTest::newRow("cross-request") << QStringLiteral("cross-request");
        QTest::newRow("unknown-kind") << QStringLiteral("unknown-kind");
        QTest::newRow("oversized-frame") << QStringLiteral("oversized");
        QTest::newRow("cancel-before-start") << QStringLiteral("cancel-before-start");
        QTest::newRow("wrong-token") << QStringLiteral("wrong-token");
    }

    void workerControlRejectsInvalidEnvelope()
    {
        QFETCH(QString, scenario);
        namespace wp = aitrain::worker_protocol;

        const QString serverName = QStringLiteral("aitrain_invalid_%1")
            .arg(QUuid::createUuid().toString(QUuid::Id128));
        QLocalServer::removeServer(serverName);
        QLocalServer server;
        QVERIFY2(server.listen(serverName), qPrintable(server.errorString()));

        const aitrain::RequestId requestId = aitrain::RequestId::create();
        const aitrain::TaskId taskId = aitrain::TaskId::create();
        const QString controlToken = QUuid::createUuid().toString(QUuid::Id128);
        aitrain::ProtocolSequenceTracker eventTracker;
        QProcess process;
        QLocalSocket* socket = nullptr;
        QByteArray buffer;
        int terminalCount = 0;
        QString terminalType;
        QString terminalErrorCode;
        QEventLoop loop;
        QTimer timeout;
        timeout.setSingleShot(true);
        connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
        connect(&process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            &loop, &QEventLoop::quit);
        connect(&server, &QLocalServer::newConnection, this, [&]() {
            socket = server.nextPendingConnection();
            connect(socket, &QLocalSocket::readyRead, this, [&]() {
                buffer.append(socket->readAll());
                int newline = buffer.indexOf('\n');
                while (newline >= 0) {
                    const QByteArray line = buffer.left(newline + 1);
                    buffer.remove(0, newline + 1);
                    aitrain::ProtocolEnvelope envelope;
                    QString error;
                    QVERIFY2(aitrain::decodeProtocolMessage(line, &envelope, &error), qPrintable(error));
                    QVERIFY2(eventTracker.observe(envelope, requestId, taskId, &error), qPrintable(error));
                     wp::TaskEvent decodedEvent;
                     QVERIFY2(wp::control::unpackTaskEvent(envelope, &decodedEvent, &error), qPrintable(error));
                     const QString type = wp::taskEventType(decodedEvent);
                     const QJsonObject payload = decodedEvent.details;
                    if (type == wp::event::ready()) {
                        QByteArray invalidBytes;
                        if (scenario == QStringLiteral("duplicate")) {
                            const QJsonObject conversionPayload{
                                {wp::field::taskId(), taskId.toString()},
                                {QStringLiteral("projectRoot"), QDir::tempPath()},
                                {wp::field::sourcePath(), QDir::tempPath()},
                                {wp::field::sourceFormat(), QStringLiteral("coco_detection")},
                                {wp::field::targetFormat(), QStringLiteral("yolo_detection")},
                                {QStringLiteral("targetDatasetId"), aitrain::DatasetId::create().toString()},
                                {QStringLiteral("targetDatasetName"), QStringLiteral("invalid-control")},
                                {wp::field::options(), QJsonObject{}}};
                            wp::TaskCommand conversionCommand;
                            QVERIFY2(wp::taskCommandFromPayload(
                                wp::command::runDatasetConversionWorkflow(), conversionPayload,
                                &conversionCommand, &error), qPrintable(error));
                            const QByteArray start = aitrain::encodeProtocolMessage(
                                wp::control::startTaskEnvelope(requestId, taskId, 1,
                                    conversionCommand, controlToken), &error);
                            invalidBytes = start + start;
                        } else if (scenario == QStringLiteral("out-of-order")) {
                            const QJsonObject conversionPayload{
                                {wp::field::taskId(), taskId.toString()},
                                {QStringLiteral("projectRoot"), QDir::tempPath()},
                                {wp::field::sourcePath(), QDir::tempPath()},
                                {wp::field::sourceFormat(), QStringLiteral("coco_detection")},
                                {wp::field::targetFormat(), QStringLiteral("yolo_detection")},
                                {QStringLiteral("targetDatasetId"), aitrain::DatasetId::create().toString()},
                                {QStringLiteral("targetDatasetName"), QStringLiteral("invalid-control")},
                                {wp::field::options(), QJsonObject{}}};
                            wp::TaskCommand conversionCommand;
                            QVERIFY2(wp::taskCommandFromPayload(
                                wp::command::runDatasetConversionWorkflow(), conversionPayload,
                                &conversionCommand, &error), qPrintable(error));
                            invalidBytes = aitrain::encodeProtocolMessage(
                                wp::control::startTaskEnvelope(requestId, taskId, 2,
                                    conversionCommand, controlToken), &error);
                            invalidBytes += aitrain::encodeProtocolMessage(
                                wp::control::startTaskEnvelope(requestId, taskId, 1,
                                    conversionCommand, controlToken), &error);
                        } else if (scenario == QStringLiteral("cross-request")) {
                            invalidBytes = aitrain::encodeProtocolMessage(
                                wp::control::cancelTaskEnvelope(
                                    aitrain::RequestId::create(), taskId, 1, controlToken), &error);
                        } else if (scenario == QStringLiteral("unknown-kind")) {
                            const QByteArray valid = aitrain::encodeProtocolMessage(
                                wp::control::cancelTaskEnvelope(requestId, taskId, 1, controlToken), &error);
                            QJsonObject object = QJsonDocument::fromJson(valid.trimmed()).object();
                            object.insert(QStringLiteral("kind"), QStringLiteral("command.unknown"));
                            invalidBytes = QJsonDocument(object).toJson(QJsonDocument::Compact) + '\n';
                        } else if (scenario == QStringLiteral("oversized")) {
                            invalidBytes = QByteArray(
                                aitrain::kProtocolMaxControlMessageBytes + 1, 'x');
                            invalidBytes.append('\n');
                        } else if (scenario == QStringLiteral("wrong-token")) {
                            const QJsonObject conversionPayload{
                                {wp::field::taskId(), taskId.toString()},
                                {QStringLiteral("projectRoot"), QDir::tempPath()},
                                {wp::field::sourcePath(), QDir::tempPath()},
                                {wp::field::sourceFormat(), QStringLiteral("coco_detection")},
                                {wp::field::targetFormat(), QStringLiteral("yolo_detection")},
                                {QStringLiteral("targetDatasetId"), aitrain::DatasetId::create().toString()},
                                {QStringLiteral("targetDatasetName"), QStringLiteral("invalid-control")},
                                {wp::field::options(), QJsonObject{}}};
                            wp::TaskCommand conversionCommand;
                            QVERIFY2(wp::taskCommandFromPayload(
                                wp::command::runDatasetConversionWorkflow(), conversionPayload,
                                &conversionCommand, &error), qPrintable(error));
                            invalidBytes = aitrain::encodeProtocolMessage(
                                wp::control::startTaskEnvelope(requestId, taskId, 1,
                                    conversionCommand, QStringLiteral("wrong-token")), &error);
                        } else {
                            invalidBytes = aitrain::encodeProtocolMessage(
                                wp::control::cancelTaskEnvelope(requestId, taskId, 1, controlToken), &error);
                        }
                        QVERIFY2(!invalidBytes.isEmpty(), qPrintable(error));
                        socket->write(invalidBytes);
                        socket->flush();
                    }
                    if (wp::isTerminalEvent(type)) {
                        ++terminalCount;
                        terminalType = type;
                        terminalErrorCode = payload.value(wp::field::errorCode()).toString();
                    }
                    newline = buffer.indexOf('\n');
                }
            });
        });

        process.setProgram(workerExecutablePath());
        process.setArguments(QStringList()
            << QStringLiteral("--server") << serverName
            << QStringLiteral("--request-id") << requestId.toString()
            << QStringLiteral("--task-id") << taskId.toString()
            << QStringLiteral("--control-token") << controlToken);
        process.start();
        timeout.start(10000);
        loop.exec();
        if (process.state() != QProcess::NotRunning) {
            process.kill();
            QVERIFY(process.waitForFinished(5000));
        }
        QCOMPARE(terminalCount, 1);
        QCOMPARE(terminalType, wp::event::failed());
        QCOMPARE(terminalErrorCode, QStringLiteral("protocol_rejected"));
    }

    void modelImportRejectsPayloadTaskIdDifferentFromControlTaskId()
    {
        namespace wp = aitrain::worker_protocol;

        const QString serverName = QStringLiteral("aitrain_import_identity_%1")
            .arg(QUuid::createUuid().toString(QUuid::Id128));
        QLocalServer::removeServer(serverName);
        QLocalServer server;
        QVERIFY2(server.listen(serverName), qPrintable(server.errorString()));

        const aitrain::RequestId requestId = aitrain::RequestId::create();
        const aitrain::TaskId controlTaskId = aitrain::TaskId::create();
        const QString controlToken = QUuid::createUuid().toString(QUuid::Id128);
        const QString mismatchedPayloadTaskId = aitrain::TaskId::create().toString();
        aitrain::ProtocolSequenceTracker eventTracker;
        QProcess process;
        QLocalSocket* socket = nullptr;
        QByteArray buffer;
        int terminalCount = 0;
        QString terminalType;
        QString terminalErrorCode;
        QString terminalMessage;
        QEventLoop loop;
        QTimer timeout;
        timeout.setSingleShot(true);
        connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
        connect(&process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            &loop, &QEventLoop::quit);
        connect(&server, &QLocalServer::newConnection, this, [&]() {
            socket = server.nextPendingConnection();
            connect(socket, &QLocalSocket::readyRead, this, [&]() {
                buffer.append(socket->readAll());
                int newline = buffer.indexOf('\n');
                while (newline >= 0) {
                    const QByteArray line = buffer.left(newline + 1);
                    buffer.remove(0, newline + 1);
                    aitrain::ProtocolEnvelope envelope;
                    QString error;
                    QVERIFY2(aitrain::decodeProtocolMessage(line, &envelope, &error), qPrintable(error));
                    QVERIFY2(eventTracker.observe(envelope, requestId, controlTaskId, &error), qPrintable(error));
                    wp::TaskEvent decodedEvent;
                    QVERIFY2(wp::control::unpackTaskEvent(envelope, &decodedEvent, &error), qPrintable(error));
                    const QString type = wp::taskEventType(decodedEvent);
                    const QJsonObject payload = decodedEvent.details;
                    if (type == wp::event::ready()) {
                        const QJsonObject importPayload{
                            {wp::field::taskId(), mismatchedPayloadTaskId},
                            {QStringLiteral("projectRoot"), QDir::tempPath()},
                            {QStringLiteral("sourceFilePath"), QCoreApplication::applicationFilePath()},
                            {QStringLiteral("manifestDraft"), QJsonObject{{QStringLiteral("nonEmpty"), true}}}};
                        wp::TaskCommand importCommand;
                        QVERIFY2(wp::taskCommandFromPayload(
                            wp::command::importModel(), importPayload, &importCommand, &error), qPrintable(error));
                        const aitrain::ProtocolEnvelope start = wp::control::startTaskEnvelope(
                            requestId, controlTaskId, 1, importCommand, controlToken);
                        const QByteArray encoded = aitrain::encodeProtocolMessage(start, &error);
                        QVERIFY2(!encoded.isEmpty(), qPrintable(error));
                        socket->write(encoded);
                        socket->flush();
                    }
                    if (wp::isTerminalEvent(type)) {
                        ++terminalCount;
                        terminalType = type;
                        terminalErrorCode = payload.value(wp::field::errorCode()).toString();
                        terminalMessage = payload.value(wp::field::message()).toString();
                    }
                    newline = buffer.indexOf('\n');
                }
            });
        });

        process.setProgram(workerExecutablePath());
        process.setArguments(QStringList()
            << QStringLiteral("--server") << serverName
            << QStringLiteral("--request-id") << requestId.toString()
            << QStringLiteral("--task-id") << controlTaskId.toString()
            << QStringLiteral("--control-token") << controlToken);
        process.start();
        timeout.start(10000);
        loop.exec();
        if (process.state() != QProcess::NotRunning) {
            process.kill();
            QVERIFY(process.waitForFinished(5000));
        }
        QCOMPARE(terminalCount, 1);
        QCOMPARE(terminalType, wp::event::failed());
        QCOMPARE(terminalErrorCode, QStringLiteral("protocol_rejected"));
        QVERIFY2(terminalMessage.contains(QStringLiteral("taskId")), qPrintable(terminalMessage));
    }

    void workerControlRequiresLaunchIdentity()
    {
        QProcess process;
        process.setProgram(workerExecutablePath());
        process.setProcessChannelMode(QProcess::MergedChannels);
        process.setArguments(QStringList() << QStringLiteral("--server") << QStringLiteral("unused_server"));
        process.start();
        QVERIFY(process.waitForFinished(5000));
        QCOMPARE(process.exitStatus(), QProcess::NormalExit);
        QCOMPARE(process.exitCode(), 2);
    }

    void workerControlRequiresLaunchToken()
    {
        QProcess process;
        process.setProgram(workerExecutablePath());
        process.setProcessChannelMode(QProcess::MergedChannels);
        const aitrain::RequestId requestId = aitrain::RequestId::create();
        const aitrain::TaskId taskId = aitrain::TaskId::create();
        process.setArguments(QStringList()
            << QStringLiteral("--server") << QStringLiteral("unused_server")
            << QStringLiteral("--request-id") << requestId.toString()
            << QStringLiteral("--task-id") << taskId.toString());
        process.start();
        QVERIFY(process.waitForFinished(5000));
        QCOMPARE(process.exitStatus(), QProcess::NormalExit);
        QCOMPARE(process.exitCode(), 2);
    }

    void runtimeDeliveryWorkerSuccessCommitsSixStepsAndEvidence()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString datasetRoot = createWorkerYoloSnapshotImportFixture(
            directory.filePath(QStringLiteral("runtime-sample-dataset")));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        const QString modelPackageId = createRuntimeDeliveryModel(projectRoot, directory.path(), &error);
        if (modelPackageId.isEmpty() && error.contains(QStringLiteral("Python 不可用"))) QSKIP(qPrintable(error));
        QVERIFY2(!modelPackageId.isEmpty(), qPrintable(error));
        const auto sampleSnapshot = commitTrainingSnapshotFixture(
            projectRoot, datasetRoot, QStringLiteral("yolo_detection"), &error);
        QVERIFY2(sampleSnapshot.snapshot.id.isValid(), qPrintable(error));

        WorkerClient client;
        bool finished = false;
        bool ok = false;
        int terminalCount = 0;
        QJsonObject workflowResult;
        QString terminalMessage;
        QStringList logs;
        connect(&client, &WorkerClient::logLine, this,
            [&logs](const QString& line) { logs.append(line); });
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::runtimeDeliveryWorkflow()) workflowResult = payload;
                if (wp::isTerminalEvent(type)) ++terminalCount;
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString& message) {
                terminalMessage = message;
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runRuntimeDeliveryWorkflow(),
            wp::runtimeDeliveryWorkflowRequest(taskId, projectRoot, modelPackageId,
                QStringLiteral("aitrain_onnxruntime"),
                sampleSnapshot.snapshot.datasetId.toString(),
                sampleSnapshot.snapshot.datasetVersionId.toString(),
                sampleSnapshot.snapshot.id.toString(),
                sampleSnapshot.snapshot.artifactId.toString(),
                QStringLiteral("images/train/a.png"), QJsonObject()), &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY2(ok, qPrintable(terminalMessage + QStringLiteral("\n") + logs.join(QStringLiteral("\n"))));
        QCOMPARE(terminalCount, 1);
        QCOMPARE(workflowResult.value(QStringLiteral("state")).toString(), QStringLiteral("succeeded"));
        QCOMPARE(workflowResult.value(QStringLiteral("steps")).toArray().size(), 6);
        QVERIFY(!workflowResult.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        QVERIFY(!workflowResult.contains(QStringLiteral("evidencePath")));
        QVERIFY(!workflowResult.contains(QStringLiteral("runtimeInvocation")));
        aitrain::ProjectStore storage;
        QVERIFY2(storage.open(QDir(projectRoot).filePath(QStringLiteral(".aitrain/project.sqlite")), &error), qPrintable(error));
        aitrain::TaskId parsedTaskId;
        QVERIFY2(aitrain::TaskId::parse(taskId, &parsedTaskId, &error), qPrintable(error));
        aitrain::TaskSnapshot stored;
        QVERIFY2(storage.task(parsedTaskId, &stored, &error), qPrintable(error));
        QCOMPARE(stored.state, aitrain::TaskState::Succeeded);
    }

    void dataQualityWorkerUsesRegisteredIdentitiesAndReturnsOnlyArtifactIds()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString datasetRoot = directory.filePath(QStringLiteral("dataset"));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        aitrain::DatasetSnapshotRecord snapshot;
        QVERIFY2(!createAnnotationRepairFixture(projectRoot, datasetRoot, &error, &snapshot).isEmpty(), qPrintable(error));

        WorkerClient client;
        bool finished = false;
        bool ok = false;
        int terminalCount = 0;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::dataQualityWorkflow()) result = payload;
                if (wp::isTerminalEvent(type)) ++terminalCount;
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runDataQualityWorkflow(),
            wp::dataQualityWorkflowRequest(taskId, projectRoot,
                snapshot.datasetId.toString(), snapshot.datasetVersionId.toString(),
                snapshot.id.toString(), snapshot.artifactId.toString(), QJsonObject()), &error),
            qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(ok);
        QCOMPARE(terminalCount, 1);
        QCOMPARE(result.value(QStringLiteral("state")).toString(), QStringLiteral("succeeded"));
        QVERIFY(!result.value(QStringLiteral("qualityReportArtifactId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("repairManifestArtifactId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        QVERIFY(!result.contains(QStringLiteral("datasetPath")));
        QVERIFY(!result.contains(QStringLiteral("reportPath")));
        QVERIFY(!result.contains(QStringLiteral("outputPath")));
        QVERIFY(!result.contains(QStringLiteral("evidencePath")));
    }

    void datasetConversionWorkerRegistersSnapshotWithoutPathPayload()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString sourcePath = createWorkerCocoConversionFixture(
            directory.filePath(QStringLiteral("coco")));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        QVERIFY2(initializeWorkerProject(projectRoot, &error), qPrintable(error));
        WorkerClient client;
        bool finished = false;
        bool ok = false;
        int terminalCount = 0;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::datasetConversionWorkflow()) result = payload;
                if (wp::isTerminalEvent(type)) ++terminalCount;
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString targetDatasetId = aitrain::DatasetId::create().toString();
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runDatasetConversionWorkflow(),
            wp::datasetConversionWorkflowRequest(taskId, projectRoot, sourcePath,
                QStringLiteral("coco_json"), QStringLiteral("yolo_detection"),
                targetDatasetId, QStringLiteral("导入数据集"), QJsonObject()), &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(ok);
        QCOMPARE(terminalCount, 1);
        QCOMPARE(result.value(QStringLiteral("state")).toString(), QStringLiteral("succeeded"));
        QCOMPARE(result.value(QStringLiteral("datasetId")).toString(), targetDatasetId);
        QVERIFY(!result.value(QStringLiteral("datasetVersionId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("snapshotId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("conversionArtifactId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("snapshotArtifactId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        QVERIFY(!result.contains(QStringLiteral("outputPath")));
        QVERIFY(!result.contains(QStringLiteral("reportPath")));
        QVERIFY(!result.contains(QStringLiteral("artifactPath")));
        QVERIFY(!result.contains(QStringLiteral("evidencePath")));
    }

    void datasetConversionWorkerBackendUnsupportedHasEvidenceAndNoSnapshot()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString sourcePath = createWorkerCocoConversionFixture(
            directory.filePath(QStringLiteral("coco")));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        QVERIFY2(initializeWorkerProject(projectRoot, &error), qPrintable(error));
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        int terminalCount = 0;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::datasetConversionWorkflow()) result = payload;
                if (wp::isTerminalEvent(type)) ++terminalCount;
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runDatasetConversionWorkflow(),
            wp::datasetConversionWorkflowRequest(taskId, projectRoot, sourcePath,
                QStringLiteral("coco_json"), QStringLiteral("voc_xml"),
                aitrain::DatasetId::create().toString(), QStringLiteral("unsupported"),
                QJsonObject()), &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(!ok);
        QCOMPARE(terminalCount, 1);
        QCOMPARE(result.value(QStringLiteral("failureCode")).toString(),
            QStringLiteral("backend_unsupported"));
        QVERIFY(result.value(QStringLiteral("snapshotId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
    }

    void datasetConversionWorkerCancellationHasOneTerminalAndEvidence()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString sourcePath = createWorkerCocoConversionFixture(
            directory.filePath(QStringLiteral("coco")));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        QVERIFY2(initializeWorkerProject(projectRoot, &error), qPrintable(error));
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        int terminalCount = 0;
        QString terminalType;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::datasetConversionWorkflow()) result = payload;
                if (wp::isTerminalEvent(type)) { ++terminalCount; terminalType = type; }
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runDatasetConversionWorkflow(),
            wp::datasetConversionWorkflowRequest(taskId, projectRoot, sourcePath,
                QStringLiteral("coco_json"), QStringLiteral("yolo_detection"),
                aitrain::DatasetId::create().toString(), QStringLiteral("cancel"),
                QJsonObject()), &error), qPrintable(error));
        client.cancel();
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(!ok);
        QCOMPARE(terminalCount, 1);
        QCOMPARE(terminalType, wp::event::canceled());
        QCOMPARE(result.value(QStringLiteral("state")).toString(), QStringLiteral("canceled"));
        QVERIFY(result.value(QStringLiteral("snapshotId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
    }

    void datasetConversionWorkerRejectsInvalidTargetIdentity()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString sourcePath = createWorkerCocoConversionFixture(
            directory.filePath(QStringLiteral("coco")));
        QVERIFY(QDir().mkpath(projectRoot));
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        int terminalCount = 0;
        QString message;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject&) {
                if (wp::isTerminalEvent(type)) ++terminalCount;
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString& valueMessage) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; message = valueMessage; finished = true;
            });
        QString error;
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runDatasetConversionWorkflow(),
            wp::datasetConversionWorkflowRequest(taskId, projectRoot, sourcePath,
                QStringLiteral("coco_json"), QStringLiteral("yolo_detection"),
                QStringLiteral("not-a-uuid"), QStringLiteral("invalid"), QJsonObject()), &error),
            qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(!ok);
        QCOMPARE(terminalCount, 1);
        QVERIFY(message.contains(QStringLiteral("目标 Dataset 身份")));
    }

    void datasetSnapshotImportWorkerRegistersSelfContainedSnapshotWithoutPaths()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString sourcePath = createWorkerYoloSnapshotImportFixture(
            directory.filePath(QStringLiteral("外部 数据集")));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        QVERIFY2(initializeWorkerProject(projectRoot, &error), qPrintable(error));
        WorkerClient client;
        bool finished = false;
        bool ok = false;
        int terminalCount = 0;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::datasetSnapshotImportWorkflow()) result = payload;
                if (wp::isTerminalEvent(type)) ++terminalCount;
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString datasetId = aitrain::DatasetId::create().toString();
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runDatasetSnapshotImportWorkflow(),
            wp::datasetSnapshotImportWorkflowRequest(taskId, projectRoot, sourcePath,
                QStringLiteral("yolo_detection"), datasetId, QStringLiteral("导入快照"),
                QJsonObject{{QStringLiteral("maxFiles"), 20000}}), &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(ok);
        QCOMPARE(terminalCount, 1);
        QCOMPARE(result.value(QStringLiteral("state")).toString(), QStringLiteral("succeeded"));
        QCOMPARE(result.value(QStringLiteral("datasetId")).toString(), datasetId);
        QVERIFY(!result.value(QStringLiteral("datasetVersionId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("snapshotId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("importPlanArtifactId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("snapshotArtifactId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        QVERIFY(!result.contains(QStringLiteral("datasetPath")));
        QVERIFY(!result.contains(QStringLiteral("outputPath")));
        QVERIFY(!result.contains(QStringLiteral("reportPath")));
        QVERIFY(!result.contains(QStringLiteral("artifactPath")));
        QVERIFY(!result.contains(QStringLiteral("evidencePath")));

        aitrain::ProjectStore storage;
        QVERIFY2(storage.open(QDir(projectRoot).filePath(
            QStringLiteral(".aitrain/project.sqlite")), &error), qPrintable(error));
        aitrain::SnapshotId snapshotId;
        QVERIFY(aitrain::SnapshotId::parse(
            result.value(QStringLiteral("snapshotId")).toString(), &snapshotId, &error));
        aitrain::DatasetSnapshotRecord snapshot;
        QVERIFY2(storage.datasetSnapshot(snapshotId, &snapshot, &error), qPrintable(error));
        QVERIFY(QFileInfo(QDir(snapshot.rootPath).filePath(QStringLiteral("images/train/a.png"))).isFile());
        QVERIFY(QFileInfo(QDir(snapshot.rootPath).filePath(QStringLiteral("dataset_snapshot.json"))).isFile());
        QVERIFY(snapshot.rootPath.contains(snapshot.artifactId.toString()));
    }

    void datasetSnapshotImportWorkerCancellationHasEvidenceAndNoSnapshot()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString sourcePath = createWorkerYoloSnapshotImportFixture(
            directory.filePath(QStringLiteral("source")));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        QVERIFY2(initializeWorkerProject(projectRoot, &error), qPrintable(error));
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        int terminalCount = 0;
        QString terminalType;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::datasetSnapshotImportWorkflow()) result = payload;
                if (wp::isTerminalEvent(type)) { ++terminalCount; terminalType = type; }
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runDatasetSnapshotImportWorkflow(),
            wp::datasetSnapshotImportWorkflowRequest(taskId, projectRoot, sourcePath,
                QStringLiteral("yolo_detection"), aitrain::DatasetId::create().toString(),
                QStringLiteral("取消"), QJsonObject()), &error), qPrintable(error));
        client.cancel();
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(!ok);
        QCOMPARE(terminalCount, 1);
        QCOMPARE(terminalType, wp::event::canceled());
        QCOMPARE(result.value(QStringLiteral("state")).toString(), QStringLiteral("canceled"));
        QVERIFY(result.value(QStringLiteral("snapshotId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
    }

    void datasetSplitWorkerReturnsIdsOnlyAndSelfContainedSnapshot()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString sourcePath = createWorkerYoloSnapshotImportFixture(
            directory.filePath(QStringLiteral("source")));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        aitrain::ProjectWorkspace workspace;
        QVERIFY2(workspace.createProject(projectRoot, &error), qPrintable(error));
        const aitrain::TaskId importTaskId = aitrain::TaskId::create();
        aitrain::TaskSnapshot task;
        QVERIFY2(workspace.startTask(importTaskId, QStringLiteral("dataset.snapshot.import"),
            QStringLiteral("dataset_snapshot_import"), &task, &error), qPrintable(error));
        aitrain::DatasetSnapshotImportWorkflowRequest importRequest;
        importRequest.sourcePath = sourcePath;
        importRequest.sourceFormat = QStringLiteral("yolo_detection");
        importRequest.targetDatasetId = aitrain::DatasetId::create();
        importRequest.targetDatasetName = QStringLiteral("源数据集");
        aitrain::DatasetSnapshotImportWorkflowResult imported;
        QVERIFY2(workspace.runDatasetSnapshotImportWorkflow(importTaskId, importRequest,
            &imported, &error), qPrintable(error));
        workspace.close();

        WorkerClient client;
        bool finished = false;
        bool ok = false;
        int terminalCount = 0;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::datasetSplitWorkflow()) result = payload;
                if (wp::isTerminalEvent(type)) ++terminalCount;
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString targetDatasetId = aitrain::DatasetId::create().toString();
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runDatasetSplitWorkflow(),
            wp::datasetSplitWorkflowRequest(taskId, projectRoot,
                imported.datasetSnapshot.datasetId.toString(),
                imported.datasetSnapshot.datasetVersionId.toString(),
                imported.datasetSnapshot.id.toString(), imported.datasetSnapshot.artifactId.toString(),
                targetDatasetId, QStringLiteral("划分目标"),
                QJsonObject{{QStringLiteral("trainRatio"), 0.5},
                    {QStringLiteral("valRatio"), 0.5}, {QStringLiteral("testRatio"), 0.0},
                    {QStringLiteral("seed"), 42}}), &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(ok);
        QCOMPARE(terminalCount, 1);
        QCOMPARE(result.value(QStringLiteral("state")).toString(), QStringLiteral("succeeded"));
        QCOMPARE(result.value(QStringLiteral("datasetId")).toString(), targetDatasetId);
        QVERIFY(!result.value(QStringLiteral("splitPlanArtifactId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("splitArtifactId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("snapshotArtifactId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        QVERIFY(!result.contains(QStringLiteral("sourcePath")));
        QVERIFY(!result.contains(QStringLiteral("datasetPath")));
        QVERIFY(!result.contains(QStringLiteral("outputPath")));
        QVERIFY(!result.contains(QStringLiteral("reportPath")));
        QVERIFY(!result.contains(QStringLiteral("artifactPath")));

        aitrain::ProjectStore storage;
        QVERIFY2(storage.open(QDir(projectRoot).filePath(
            QStringLiteral(".aitrain/project.sqlite")), &error), qPrintable(error));
        aitrain::SnapshotId snapshotId;
        QVERIFY(aitrain::SnapshotId::parse(result.value(QStringLiteral("snapshotId")).toString(),
            &snapshotId, &error));
        aitrain::DatasetSnapshotRecord snapshot;
        QVERIFY2(storage.datasetSnapshot(snapshotId, &snapshot, &error), qPrintable(error));
        QVERIFY(QFileInfo(QDir(snapshot.rootPath).filePath(QStringLiteral("dataset_snapshot.json"))).isFile());
        QVERIFY(!QFileInfo(QDir(snapshot.rootPath).filePath(QStringLiteral("split_plan.json"))).exists());
    }

    void diagnosticsWorkerReturnsIdsOnly()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        QVERIFY2(initializeWorkerProject(projectRoot, &error), qPrintable(error));
        WorkerClient client;
        bool finished = false;
        bool ok = false;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::diagnosticsWorkflow()) result = payload;
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runDiagnosticsWorkflow(),
            wp::diagnosticsWorkflowRequest(taskId, projectRoot,
                QJsonObject{{QStringLiteral("probeTimeoutMs"), 500},
                    {QStringLiteral("probeOutputBytes"), 1024}}), &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(ok);
        QCOMPARE(result.value(QStringLiteral("state")).toString(), QStringLiteral("succeeded"));
        QVERIFY(!result.value(QStringLiteral("workflowRunId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("diagnosticsArtifactId")).toString().isEmpty());
        QVERIFY(!result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        for (const QString& key : {QStringLiteral("path"), QStringLiteral("outputPath"),
                 QStringLiteral("reportPath"), QStringLiteral("bundlePath"),
                 QStringLiteral("manifestPath"), QStringLiteral("artifactPath")}) {
            QVERIFY2(!result.contains(key), qPrintable(key));
        }
    }

    void diagnosticsWorkerCancellationReturnsEvidence()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        QVERIFY2(initializeWorkerProject(projectRoot, &error), qPrintable(error));
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        QString terminalType;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::diagnosticsWorkflow()) result = payload;
                if (wp::isTerminalEvent(type)) terminalType = type;
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runDiagnosticsWorkflow(),
            wp::diagnosticsWorkflowRequest(taskId, projectRoot,
                QJsonObject{{QStringLiteral("probeTimeoutMs"), 500}}), &error), qPrintable(error));
        client.cancel();
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(!ok);
        QCOMPARE(terminalType, wp::event::canceled());
        QCOMPARE(result.value(QStringLiteral("state")).toString(), QStringLiteral("canceled"));
        QVERIFY(!result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        QVERIFY(result.value(QStringLiteral("diagnosticsArtifactId")).toString().isEmpty());
    }

    void dataQualityWorkerIdentityMismatchFailsWithEvidence()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString datasetRoot = directory.filePath(QStringLiteral("dataset"));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        aitrain::DatasetSnapshotRecord snapshot;
        QVERIFY2(!createAnnotationRepairFixture(projectRoot, datasetRoot, &error, &snapshot).isEmpty(), qPrintable(error));
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        int terminalCount = 0;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::dataQualityWorkflow()) result = payload;
                if (wp::isTerminalEvent(type)) ++terminalCount;
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runDataQualityWorkflow(),
            wp::dataQualityWorkflowRequest(taskId, projectRoot,
                aitrain::DatasetId::create().toString(), snapshot.datasetVersionId.toString(),
                snapshot.id.toString(), snapshot.artifactId.toString(), QJsonObject()), &error),
            qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(!ok);
        QCOMPARE(terminalCount, 1);
        QCOMPARE(result.value(QStringLiteral("state")).toString(), QStringLiteral("failed"));
        QCOMPARE(result.value(QStringLiteral("message")).toString(),
            QStringLiteral("quality.snapshot_identity_mismatch"));
        QVERIFY(!result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        QVERIFY(result.value(QStringLiteral("qualityReportArtifactId")).toString().isEmpty());
    }

    void dataQualityWorkerCancellationHasOneTerminalAndEvidence()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString datasetRoot = directory.filePath(QStringLiteral("dataset"));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        aitrain::DatasetSnapshotRecord snapshot;
        QVERIFY2(!createAnnotationRepairFixture(projectRoot, datasetRoot, &error, &snapshot).isEmpty(), qPrintable(error));
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        int terminalCount = 0;
        QString terminalType;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::dataQualityWorkflow()) result = payload;
                if (wp::isTerminalEvent(type)) { ++terminalCount; terminalType = type; }
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runDataQualityWorkflow(),
            wp::dataQualityWorkflowRequest(taskId, projectRoot,
                snapshot.datasetId.toString(), snapshot.datasetVersionId.toString(),
                snapshot.id.toString(), snapshot.artifactId.toString(), QJsonObject()), &error),
            qPrintable(error));
        client.cancel();
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(!ok);
        QCOMPARE(terminalCount, 1);
        QCOMPARE(terminalType, wp::event::canceled());
        QCOMPARE(result.value(QStringLiteral("state")).toString(), QStringLiteral("canceled"));
        QVERIFY(!result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
    }

    void annotationSessionWorkerUsesArtifactIdsAndCreatesNoPathPayload()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString datasetRoot = directory.filePath(QStringLiteral("dataset"));
        const QString workingDirectory = directory.filePath(QStringLiteral("annotation-work"));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        const QString repairArtifactId = createAnnotationRepairFixture(projectRoot, datasetRoot, &error);
        QVERIFY2(!repairArtifactId.isEmpty(), qPrintable(error));

        WorkerClient createClient;
        bool createFinished = false;
        bool createOk = false;
        int createTerminalCount = 0;
        QJsonObject createResult;
        connectWorkerEvents(&createClient, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::annotationSession()) createResult = payload;
                if (wp::isTerminalEvent(type)) ++createTerminalCount;
            });
        connect(&createClient, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                createOk = value == WorkerClient::WorkerTerminalStatus::Succeeded; createFinished = true;
            });
        const QString createTaskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(createClient, workerExecutablePath(),
            wp::command::createAnnotationSession(),
            wp::annotationSessionCreateRequest(createTaskId, projectRoot, repairArtifactId,
                workingDirectory, QJsonObject{{QStringLiteral("tool"), QStringLiteral("X-AnyLabeling")}},
                QJsonObject()), &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(createFinished, 30000);
        QVERIFY(createOk);
        QCOMPARE(createTerminalCount, 1);
        const QString sessionArtifactId = createResult.value(QStringLiteral("sessionArtifactId")).toString();
        QVERIFY(!sessionArtifactId.isEmpty());
        QVERIFY(!createResult.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        QVERIFY(!createResult.contains(QStringLiteral("manifestPath")));
        QVERIFY(!createResult.contains(QStringLiteral("reportPath")));
        QVERIFY(!createResult.contains(QStringLiteral("outputPath")));
        QVERIFY(!createResult.contains(QStringLiteral("workingDirectory")));

        WorkerClient syncClient;
        bool syncFinished = false;
        bool syncOk = false;
        int syncTerminalCount = 0;
        QJsonObject syncResult;
        connectWorkerEvents(&syncClient, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::annotationSync()) syncResult = payload;
                if (wp::isTerminalEvent(type)) ++syncTerminalCount;
            });
        connect(&syncClient, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                syncOk = value == WorkerClient::WorkerTerminalStatus::Succeeded; syncFinished = true;
            });
        const QString syncTaskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(syncClient, workerExecutablePath(),
            wp::command::syncAnnotationSession(),
            wp::annotationSessionSyncRequest(syncTaskId, projectRoot, sessionArtifactId,
                workingDirectory, QJsonObject()), &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(syncFinished, 30000);
        QVERIFY(syncOk);
        QCOMPARE(syncTerminalCount, 1);
        QCOMPARE(syncResult.value(QStringLiteral("status")).toString(), QStringLiteral("no_changes"));
        QVERIFY(!syncResult.value(QStringLiteral("syncReportArtifactId")).toString().isEmpty());
        QVERIFY(!syncResult.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        QVERIFY(!syncResult.value(QStringLiteral("newDatasetVersionCreated")).toBool());
        QVERIFY(!syncResult.contains(QStringLiteral("reportPath")));
        QVERIFY(!syncResult.contains(QStringLiteral("outputPath")));
        QVERIFY(!syncResult.contains(QStringLiteral("workingDirectory")));
    }

    void annotationSessionWorkerCancellationHasOneTerminalAndEvidence()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString datasetRoot = directory.filePath(QStringLiteral("dataset"));
        const QString workingDirectory = directory.filePath(QStringLiteral("annotation-work"));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        const QString repairArtifactId = createAnnotationRepairFixture(projectRoot, datasetRoot, &error);
        QVERIFY2(!repairArtifactId.isEmpty(), qPrintable(error));

        WorkerClient client;
        bool finished = false;
        bool ok = true;
        int terminalCount = 0;
        QString terminalType;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::annotationSession()) result = payload;
                if (wp::isTerminalEvent(type)) { ++terminalCount; terminalType = type; }
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::createAnnotationSession(),
            wp::annotationSessionCreateRequest(taskId, projectRoot, repairArtifactId,
                workingDirectory, QJsonObject(), QJsonObject()), &error), qPrintable(error));
        client.cancel();
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(!ok);
        QCOMPARE(terminalCount, 1);
        QCOMPARE(terminalType, wp::event::canceled());
        QCOMPARE(result.value(QStringLiteral("state")).toString(), QStringLiteral("canceled"));
        QVERIFY(!result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        QVERIFY(result.value(QStringLiteral("sessionArtifactId")).toString().isEmpty());
    }

    void ocrOfficialReportImportAndAcceptanceWorkerUseArtifactBoundary()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        const OcrAcceptanceWorkerFixture fixture = createOcrAcceptanceWorkerFixture(
            projectRoot, directory.path(), true, &error);
        QVERIFY2(!fixture.detSnapshotId.isEmpty(), qPrintable(error));
        const auto source = [](const QString& report, const QString& snapshotId) {
            return QJsonObject{{QStringLiteral("reportPath"), report},
                {QStringLiteral("snapshotId"), snapshotId},
                {QStringLiteral("snapshotArtifactId"), QString()}};
        };

        WorkerClient importClient;
        bool importFinished = false;
        bool importOk = false;
        int importTerminalCount = 0;
        QJsonObject importResult;
        connectWorkerEvents(&importClient, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::ocrOfficialReportsImported()) importResult = payload;
                if (wp::isTerminalEvent(type)) ++importTerminalCount;
            });
        connect(&importClient, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus status, const QString&) {
                importOk = status == WorkerClient::WorkerTerminalStatus::Succeeded; importFinished = true;
            });
        const QString importTaskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(importClient, workerExecutablePath(),
            wp::command::importOcrOfficialReports(),
            wp::ocrOfficialReportImportRequest(importTaskId, projectRoot,
                source(fixture.detReportPath, fixture.detSnapshotId),
                source(fixture.recReportPath, fixture.recSnapshotId),
                source(fixture.systemReportPath, fixture.systemSnapshotId),
                QStringLiteral("customer-batch-a"), QStringLiteral("line-a"),
                QStringLiteral("customer_domain")), &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(importFinished, 30000);
        QVERIFY(importOk);
        QCOMPARE(importTerminalCount, 1);
        const QString detArtifactId = importResult.value(QStringLiteral("detReportArtifactId")).toString();
        const QString recArtifactId = importResult.value(QStringLiteral("recReportArtifactId")).toString();
        const QString systemArtifactId = importResult.value(QStringLiteral("systemReportArtifactId")).toString();
        QVERIFY(!detArtifactId.isEmpty() && !recArtifactId.isEmpty() && !systemArtifactId.isEmpty());
        QVERIFY(!importResult.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        QVERIFY(!importResult.contains(QStringLiteral("reportPath")));
        QVERIFY(!importResult.contains(QStringLiteral("artifactPath")));
        QVERIFY(!importResult.contains(QStringLiteral("outputPath")));

        WorkerClient acceptanceClient;
        bool acceptanceFinished = false;
        bool acceptanceOk = false;
        int acceptanceTerminalCount = 0;
        QJsonObject acceptanceResult;
        connectWorkerEvents(&acceptanceClient, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::ocrAcceptanceWorkflow()) acceptanceResult = payload;
                if (wp::isTerminalEvent(type)) ++acceptanceTerminalCount;
            });
        connect(&acceptanceClient, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus status, const QString&) {
                acceptanceOk = status == WorkerClient::WorkerTerminalStatus::Succeeded; acceptanceFinished = true;
            });
        const QString acceptanceTaskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(acceptanceClient, workerExecutablePath(),
            wp::command::runOcrAcceptanceWorkflow(),
            wp::ocrAcceptanceWorkflowRequest(acceptanceTaskId, projectRoot,
                detArtifactId, recArtifactId, systemArtifactId, QJsonObject()), &error),
            qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(acceptanceFinished, 30000);
        QVERIFY(acceptanceOk);
        QCOMPARE(acceptanceTerminalCount, 1);
        QCOMPARE(acceptanceResult.value(QStringLiteral("state")).toString(), QStringLiteral("succeeded"));
        QVERIFY(acceptanceResult.value(QStringLiteral("productionAccepted")).toBool());
        QVERIFY(!acceptanceResult.value(QStringLiteral("acceptanceReportArtifactId")).toString().isEmpty());
        QVERIFY(!acceptanceResult.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        QVERIFY(!acceptanceResult.contains(QStringLiteral("reportPath")));
        QVERIFY(!acceptanceResult.contains(QStringLiteral("artifactPath")));
        QVERIFY(!acceptanceResult.contains(QStringLiteral("outputPath")));
    }

    void ocrOfficialReportImportMissingSystemAccuracyFailsPreciselyWithoutArtifact()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        const OcrAcceptanceWorkerFixture fixture = createOcrAcceptanceWorkerFixture(
            projectRoot, directory.path(), false, &error);
        QVERIFY2(!fixture.detSnapshotId.isEmpty(), qPrintable(error));
        const auto source = [](const QString& report, const QString& snapshotId) {
            return QJsonObject{{QStringLiteral("reportPath"), report},
                {QStringLiteral("snapshotId"), snapshotId}};
        };
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        int terminalCount = 0;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::ocrOfficialReportsImported()) result = payload;
                if (wp::isTerminalEvent(type)) ++terminalCount;
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::importOcrOfficialReports(),
            wp::ocrOfficialReportImportRequest(taskId, projectRoot,
                source(fixture.detReportPath, fixture.detSnapshotId),
                source(fixture.recReportPath, fixture.recSnapshotId),
                source(fixture.systemReportPath, fixture.systemSnapshotId),
                QStringLiteral("customer-batch-a"), QStringLiteral("line-a"),
                QStringLiteral("customer_domain")), &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(!ok);
        QCOMPARE(terminalCount, 1);
        QVERIFY(result.value(QStringLiteral("message")).toString().startsWith(
            QStringLiteral("ocr_report_import.system_accuracy_unsupported:")));
        QVERIFY(result.value(QStringLiteral("detReportArtifactId")).toString().isEmpty());
        QVERIFY(result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());

        aitrain::ProjectStore storage;
        QVERIFY2(storage.open(QDir(projectRoot).filePath(QStringLiteral(".aitrain/project.sqlite")), &error), qPrintable(error));
        aitrain::TaskId parsedTaskId;
        QVERIFY2(aitrain::TaskId::parse(taskId, &parsedTaskId, &error), qPrintable(error));
        QCOMPARE(storage.artifactsForTask(parsedTaskId, {50, {}}, &error).items.size(), 0);
    }

    void ocrAcceptanceWorkerRejectsPublicEvidenceWithOneTerminalAndEvidence()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        const OcrAcceptanceWorkerFixture fixture = createOcrAcceptanceWorkerFixture(
            projectRoot, directory.path(), true, &error);
        const aitrain::OcrOfficialReportImportResult imported =
            importOcrFixtureDirect(projectRoot, fixture, QStringLiteral("public"), &error);
        QVERIFY2(imported.detReportArtifactId.isValid(), qPrintable(error));
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        int terminalCount = 0;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::ocrAcceptanceWorkflow()) result = payload;
                if (wp::isTerminalEvent(type)) ++terminalCount;
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runOcrAcceptanceWorkflow(),
            wp::ocrAcceptanceWorkflowRequest(taskId, projectRoot,
                imported.detReportArtifactId.toString(), imported.recReportArtifactId.toString(),
                imported.systemReportArtifactId.toString(), QJsonObject()), &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(!ok);
        QCOMPARE(terminalCount, 1);
        QCOMPARE(result.value(QStringLiteral("state")).toString(), QStringLiteral("failed"));
        QVERIFY(!result.value(QStringLiteral("productionAccepted")).toBool());
        QVERIFY(!result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        QVERIFY(result.value(QStringLiteral("message")).toString().contains(
            QStringLiteral("customer_domain")));
    }

    void ocrAcceptanceWorkerCancellationHasOneTerminalAndEvidence()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        const OcrAcceptanceWorkerFixture fixture = createOcrAcceptanceWorkerFixture(
            projectRoot, directory.path(), true, &error);
        const aitrain::OcrOfficialReportImportResult imported =
            importOcrFixtureDirect(projectRoot, fixture, QStringLiteral("customer_domain"), &error);
        QVERIFY2(imported.detReportArtifactId.isValid(), qPrintable(error));
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        int terminalCount = 0;
        QString terminalType;
        QJsonObject result;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::ocrAcceptanceWorkflow()) result = payload;
                if (wp::isTerminalEvent(type)) { ++terminalCount; terminalType = type; }
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runOcrAcceptanceWorkflow(),
            wp::ocrAcceptanceWorkflowRequest(taskId, projectRoot,
                imported.detReportArtifactId.toString(), imported.recReportArtifactId.toString(),
                imported.systemReportArtifactId.toString(), QJsonObject()), &error), qPrintable(error));
        client.cancel();
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(!ok);
        QCOMPARE(terminalCount, 1);
        QCOMPARE(terminalType, wp::event::canceled());
        QCOMPARE(result.value(QStringLiteral("state")).toString(), QStringLiteral("canceled"));
        QVERIFY(!result.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
    }

    void runtimeDeliveryWorkerFailureHasOneTerminalAndEvidence()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        const QString modelPackageId = createRuntimeDeliveryModel(projectRoot, directory.path(), &error);
        if (modelPackageId.isEmpty() && error.contains(QStringLiteral("Python 不可用"))) QSKIP(qPrintable(error));
        QVERIFY2(!modelPackageId.isEmpty(), qPrintable(error));
        const QString datasetRoot = createWorkerYoloSnapshotImportFixture(
            directory.filePath(QStringLiteral("runtime-sample-dataset")));
        const auto sampleSnapshot = commitTrainingSnapshotFixture(
            projectRoot, datasetRoot, QStringLiteral("yolo_detection"), &error);
        QVERIFY2(sampleSnapshot.snapshot.id.isValid(), qPrintable(error));
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        int terminalCount = 0;
        QJsonObject workflowResult;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::runtimeDeliveryWorkflow()) workflowResult = payload;
                if (wp::isTerminalEvent(type)) ++terminalCount;
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runRuntimeDeliveryWorkflow(),
            wp::runtimeDeliveryWorkflowRequest(taskId, projectRoot, modelPackageId,
                QStringLiteral("aitrain_ncnn"),
                sampleSnapshot.snapshot.datasetId.toString(),
                sampleSnapshot.snapshot.datasetVersionId.toString(),
                sampleSnapshot.snapshot.id.toString(),
                sampleSnapshot.snapshot.artifactId.toString(),
                QStringLiteral("images/train/a.png"), QJsonObject()), &error), qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(!ok);
        QCOMPARE(terminalCount, 1);
        QCOMPARE(workflowResult.value(QStringLiteral("state")).toString(), QStringLiteral("failed"));
        QVERIFY(!workflowResult.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
        QCOMPARE(workflowResult.value(QStringLiteral("steps")).toArray().size(), 6);
    }

    void runtimeDeliveryWorkerCancellationHasOneTerminalAndEvidence()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString datasetRoot = createWorkerYoloSnapshotImportFixture(
            directory.filePath(QStringLiteral("runtime-sample-dataset")));
        QVERIFY(QDir().mkpath(projectRoot));
        QString error;
        const QString modelPackageId = createRuntimeDeliveryModel(projectRoot, directory.path(), &error);
        if (modelPackageId.isEmpty() && error.contains(QStringLiteral("Python 不可用"))) QSKIP(qPrintable(error));
        QVERIFY2(!modelPackageId.isEmpty(), qPrintable(error));
        const auto sampleSnapshot = commitTrainingSnapshotFixture(
            projectRoot, datasetRoot, QStringLiteral("yolo_detection"), &error);
        QVERIFY2(sampleSnapshot.snapshot.id.isValid(), qPrintable(error));
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        int terminalCount = 0;
        QJsonObject workflowResult;
        QString terminalType;
        connectWorkerEvents(&client, this,
            [&](const QString& type, const QJsonObject& payload) {
                if (type == wp::event::runtimeDeliveryWorkflow()) workflowResult = payload;
                if (wp::isTerminalEvent(type)) { ++terminalCount; terminalType = type; }
            });
        connect(&client, &WorkerClient::finished, this,
            [&](WorkerClient::WorkerTerminalStatus value, const QString&) {
                ok = value == WorkerClient::WorkerTerminalStatus::Succeeded; finished = true;
            });
        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runRuntimeDeliveryWorkflow(),
            wp::runtimeDeliveryWorkflowRequest(taskId, projectRoot, modelPackageId,
                QStringLiteral("aitrain_onnxruntime"),
                sampleSnapshot.snapshot.datasetId.toString(),
                sampleSnapshot.snapshot.datasetVersionId.toString(),
                sampleSnapshot.snapshot.id.toString(),
                sampleSnapshot.snapshot.artifactId.toString(),
                QStringLiteral("images/train/a.png"), QJsonObject()), &error), qPrintable(error));
        client.cancel();
        QTRY_VERIFY_WITH_TIMEOUT(finished, 30000);
        QVERIFY(!ok);
        QCOMPARE(terminalCount, 1);
        QCOMPARE(terminalType, wp::event::canceled());
        QCOMPARE(workflowResult.value(QStringLiteral("state")).toString(), QStringLiteral("canceled"));
        QVERIFY(!workflowResult.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
    }

    void runtimeDeliveryWorkerRejectsInvalidTaskIdBeforeLaunch()
    {
        namespace wp = aitrain::worker_protocol;
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        QVERIFY(QDir().mkpath(projectRoot));
        WorkerClient client;
        QString error;
        const QString taskId = QStringLiteral("not-a-task-uuid");
        QVERIFY2(!startTaskFromPayload(client, workerExecutablePath(),
            wp::command::runRuntimeDeliveryWorkflow(),
            wp::runtimeDeliveryWorkflowRequest(taskId, projectRoot,
                QUuid::createUuid().toString(QUuid::WithoutBraces), QStringLiteral("aitrain_onnxruntime"),
                QUuid::createUuid().toString(QUuid::WithoutBraces),
                QUuid::createUuid().toString(QUuid::WithoutBraces),
                QUuid::createUuid().toString(QUuid::WithoutBraces),
                QUuid::createUuid().toString(QUuid::WithoutBraces),
                QStringLiteral("images/train/a.png"), QJsonObject()), &error), qPrintable(error));
        QVERIFY(error.contains(QStringLiteral("UUID"), Qt::CaseInsensitive));
        QVERIFY(!client.isRunning());
    }

    void mismatchedCapabilityProfileIsRejected()
    {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString projectRoot = directory.filePath(QStringLiteral("project"));
        const QString datasetRoot = directory.filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(datasetRoot);
        QString error;
        const auto snapshot = commitTrainingSnapshotFixture(
            projectRoot, datasetRoot, QStringLiteral("yolo_detection"), &error);
        QVERIFY2(snapshot.snapshot.id.isValid(), qPrintable(error));
        QJsonObject request{
            {QStringLiteral("taskId"), QUuid::createUuid().toString(QUuid::WithoutBraces)},
            {QStringLiteral("projectRoot"), projectRoot},
            {QStringLiteral("capabilityId"), QStringLiteral("yolo")},
            {QStringLiteral("taskType"), QStringLiteral("detection")},
            {QStringLiteral("trainingBackend"), QStringLiteral("ultralytics_yolo_segment")}
        };
        const QJsonObject identity = trainingSnapshotIdentity(snapshot);
        for (auto it = identity.constBegin(); it != identity.constEnd(); ++it) request.insert(it.key(), it.value());
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        QString message;
        QStringList logs;
        connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) { logs.append(line); });
        connect(&client, &WorkerClient::finished, this,
            [&finished, &ok, &message](WorkerClient::WorkerTerminalStatus result, const QString& value) {
            finished = true;
            ok = result == WorkerClient::WorkerTerminalStatus::Succeeded;
            message = value;
        });
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            aitrain::worker_protocol::command::runTrainingWorkflow(), request, &error),
            qPrintable(error));
        QTRY_VERIFY_WITH_TIMEOUT(finished, 10000);
        QVERIFY(!ok);
        const QString evidence = message + QStringLiteral("\n") + logs.join(QStringLiteral("\n"));
        QVERIFY2(evidence.contains(QStringLiteral("能力矩阵")) || evidence.contains(QStringLiteral("不支持"))
            || evidence.contains(QStringLiteral("Workflow Profile")), qPrintable(evidence));
    }

    void yoloWorkflowTerminalEvidence_data()
    {
        QTest::addColumn<QString>("modelName");
        QTest::addColumn<bool>("requestCancel");
        QTest::addColumn<QString>("terminalEvent");
        QTest::addColumn<int>("expectedState");
        QTest::newRow("failed") << QStringLiteral("fake-yolo-fail.yaml") << false
                                 << QStringLiteral("failed") << static_cast<int>(aitrain::TaskState::Failed);
        QTest::newRow("canceled") << QStringLiteral("fake-yolo-slow.yaml") << true
                                   << QStringLiteral("canceled") << static_cast<int>(aitrain::TaskState::Canceled);
    }

    void yoloWorkflowTerminalEvidence()
    {
        QFETCH(QString, modelName);
        QFETCH(bool, requestCancel);
        QFETCH(QString, terminalEvent);
        QFETCH(int, expectedState);
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available.");
        }
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString projectRoot = dir.filePath(QStringLiteral("project"));
        const QString datasetRoot = QDir(projectRoot).filePath(QStringLiteral("dataset"));
        writeTinyDetectionDataset(datasetRoot);
        const QString fakePackageRoot = dir.filePath(QStringLiteral("fake_ultralytics"));
        writeFakeUltralyticsPackage(fakePackageRoot);
        QString error;
        const auto snapshot = commitTrainingSnapshotFixture(
            projectRoot, datasetRoot, QStringLiteral("yolo_detection"), &error);
        QVERIFY2(snapshot.snapshot.id.isValid(), qPrintable(error));
        ScopedEnvironment environment;
        environment.set("AITRAIN_PYTHON_EXECUTABLE", python);
        const QString inheritedPythonPath = QString::fromLocal8Bit(qgetenv("PYTHONPATH"));
        environment.set("PYTHONPATH", inheritedPythonPath.isEmpty()
            ? fakePackageRoot
            : fakePackageRoot + QDir::listSeparator() + inheritedPythonPath);
        const QString taskIdText = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QJsonObject request{
            {QStringLiteral("taskId"), taskIdText},
            {QStringLiteral("projectRoot"), projectRoot},
            {QStringLiteral("capabilityId"), QStringLiteral("yolo")},
            {QStringLiteral("taskType"), QStringLiteral("detection")},
            {QStringLiteral("trainingBackend"), QStringLiteral("ultralytics_yolo_detect")},
            {QStringLiteral("deploymentSampleRelativePath"), QStringLiteral("images/val/a.png")},
            {QStringLiteral("parameters"), QJsonObject{
                {QStringLiteral("model"), modelName},
                {QStringLiteral("epochs"), 1},
                {QStringLiteral("imageSize"), 32},
                {QStringLiteral("batchSize"), 1},
                {QStringLiteral("workers"), 0},
                {QStringLiteral("device"), QStringLiteral("cpu")},
                {QStringLiteral("cancellationGraceMs"), 1000}}}
        };
        const QJsonObject identity = trainingSnapshotIdentity(snapshot);
        for (auto it = identity.constBegin(); it != identity.constEnd(); ++it) request.insert(it.key(), it.value());
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        bool cancelSent = false;
        QString observedTerminal;
        QString evidencePath;
        QSet<QString> artifactKinds;
        QStringList logs;
        connectWorkerEvents(&client, this,
            [&client, &artifactKinds, &observedTerminal, &evidencePath, &cancelSent, requestCancel, projectRoot](const QString& type, const QJsonObject& payload) {
                if (type == QStringLiteral("artifact")) {
                    const QString kind = payload.value(QStringLiteral("kind")).toString();
                    artifactKinds.insert(kind);
                    if (kind == QStringLiteral("evidence_bundle")) evidencePath = committedArtifactEventPath(projectRoot, payload);
                }
                if (type == QStringLiteral("failed") || type == QStringLiteral("canceled")) {
                    observedTerminal = type;
                }
                if (requestCancel && !cancelSent && type == QStringLiteral("log")
                    && payload.value(QStringLiteral("workflowEventKind")).toString() == QStringLiteral("event.log")
                    && payload.value(QStringLiteral("message")).toString().contains(QStringLiteral("Prepared Ultralytics"))) {
                    cancelSent = true;
                    QTimer::singleShot(0, &client, [&client]() { client.cancel(); });
                }
            });
        connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) { logs.append(line); });
        connect(&client, &WorkerClient::finished, this,
            [&finished, &ok](WorkerClient::WorkerTerminalStatus result, const QString&) {
            finished = true;
            ok = result == WorkerClient::WorkerTerminalStatus::Succeeded;
        });
        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            aitrain::worker_protocol::command::runTrainingWorkflow(), request, &error),
            qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(finished, qPrintable(logs.join(QStringLiteral("\n"))), 20000);
        QVERIFY(!ok);
        if (requestCancel) QVERIFY(cancelSent);
        QCOMPARE(observedTerminal, terminalEvent);
        QVERIFY(artifactKinds.contains(QStringLiteral("evidence_bundle")));
        QVERIFY(!artifactKinds.contains(QStringLiteral("model_manifest")));
        const QJsonObject terminalEvidence = readJsonObject(evidencePath);
        QCOMPARE(terminalEvidence.value(QStringLiteral("task")).toObject().value(QStringLiteral("state")).toString(),
            terminalEvent == QStringLiteral("failed") ? QStringLiteral("failed") : QStringLiteral("canceled"));
        QVERIFY(!terminalEvidence.value(QStringLiteral("task")).toObject()
            .value(QStringLiteral("failure")).toObject().value(QStringLiteral("code")).toString().isEmpty());

        aitrain::ProjectStore storage;
        QVERIFY2(storage.open(QDir(projectRoot).filePath(QStringLiteral(".aitrain/project.sqlite")), &error), qPrintable(error));
        aitrain::TaskId taskId;
        QVERIFY2(aitrain::TaskId::parse(taskIdText, &taskId, &error), qPrintable(error));
        aitrain::TaskSnapshot task;
        QVERIFY2(storage.task(taskId, &task, &error), qPrintable(error));
        QCOMPARE(static_cast<int>(task.state), expectedState);
        QVERIFY(task.failure.code != aitrain::FailureCode::None);
    }

    void legacyTrainingPathFieldsAreRejected()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString projectRoot = dir.filePath(QStringLiteral("project"));
        QVERIFY(QDir().mkpath(projectRoot));
        const QString taskIdText = QUuid::createUuid().toString(QUuid::WithoutBraces);
        const QJsonObject request{
            {QStringLiteral("taskId"), taskIdText},
            {QStringLiteral("projectRoot"), projectRoot},
            {QStringLiteral("datasetPath"), dir.filePath(QStringLiteral("legacy-dataset"))},
            {QStringLiteral("format"), QStringLiteral("yolo_detection")},
            {QStringLiteral("capabilityId"), QStringLiteral("yolo")},
            {QStringLiteral("taskType"), QStringLiteral("detection")},
            {QStringLiteral("trainingBackend"), QStringLiteral("ultralytics_yolo_detect")},
            {QStringLiteral("sampleImagePath"), dir.filePath(QStringLiteral("legacy-sample.png"))}
        };
        WorkerClient client;
        bool finished = false;
        bool ok = true;
        QString terminalMessage;
        QStringList logs;
        connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) { logs.append(line); });
        connect(&client, &WorkerClient::finished, this,
            [&finished, &ok, &terminalMessage](WorkerClient::WorkerTerminalStatus result, const QString& message) {
            finished = true;
            ok = result == WorkerClient::WorkerTerminalStatus::Succeeded;
            terminalMessage = message;
        });
        QString error;
        QVERIFY(!startTaskFromPayload(client, workerExecutablePath(),
            aitrain::worker_protocol::command::runTrainingWorkflow(), request, &error));
        QVERIFY(!error.isEmpty());
        QVERIFY(!finished);
        QVERIFY(ok);
    }

    void workerRunsProfiledTrainingWorkflowEndToEnd_data()
    {
        QTest::addColumn<QString>("datasetFormat");
        QTest::addColumn<QString>("taskType");
        QTest::addColumn<QString>("trainingBackend");
        QTest::addColumn<QString>("modelName");
        QTest::addColumn<QString>("modelFamily");
        QTest::addColumn<QString>("decoder");

        QTest::newRow("detection")
            << QStringLiteral("yolo_detection")
            << QStringLiteral("detection")
            << QStringLiteral("ultralytics_yolo_detect")
            << QStringLiteral("fake-yolo.yaml")
            << QStringLiteral("yolo_detection")
            << QStringLiteral("yolo_detection_v8");
        QTest::newRow("segmentation")
            << QStringLiteral("yolo_segmentation")
            << QStringLiteral("segmentation")
            << QStringLiteral("ultralytics_yolo_segment")
            << QStringLiteral("fake-yolo-seg.yaml")
            << QStringLiteral("yolo_segmentation")
            << QStringLiteral("yolo_segmentation_v8");
        QTest::newRow("obb")
            << QStringLiteral("yolo_obb")
            << QStringLiteral("obb_detection")
            << QStringLiteral("ultralytics_yolo_obb")
            << QStringLiteral("fake-yolo-obb.yaml")
            << QStringLiteral("yolo_obb")
            << QStringLiteral("yolo_obb_v8");
        QTest::newRow("smp-semantic-segmentation")
            << QStringLiteral("semantic_segmentation_mask")
            << QStringLiteral("semantic_segmentation")
            << QStringLiteral("smp_semantic_segmentation")
            << QStringLiteral("smp_unet_resnet34")
            << QStringLiteral("semantic_segmentation")
            << QStringLiteral("smp_semantic_segmentation");
        QTest::newRow("anomalib-patchcore")
            << QStringLiteral("anomaly_folder")
            << QStringLiteral("anomaly_detection")
            << QStringLiteral("anomalib_patchcore")
            << QStringLiteral("anomalib_patchcore_wide_resnet50_2")
            << QStringLiteral("anomaly_detection")
            << QStringLiteral("anomalib_python_sidecar_v1");
        QTest::newRow("anomalib-efficientad")
            << QStringLiteral("anomaly_folder")
            << QStringLiteral("anomaly_detection")
            << QStringLiteral("anomalib_efficientad")
            << QStringLiteral("anomalib_efficientad_s")
            << QStringLiteral("anomaly_detection")
            << QStringLiteral("anomalib_python_sidecar_v1");
        QTest::newRow("paddleocr-det-official")
            << QStringLiteral("paddleocr_det")
            << QStringLiteral("ocr_detection")
            << QStringLiteral("paddleocr_det_official")
            << QStringLiteral("PP-OCRv5_mobile_det")
            << QStringLiteral("ocr_detection")
            << QStringLiteral("paddleocr_official_det_v1");
        QTest::newRow("paddleocr-rec-official")
            << QStringLiteral("paddleocr_rec")
            << QStringLiteral("ocr_recognition")
            << QStringLiteral("paddleocr_rec_official")
            << QStringLiteral("PP-OCRv5_mobile_rec")
            << QStringLiteral("ocr_recognition")
            << QStringLiteral("paddleocr_official_rec_v1");
    }

    void workerRunsProfiledTrainingWorkflowEndToEnd()
    {
        QFETCH(QString, datasetFormat);
        QFETCH(QString, taskType);
        QFETCH(QString, trainingBackend);
        QFETCH(QString, modelName);
        QFETCH(QString, modelFamily);
        QFETCH(QString, decoder);
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available.");
        }
        if (datasetFormat != QStringLiteral("anomaly_folder")
            && !datasetFormat.startsWith(QStringLiteral("paddleocr_"))
            && !pythonCanImportModule(python, QStringLiteral("onnx"))) {
            QSKIP("Python onnx package is not available for the  YOLO workflow fixture.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QString projectRoot = dir.filePath(QStringLiteral("project"));
        const QString datasetRoot = QDir(projectRoot).filePath(QStringLiteral("dataset"));
        QVERIFY(QDir().mkpath(datasetRoot));
        if (datasetFormat == QStringLiteral("yolo_segmentation")) {
            writeTinySegmentationDataset(datasetRoot);
        } else if (datasetFormat == QStringLiteral("yolo_obb")) {
            writeTinyObbDataset(datasetRoot);
        } else if (datasetFormat == QStringLiteral("semantic_segmentation_mask")) {
            writeTinySemanticMaskDataset(datasetRoot);
        } else if (datasetFormat == QStringLiteral("anomaly_folder")) {
            writeTinyPng(QDir(datasetRoot).filePath(QStringLiteral("train/good/a.png")));
            writeTinyPng(QDir(datasetRoot).filePath(QStringLiteral("val/good/b.png")));
            writeTinyPng(QDir(datasetRoot).filePath(QStringLiteral("test/anomaly/ng.png")));
            writeTinyMaskPng(QDir(datasetRoot).filePath(QStringLiteral("masks/test/anomaly/ng.png")));
        } else if (datasetFormat == QStringLiteral("paddleocr_det")) {
            writeTinyOcrDetDataset(datasetRoot);
        } else if (datasetFormat == QStringLiteral("paddleocr_rec")) {
            writeTinyOcrRecDataset(datasetRoot);
        } else {
            writeTinyDetectionDataset(datasetRoot);
        }
        const QString deploymentSampleRelativePath =
            datasetFormat == QStringLiteral("semantic_segmentation_mask")
                ? QStringLiteral("images/val/b.png")
                : (datasetFormat == QStringLiteral("anomaly_folder")
                    ? QStringLiteral("val/good/b.png")
                    : (datasetFormat.startsWith(QStringLiteral("paddleocr_"))
                        ? QStringLiteral("images/a.png") : QStringLiteral("images/val/a.png")));
        const QString fakePackageRoot = dir.filePath(QStringLiteral("fake_ultralytics"));
        QVERIFY(QDir().mkpath(fakePackageRoot));
        // 训练 fixture 只生成必要的 checkpoint；指标事件由独立适配器测试覆盖。
        writeFakeUltralyticsPackage(fakePackageRoot);
        const QString fakePaddleOcrRepo = datasetFormat.startsWith(QStringLiteral("paddleocr_"))
            ? writeFakePaddleOcrRepo(dir.filePath(QStringLiteral("fake_paddleocr"))) : QString();
        if (datasetFormat.startsWith(QStringLiteral("paddleocr_"))) {
            QVERIFY(!fakePaddleOcrRepo.isEmpty());
        }
        QString error;
        QString trainingWorker = workerExecutablePath();
        if (datasetFormat == QStringLiteral("semantic_segmentation_mask")) {
            trainingWorker = workerWithLocalTrainersFixture(dir.path(), false, &error);
        } else if (datasetFormat == QStringLiteral("anomaly_folder")) {
            trainingWorker = workerWithLocalTrainersFixture(dir.path(), true, &error);
        }
        QVERIFY2(QFileInfo(trainingWorker).isFile(), qPrintable(error));
        const auto snapshot = commitTrainingSnapshotFixture(projectRoot, datasetRoot, datasetFormat, &error);
        QVERIFY2(snapshot.snapshot.id.isValid(), qPrintable(error));

        ScopedEnvironment environment;
        environment.set("AITRAIN_PYTHON_EXECUTABLE", python);
        const QString inheritedPythonPath = QString::fromLocal8Bit(qgetenv("PYTHONPATH"));
        environment.set("PYTHONPATH", inheritedPythonPath.isEmpty()
            ? fakePackageRoot
            : fakePackageRoot + QDir::listSeparator() + inheritedPythonPath);
        if (!fakePaddleOcrRepo.isEmpty()) {
            environment.set("AITRAIN_PADDLEOCR_REPO", fakePaddleOcrRepo);
        }

        const QString taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QJsonObject parameters{
            {QStringLiteral("model"), modelName},
            {QStringLiteral("epochs"), 1},
            {QStringLiteral("imageSize"), 32},
            {QStringLiteral("batchSize"), 1},
            {QStringLiteral("workers"), 0},
            {QStringLiteral("device"), QStringLiteral("cpu")},
            {QStringLiteral("ultralyticsExportArgs"), QJsonObject{
                {QStringLiteral("format"), QStringLiteral("onnx")},
                {QStringLiteral("imgsz"), 32},
                {QStringLiteral("batch"), 1},
                {QStringLiteral("device"), QStringLiteral("cpu")}}}
        };
        QJsonObject request{
            {QStringLiteral("taskId"), taskId},
            {QStringLiteral("projectRoot"), projectRoot},
            {QStringLiteral("capabilityId"), datasetFormat == QStringLiteral("semantic_segmentation_mask")
                ? QStringLiteral("semantic_segmentation")
                : (datasetFormat == QStringLiteral("anomaly_folder")
                    ? QStringLiteral("anomaly_detection")
                    : (datasetFormat.startsWith(QStringLiteral("paddleocr_"))
                        ? QStringLiteral("paddleocr") : QStringLiteral("yolo")))},
            {QStringLiteral("taskType"), taskType},
            {QStringLiteral("trainingBackend"), trainingBackend},
            {QStringLiteral("deploymentSampleRelativePath"), deploymentSampleRelativePath},
            {QStringLiteral("parameters"), parameters}
        };
        const QJsonObject identity = trainingSnapshotIdentity(snapshot);
        for (auto it = identity.constBegin(); it != identity.constEnd(); ++it) request.insert(it.key(), it.value());

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        QStringList logs;
        bool finished = false;
        bool ok = false;
        QString finishedMessage;
        connectWorkerEvents(&client, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::logLine, this, [&logs](const QString& line) {
            logs.append(line);
        });
        connect(&client, &WorkerClient::finished, this,
            [&finished, &ok, &finishedMessage](WorkerClient::WorkerTerminalStatus result, const QString& message) {
            finished = true;
            ok = result == WorkerClient::WorkerTerminalStatus::Succeeded;
            finishedMessage = message;
        });

        QVERIFY2(startTaskFromPayload(client, trainingWorker,
            aitrain::worker_protocol::command::runTrainingWorkflow(), request, &error),
            qPrintable(error));
        QTRY_VERIFY2_WITH_TIMEOUT(finished,
            qPrintable(QStringLiteral(" Worker did not finish. Logs:\n%1").arg(logs.join(QStringLiteral("\n")))), 60000);
        QVERIFY2(ok, qPrintable(QStringList({finishedMessage, logs.join(QStringLiteral("\n"))}).join(QStringLiteral("\n"))));
        QTRY_VERIFY_WITH_TIMEOUT(!client.isRunning(), 5000);

        QStringList stepKinds;
        QSet<QString> artifactKinds;
        QHash<QString, QString> artifactPaths;
        bool sawFailure = false;
        for (const auto& message : messages) {
            if (message.first == QStringLiteral("log")) {
                const QString text = message.second.value(QStringLiteral("message")).toString();
                for (const QString& kind : {QStringLiteral("Train"), QStringLiteral("Evaluate"), QStringLiteral("Export"),
                         QStringLiteral("DeploymentValidate"), QStringLiteral("RegisterModel"), QStringLiteral("RenderDeliveryReport")}) {
                    if (text.contains(kind) && !stepKinds.contains(kind)) {
                        stepKinds.append(kind);
                    }
                }
            } else if (message.first == QStringLiteral("artifact")) {
                const QString kind = message.second.value(QStringLiteral("kind")).toString();
                artifactKinds.insert(kind);
                const QString committedPath = committedArtifactEventPath(projectRoot, message.second);
                if (!committedPath.isEmpty()) {
                    artifactPaths.insert(kind, committedPath);
                    QVERIFY2(QFileInfo(committedPath).exists(), qPrintable(kind));
                }
            } else if (message.first == QStringLiteral("failed") || message.first == QStringLiteral("canceled")) {
                sawFailure = true;
            }
        }
        QVERIFY(!sawFailure);
        QCOMPARE(stepKinds, QStringList({QStringLiteral("Train"), QStringLiteral("Evaluate"), QStringLiteral("Export"),
            QStringLiteral("DeploymentValidate"), QStringLiteral("RegisterModel"), QStringLiteral("RenderDeliveryReport")}));
        QVERIFY(artifactKinds.contains(QStringLiteral("model_manifest")));
        const bool paddleOcr = datasetFormat.startsWith(QStringLiteral("paddleocr_"));
        const QString deploymentReportKind = paddleOcr
            ? QStringLiteral("deployment_report") : QStringLiteral("deployment_validation_report");
        QVERIFY(artifactKinds.contains(deploymentReportKind));
        QVERIFY(artifactKinds.contains(paddleOcr ? QStringLiteral("prediction") : QStringLiteral("deployment_predictions")));
        QVERIFY(artifactKinds.contains(paddleOcr ? QStringLiteral("preview") : QStringLiteral("deployment_overlay")));
        const QJsonObject manifest = readJsonObject(artifactPaths.value(QStringLiteral("model_manifest")));
        QCOMPARE(manifest.value(QStringLiteral("modelFamily")).toString(), modelFamily);
        QCOMPARE(manifest.value(QStringLiteral("decoder")).toString(), decoder);
        QCOMPARE(manifest.value(QStringLiteral("runtimeRoutes")).toArray(),
            QJsonArray({datasetFormat == QStringLiteral("anomaly_folder")
                ? QStringLiteral("anomalib_python")
                : (paddleOcr ? QStringLiteral("paddleocr_official") : QStringLiteral("aitrain_onnxruntime"))}));
        const QJsonObject deployment = readJsonObject(artifactPaths.value(deploymentReportKind));
        QCOMPARE(deployment.value(QStringLiteral("status")).toString(), QStringLiteral("passed"));
        if (datasetFormat != QStringLiteral("anomaly_folder") && !paddleOcr) {
            QCOMPARE(deployment.value(QStringLiteral("details")).toObject().value(QStringLiteral("taskType")).toString(), taskType);
        } else if (datasetFormat == QStringLiteral("anomaly_folder")) {
            QCOMPARE(manifest.value(QStringLiteral("artifactFormat")).toString(), QStringLiteral("anomalib_bundle"));
            QCOMPARE(manifest.value(QStringLiteral("opset")).toInt(), 0);
            QCOMPARE(deployment.value(QStringLiteral("runtime")).toString(), QStringLiteral("anomalib_python"));
            QCOMPARE(deployment.value(QStringLiteral("taskType")).toString(), QStringLiteral("anomaly_detection"));
            QVERIFY(artifactKinds.contains(QStringLiteral("deployment_heatmap")));
            QVERIFY(artifactKinds.contains(QStringLiteral("deployment_mask")));
        } else {
            QCOMPARE(manifest.value(QStringLiteral("artifactFormat")).toString(), QStringLiteral("paddleocr_inference_bundle"));
            QCOMPARE(manifest.value(QStringLiteral("opset")).toInt(), 0);
            QCOMPARE(deployment.value(QStringLiteral("runtimeRoute")).toString(), QStringLiteral("paddleocr_official"));
            QCOMPARE(deployment.value(QStringLiteral("taskType")).toString(), taskType);
            QVERIFY(deployment.value(QStringLiteral("inventoryVerified")).toBool());
        }
        if (taskType == QStringLiteral("obb_detection")) {
            QCOMPARE(manifest.value(QStringLiteral("taskType")).toString(), QStringLiteral("obb_detection"));
            QCOMPARE(manifest.value(QStringLiteral("classNames")).toArray(),
                QJsonArray({QStringLiteral("ship"), QStringLiteral("plane")}));
            QCOMPARE(manifest.value(QStringLiteral("postprocessing")).toObject()
                .value(QStringLiteral("id")).toString(), QStringLiteral("yolo_obb_nms"));
            const QJsonObject predictions = readJsonObject(artifactPaths.value(QStringLiteral("deployment_predictions")));
            QCOMPARE(predictions.value(QStringLiteral("schemaVersion")).toInt(), 2);
            QCOMPARE(predictions.value(QStringLiteral("modelFamily")).toString(), QStringLiteral("yolo_obb"));
            QCOMPARE(predictions.value(QStringLiteral("taskType")).toString(), QStringLiteral("obb_detection"));
            QCOMPARE(predictions.value(QStringLiteral("runtime")).toString(), QStringLiteral("aitrain_onnxruntime"));
            const QJsonArray values = predictions.value(QStringLiteral("predictions")).toArray();
            QCOMPARE(values.size(), 1);
            const QJsonObject prediction = values.first().toObject();
            QCOMPARE(prediction.value(QStringLiteral("taskType")).toString(), QStringLiteral("obb_detection"));
            QCOMPARE(prediction.value(QStringLiteral("classId")).toInt(), 1);
            QCOMPARE(prediction.value(QStringLiteral("className")).toString(), QStringLiteral("plane"));
            QVERIFY(qAbs(prediction.value(QStringLiteral("confidence")).toDouble() - 0.9) < 0.0001);
            const QJsonArray xywhr = prediction.value(QStringLiteral("xywhr")).toArray();
            QCOMPARE(xywhr.size(), 5);
            QVERIFY(qAbs(xywhr.at(4).toDouble() - 0.25) < 0.0001);
            QCOMPARE(prediction.value(QStringLiteral("points")).toArray().size(), 4);
            QVERIFY(prediction.value(QStringLiteral("bbox")).isObject());
        } else if (taskType == QStringLiteral("semantic_segmentation")) {
            QCOMPARE(manifest.value(QStringLiteral("classNames")).toArray(),
                QJsonArray({QStringLiteral("background"), QStringLiteral("part"), QStringLiteral("scratch")}));
            QCOMPARE(manifest.value(QStringLiteral("opset")).toInt(), 13);
            const QJsonObject predictions = readJsonObject(artifactPaths.value(QStringLiteral("deployment_predictions")));
            QCOMPARE(predictions.value(QStringLiteral("modelFamily")).toString(), QStringLiteral("semantic_segmentation"));
            QCOMPARE(predictions.value(QStringLiteral("taskType")).toString(), QStringLiteral("semantic_segmentation"));
            const QJsonArray values = predictions.value(QStringLiteral("predictions")).toArray();
            QCOMPARE(values.size(), 1);
            QCOMPARE(values.first().toObject().value(QStringLiteral("taskType")).toString(),
                QStringLiteral("semantic_segmentation"));
            QVERIFY(values.first().toObject().value(QStringLiteral("pixelCounts")).isObject());
        }
        const QStringList evidenceLogs = logs.filter(QStringLiteral("Evidence Bundle"));
        QVERIFY2(artifactKinds.contains(QStringLiteral("evidence_bundle")), qPrintable(evidenceLogs.join(QStringLiteral("\n"))));
        const QJsonObject evidence = readJsonObject(artifactPaths.value(QStringLiteral("evidence_bundle")));
        QVERIFY(!evidence.value(QStringLiteral("datasetSnapshotId")).toString().isEmpty());
        QCOMPARE(evidence.value(QStringLiteral("runtimeStatus")).toObject()
            .value(QStringLiteral("steps")).toArray().size(), 8);
        if (taskType == QStringLiteral("segmentation") || taskType == QStringLiteral("obb_detection")
            || taskType == QStringLiteral("semantic_segmentation") || taskType == QStringLiteral("anomaly_detection")
            || taskType.startsWith(QStringLiteral("ocr_"))) {
            QStringList limitations;
            for (const QJsonValue& value : evidence.value(QStringLiteral("limitations")).toArray()) {
                limitations.append(value.toString());
            }
            if (taskType == QStringLiteral("anomaly_detection")) {
                QVERIFY(limitations.join(QStringLiteral("\n")).contains(QStringLiteral("不声明 AITrain C++ ONNX")));
            } else if (taskType.startsWith(QStringLiteral("ocr_"))) {
                QVERIFY(limitations.join(QStringLiteral("\n")).contains(QStringLiteral("不声明 AITrain C++ OCR")));
            } else {
                QVERIFY(limitations.join(QStringLiteral("\n")).contains(QStringLiteral("ONNX Runtime")));
            }
            if (taskType == QStringLiteral("obb_detection")) {
                QVERIFY(limitations.join(QStringLiteral("\n")).contains(QStringLiteral("不声明 NCNN 或 TensorRT")));
            } else if (taskType == QStringLiteral("semantic_segmentation")) {
                QVERIFY(limitations.join(QStringLiteral("\n")).contains(QStringLiteral("不支持 NCNN 或 TensorRT")));
            }
        }
    }

private slots:
    void paddleOcrSystemV6UsesInferenceMetadataAlgorithm()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available for the official PaddleOCR System adapter test.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QDir root(dir.path());
        const QString recModelDir = root.filePath(QStringLiteral("rec-model"));
        const QString detModelDir = root.filePath(QStringLiteral("det-model"));
        QDir().mkpath(recModelDir);
        QDir().mkpath(detModelDir);
        writeTextFile(QDir(recModelDir).filePath(QStringLiteral("inference.yml")),
            QStringLiteral("Global:\n  model_name: PP-OCRv6_medium_rec\nArchitecture:\n  algorithm: SVTR_LCNet\n"));

        const QString requestPath = root.filePath(QStringLiteral("system-request.json"));
        const QString outputPath = root.filePath(QStringLiteral("system-output"));
        QJsonObject parameters;
        parameters.insert(QStringLiteral("prepareOnly"), true);
        parameters.insert(QStringLiteral("detModelDir"), detModelDir);
        parameters.insert(QStringLiteral("recModelDir"), recModelDir);
        parameters.insert(QStringLiteral("dictionaryFile"), root.filePath(QStringLiteral("dict.txt")));
        parameters.insert(QStringLiteral("inferenceImage"), root.filePath(QStringLiteral("sample.png")));
        parameters.insert(QStringLiteral("recModelPreset"), QStringLiteral("PP-OCRv6_medium_rec"));
        QJsonObject request;
        request.insert(QStringLiteral("protocolVersion"), 1);
        request.insert(QStringLiteral("taskId"), QStringLiteral("paddleocr-system-v6-command"));
        request.insert(QStringLiteral("taskType"), QStringLiteral("ocr"));
        request.insert(QStringLiteral("datasetPath"), root.filePath(QStringLiteral("sample.png")));
        request.insert(QStringLiteral("outputPath"), outputPath);
        request.insert(QStringLiteral("backend"), QStringLiteral("paddleocr_system_official"));
        request.insert(QStringLiteral("parameters"), parameters);
        writeTextFile(requestPath, QString::fromUtf8(QJsonDocument(request).toJson(QJsonDocument::Indented)));

        QProcess process;
        QProcessEnvironment standaloneEnvironment = QProcessEnvironment::systemEnvironment();
        standaloneEnvironment.insert(QStringLiteral("AITRAIN_STANDALONE_ADAPTER_PROTOCOL"), QStringLiteral("1"));
        process.setProcessEnvironment(standaloneEnvironment);
        const QString adapterPath = repoRelativeFilePath(QStringLiteral("python_trainers/ocr_system/paddleocr_system_official_adapter.py"));
        process.setWorkingDirectory(QFileInfo(adapterPath).absolutePath() + QStringLiteral("/../.."));
        process.start(python, QStringList()
                << adapterPath
                << QStringLiteral("--request")
                << requestPath);
        QVERIFY2(process.waitForFinished(15000), qPrintable(QString::fromUtf8(process.readAllStandardError())));
        QCOMPARE(process.exitCode(), 0);

        const QJsonObject report = readJsonObject(QDir(outputPath).filePath(QStringLiteral("paddleocr_official_system_report.json")));
        QCOMPARE(report.value(QStringLiteral("recAlgorithm")).toString(), QStringLiteral("SVTR_LCNet"));
        const QJsonArray command = report.value(QStringLiteral("predictCommand")).toArray();
        bool sawAlgorithm = false;
        for (const QJsonValue& value : command) {
            sawAlgorithm = sawAlgorithm || value.toString() == QStringLiteral("--rec_algorithm=SVTR_LCNet");
        }
        QVERIFY(sawAlgorithm);
    }

    void paddleOcrSystemV6FailsWithoutAlgorithmMetadata()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available for the official PaddleOCR System adapter test.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QDir root(dir.path());
        const QString recModelDir = root.filePath(QStringLiteral("rec-model"));
        const QString detModelDir = root.filePath(QStringLiteral("det-model"));
        QDir().mkpath(recModelDir);
        QDir().mkpath(detModelDir);
        writeTextFile(QDir(recModelDir).filePath(QStringLiteral("inference.yml")),
            QStringLiteral("Global:\n  model_name: PP-OCRv6_medium_rec\n"));

        const QString requestPath = root.filePath(QStringLiteral("system-request.json"));
        const QString outputPath = root.filePath(QStringLiteral("system-output"));
        QJsonObject parameters;
        parameters.insert(QStringLiteral("prepareOnly"), true);
        parameters.insert(QStringLiteral("detModelDir"), detModelDir);
        parameters.insert(QStringLiteral("recModelDir"), recModelDir);
        parameters.insert(QStringLiteral("dictionaryFile"), root.filePath(QStringLiteral("dict.txt")));
        parameters.insert(QStringLiteral("inferenceImage"), root.filePath(QStringLiteral("sample.png")));
        parameters.insert(QStringLiteral("recModelPreset"), QStringLiteral("PP-OCRv6_medium_rec"));
        QJsonObject request;
        request.insert(QStringLiteral("protocolVersion"), 1);
        request.insert(QStringLiteral("taskId"), QStringLiteral("paddleocr-system-v6-missing-algorithm"));
        request.insert(QStringLiteral("taskType"), QStringLiteral("ocr"));
        request.insert(QStringLiteral("datasetPath"), root.filePath(QStringLiteral("sample.png")));
        request.insert(QStringLiteral("outputPath"), outputPath);
        request.insert(QStringLiteral("backend"), QStringLiteral("paddleocr_system_official"));
        request.insert(QStringLiteral("parameters"), parameters);
        writeTextFile(requestPath, QString::fromUtf8(QJsonDocument(request).toJson(QJsonDocument::Indented)));

        QProcess process;
        QProcessEnvironment standaloneEnvironment = QProcessEnvironment::systemEnvironment();
        standaloneEnvironment.insert(QStringLiteral("AITRAIN_STANDALONE_ADAPTER_PROTOCOL"), QStringLiteral("1"));
        process.setProcessEnvironment(standaloneEnvironment);
        const QString adapterPath = repoRelativeFilePath(QStringLiteral("python_trainers/ocr_system/paddleocr_system_official_adapter.py"));
        process.setWorkingDirectory(QFileInfo(adapterPath).absolutePath() + QStringLiteral("/../.."));
        process.start(python, QStringList()
                << adapterPath
                << QStringLiteral("--request")
                << requestPath);
        QVERIFY2(process.waitForFinished(15000), qPrintable(QString::fromUtf8(process.readAllStandardError())));
        QVERIFY(process.exitCode() != 0);
        const QString output = QString::fromUtf8(process.readAllStandardOutput());
        QVERIFY2(output.contains(QStringLiteral("rec_algorithm_missing")), qPrintable(output));
    }

    void paddleOcrSystemV5ServerRecUsesHgNetAlgorithm()
    {
        const QString python = pythonExecutablePath();
        if (python.isEmpty()) {
            QSKIP("Python executable is not available for the official PaddleOCR System adapter test.");
        }

        QTemporaryDir dir;
        QVERIFY(dir.isValid());
        const QDir root(dir.path());
        const QString recModelDir = root.filePath(QStringLiteral("rec-model"));
        const QString detModelDir = root.filePath(QStringLiteral("det-model"));
        QDir().mkpath(recModelDir);
        QDir().mkpath(detModelDir);
        writeTextFile(QDir(recModelDir).filePath(QStringLiteral("inference.yml")),
            QStringLiteral("Global:\n  model_name: PP-OCRv5_server_rec\nArchitecture:\n  algorithm: SVTR_HGNet\n"));

        const QString requestPath = root.filePath(QStringLiteral("system-request.json"));
        const QString outputPath = root.filePath(QStringLiteral("system-output"));
        QJsonObject parameters;
        parameters.insert(QStringLiteral("prepareOnly"), true);
        parameters.insert(QStringLiteral("detModelDir"), detModelDir);
        parameters.insert(QStringLiteral("recModelDir"), recModelDir);
        parameters.insert(QStringLiteral("dictionaryFile"), root.filePath(QStringLiteral("dict.txt")));
        parameters.insert(QStringLiteral("inferenceImage"), root.filePath(QStringLiteral("sample.png")));
        parameters.insert(QStringLiteral("recModelPreset"), QStringLiteral("PP-OCRv5_server_rec"));
        QJsonObject request;
        request.insert(QStringLiteral("protocolVersion"), 1);
        request.insert(QStringLiteral("taskId"), QStringLiteral("paddleocr-system-v5-command"));
        request.insert(QStringLiteral("taskType"), QStringLiteral("ocr"));
        request.insert(QStringLiteral("datasetPath"), root.filePath(QStringLiteral("sample.png")));
        request.insert(QStringLiteral("outputPath"), outputPath);
        request.insert(QStringLiteral("backend"), QStringLiteral("paddleocr_system_official"));
        request.insert(QStringLiteral("parameters"), parameters);
        writeTextFile(requestPath, QString::fromUtf8(QJsonDocument(request).toJson(QJsonDocument::Indented)));

        QProcess process;
        QProcessEnvironment standaloneEnvironment = QProcessEnvironment::systemEnvironment();
        standaloneEnvironment.insert(QStringLiteral("AITRAIN_STANDALONE_ADAPTER_PROTOCOL"), QStringLiteral("1"));
        process.setProcessEnvironment(standaloneEnvironment);
        const QString adapterPath = repoRelativeFilePath(QStringLiteral("python_trainers/ocr_system/paddleocr_system_official_adapter.py"));
        process.setWorkingDirectory(QFileInfo(adapterPath).absolutePath() + QStringLiteral("/../.."));
        process.start(python, QStringList()
                << adapterPath
                << QStringLiteral("--request")
                << requestPath);
        QVERIFY2(process.waitForFinished(15000), qPrintable(QString::fromUtf8(process.readAllStandardError())));
        QCOMPARE(process.exitCode(), 0);

        const QJsonObject report = readJsonObject(QDir(outputPath).filePath(QStringLiteral("paddleocr_official_system_report.json")));
        QCOMPARE(report.value(QStringLiteral("recAlgorithm")).toString(), QStringLiteral("SVTR_HGNet"));
        const QJsonArray command = report.value(QStringLiteral("predictCommand")).toArray();
        bool sawHgNet = false;
        for (const QJsonValue& value : command) {
            sawHgNet = sawHgNet || value.toString() == QStringLiteral("--rec_algorithm=SVTR_HGNet");
        }
        QVERIFY(sawHgNet);
    }

    void workerEnvironmentCheckReportsOfficialTrainerProfiles()
    {
        QTemporaryDir projectDir;
        QVERIFY(projectDir.isValid());
        const aitrain::TaskId taskId = aitrain::TaskId::create();
        QString error;
        QVERIFY2(initializeWorkerProject(projectDir.path(), &error), qPrintable(error));

        WorkerClient client;
        QVector<QPair<QString, QJsonObject>> messages;
        bool finished = false;
        connectWorkerEvents(&client, this, [&messages](const QString& type, const QJsonObject& payload) {
            messages.append(qMakePair(type, payload));
        });
        connect(&client, &WorkerClient::idle, this, [&finished]() {
            finished = true;
        });

        QVERIFY2(startTaskFromPayload(client, workerExecutablePath(),
            aitrain::worker_protocol::command::runEnvironmentCheckWorkflow(),
            QJsonObject{{aitrain::worker_protocol::field::taskId(), taskId.toString()},
                {QStringLiteral("projectRoot"), projectDir.path()}}, &error), qPrintable(error));
        // 环境检查会探测多个本地 Python/SDK 配置；在繁忙 Windows 主机上已观测到超过 15 秒。
        // 该超时只覆盖异步 Worker 完成，不放宽检查结果断言。
        QTRY_VERIFY_WITH_TIMEOUT(finished, 60000);

        bool sawResult = false;
        for (const auto& message : messages) {
            if (message.first != QStringLiteral("environmentCheckWorkflow")) continue;
            sawResult = true;
            QCOMPARE(message.second.value(QStringLiteral("taskId")).toString(), taskId.toString());
            QVERIFY(!message.second.value(QStringLiteral("reportArtifactId")).toString().isEmpty());
            QVERIFY(!message.second.value(QStringLiteral("evidenceArtifactId")).toString().isEmpty());
            QVERIFY(!message.second.contains(QStringLiteral("checks")));
            QVERIFY(!message.second.contains(QStringLiteral("profiles")));
            QVERIFY(!message.second.contains(QStringLiteral("reportPath")));
        }
        QVERIFY(sawResult);

        aitrain::ProjectWorkspace workspace;
        QVERIFY2(workspace.open(projectDir.path(), &error), qPrintable(error));
        QJsonObject report;
        QVERIFY2(workspace.environmentCheckReportForTask(taskId, &report, &error), qPrintable(error));
        const QJsonArray checks = report.value(QStringLiteral("checks")).toArray();
        const QJsonObject profiles = report.value(QStringLiteral("profiles")).toObject();
        QVERIFY(!checks.isEmpty());
        QVERIFY(profiles.contains(QStringLiteral("yolo")));
        QVERIFY(profiles.contains(QStringLiteral("ocr")));
        QVERIFY(profiles.contains(QStringLiteral("tensorrt")));
        for (const QJsonValue& value : checks)
            QVERIFY(!value.toObject().contains(QStringLiteral("details")));
        const QByteArray safeReport = QJsonDocument(report).toJson(QJsonDocument::Compact);
        QVERIFY(!safeReport.contains("C:/"));
        QVERIFY(!safeReport.contains("D:/"));
        QVERIFY(!safeReport.contains("\\\\"));
    }
};

QTEST_MAIN(OcrSegmentationWorkerTests)
#include "tst_ocr_segmentation_worker.moc"
