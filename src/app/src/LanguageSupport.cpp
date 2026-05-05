#include "LanguageSupport.h"

#include <QApplication>
#include <QAbstractButton>
#include <QCoreApplication>
#include <QDir>
#include <QGroupBox>
#include <QHash>
#include <QLabel>
#include <QLineEdit>
#include <QLocale>
#include <QPlainTextEdit>
#include <QComboBox>
#include <QSettings>
#include <QStringList>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTextEdit>
#include <QTranslator>

namespace aitrain_app {
namespace {

QString normalizeLanguageCode(const QString& languageCode)
{
    if (languageCode == QStringLiteral("en") || languageCode == QStringLiteral("en_US")) {
        return QStringLiteral("en_US");
    }
    return QStringLiteral("zh_CN");
}

QStringList translationSearchRoots()
{
    const QString appDir = QApplication::applicationDirPath();
    return {
        QStringLiteral(":/translations"),
        QDir(appDir).absoluteFilePath(QStringLiteral("translations")),
        QDir(appDir).absoluteFilePath(QStringLiteral("../translations")),
        QDir(appDir).absoluteFilePath(QStringLiteral("../../translations")),
        QDir::current().absoluteFilePath(QStringLiteral("translations"))
    };
}

QString fallbackEnglishTranslation(const QString& source)
{
    static const QHash<QString, QString> translations = {
        {QStringLiteral("检测"), QStringLiteral("Detection")},
        {QStringLiteral("分割"), QStringLiteral("Segmentation")},
        {QStringLiteral("OCR 检测"), QStringLiteral("OCR Detection")},
        {QStringLiteral("OCR 识别"), QStringLiteral("OCR Recognition")},
        {QStringLiteral("OCR 端到端"), QStringLiteral("OCR End-to-End")},
        {QStringLiteral("未选择"), QStringLiteral("Not selected")},
        {QStringLiteral("Ultralytics YOLO 检测（官方）"), QStringLiteral("Ultralytics YOLO Detection (Official)")},
        {QStringLiteral("Ultralytics YOLO 分割（官方）"), QStringLiteral("Ultralytics YOLO Segmentation (Official)")},
        {QStringLiteral("PaddleOCR Det（官方/隔离环境）"), QStringLiteral("PaddleOCR Det (Official / Isolated Env)")},
        {QStringLiteral("PaddleOCR PP-OCRv4 Rec（官方/隔离环境）"), QStringLiteral("PaddleOCR PP-OCRv4 Rec (Official / Isolated Env)")},
        {QStringLiteral("PaddleOCR System 推理（官方）"), QStringLiteral("PaddleOCR System Inference (Official)")},
        {QStringLiteral("Tiny detector（高级/占位）"), QStringLiteral("Tiny detector (Advanced / Placeholder)")},
        {QStringLiteral("Python mock（高级/协议测试）"), QStringLiteral("Python mock (Advanced / Protocol Test)")},
        {QStringLiteral("ONNX 模型"), QStringLiteral("ONNX Model")},
        {QStringLiteral("NCNN param/bin（onnx2ncnn）"), QStringLiteral("NCNN param/bin (onnx2ncnn)")},
        {QStringLiteral("AITrain JSON（诊断）"), QStringLiteral("AITrain JSON (Diagnostic)")},
        {QStringLiteral("TensorRT Engine（RTX / SM 75+ 外部验收）"), QStringLiteral("TensorRT Engine (RTX / SM 75+ External Acceptance)")},
        {QStringLiteral("排队中"), QStringLiteral("Queued")},
        {QStringLiteral("运行中"), QStringLiteral("Running")},
        {QStringLiteral("已暂停"), QStringLiteral("Paused")},
        {QStringLiteral("已完成"), QStringLiteral("Completed")},
        {QStringLiteral("失败"), QStringLiteral("Failed")},
        {QStringLiteral("已取消"), QStringLiteral("Canceled")},
        {QStringLiteral("未知"), QStringLiteral("Unknown")},
        {QStringLiteral("训练"), QStringLiteral("Training")},
        {QStringLiteral("校验"), QStringLiteral("Validation")},
        {QStringLiteral("导出"), QStringLiteral("Export")},
        {QStringLiteral("推理"), QStringLiteral("Inference")},
        {QStringLiteral("评估"), QStringLiteral("Evaluation")},
        {QStringLiteral("基准"), QStringLiteral("Benchmark")},
        {QStringLiteral("质检"), QStringLiteral("Quality Check")},
        {QStringLiteral("快照"), QStringLiteral("Snapshot")},
        {QStringLiteral("流水线"), QStringLiteral("Pipeline")},
        {QStringLiteral("任务"), QStringLiteral("Task")},
        {QStringLiteral("通过"), QStringLiteral("Passed")},
        {QStringLiteral("警告"), QStringLiteral("Warning")},
        {QStringLiteral("缺失"), QStringLiteral("Missing")},
        {QStringLiteral("错误"), QStringLiteral("Error")},
        {QStringLiteral("信息"), QStringLiteral("Info")},
        {QStringLiteral("YOLO 检测"), QStringLiteral("YOLO Detection")},
        {QStringLiteral("YOLO 分割"), QStringLiteral("YOLO Segmentation")},
        {QStringLiteral("当前模型能力：官方 Ultralytics YOLO detection。适合 YOLO bbox 数据，输出 best.pt、ONNX、训练报告，可继续做 ONNX Runtime 推理和 overlay 验证。"), QStringLiteral("Current capability: official Ultralytics YOLO detection. Suitable for YOLO bbox data; outputs best.pt, ONNX, and a training report for ONNX Runtime inference and overlay validation.")},
        {QStringLiteral("当前模型能力：官方 Ultralytics YOLO segmentation。适合 YOLO polygon 数据，输出 mask 指标、best.pt、ONNX，并可生成 mask prediction JSON 与 overlay。"), QStringLiteral("Current capability: official Ultralytics YOLO segmentation. Suitable for YOLO polygon data; outputs mask metrics, best.pt, ONNX, mask prediction JSON, and overlays.")},
        {QStringLiteral("当前模型能力：PaddlePaddle CTC OCR Rec。适合 rec_gt.txt + dict.txt 小规模识别数据，可导出 ONNX 并走 C++ CTC greedy decode。"), QStringLiteral("Current capability: PaddlePaddle CTC OCR Rec. Suitable for small rec_gt.txt + dict.txt recognition datasets; exports ONNX for C++ CTC greedy decode.")},
        {QStringLiteral("当前模型能力：官方 PaddleOCR PP-OCRv4 Rec 适配器。适合隔离 OCR Python 环境，记录 train/export/predict 命令、checkpoint、inference model 和官方预测报告。"), QStringLiteral("Current capability: official PaddleOCR PP-OCRv4 Rec adapter. Best used in an isolated OCR Python environment; records train/export/predict commands, checkpoints, inference models, and official prediction reports.")},
        {QStringLiteral("当前模型能力：官方 PaddleOCR PP-OCRv4 Det 适配器。适合 PaddleOCR 原生 det_gt.txt 数据，输出官方配置、checkpoint、inference model 和报告。"), QStringLiteral("Current capability: official PaddleOCR PP-OCRv4 Det adapter. Suitable for native PaddleOCR det_gt.txt data; outputs official config, checkpoint, inference model, and report.")},
        {QStringLiteral("当前模型能力：官方 PaddleOCR 端到端推理编排。使用已导出的 Det/Rec inference model 调用 predict_system.py；本阶段不做 C++ DB 后处理。"), QStringLiteral("Current capability: official PaddleOCR end-to-end inference orchestration. Calls predict_system.py with exported Det/Rec inference models; C++ DB postprocess is not included in this phase.")},
        {QStringLiteral("高级/诊断：C++ tiny detector 占位训练，仅验证平台链路、checkpoint、ONNX 和回归测试，不代表真实 YOLO 能力。"), QStringLiteral("Advanced / diagnostic: C++ tiny detector placeholder training. It only verifies platform flow, checkpoints, ONNX, and regression tests; it is not real YOLO capability.")},
        {QStringLiteral("高级/诊断：Python 协议测试后端，只验证 Worker JSON Lines 协议，不产生真实模型。"), QStringLiteral("Advanced / diagnostic: Python protocol test backend. It only verifies the Worker JSON Lines protocol and does not produce a real model.")},
        {QStringLiteral("当前模型能力：通过 Worker 执行，产物、指标和失败原因会写入任务历史。"), QStringLiteral("Current capability: executed through the Worker; artifacts, metrics, and failure reasons are written to task history.")},
        {QStringLiteral("主交付格式，可继续进入推理验证。"), QStringLiteral("Primary delivery format; can continue to inference validation.")},
        {QStringLiteral("需要配置 onnx2ncnn，输出 param/bin。"), QStringLiteral("Requires onnx2ncnn; outputs param/bin.")},
        {QStringLiteral("需要 RTX / SM 75+ 真机外部验收。"), QStringLiteral("Requires external validation on real RTX / SM 75+ hardware.")},
        {QStringLiteral("仅用于 tiny detector 诊断，不代表真实 YOLO/OCR。"), QStringLiteral("Only for tiny detector diagnostics; not real YOLO/OCR capability.")},
        {QStringLiteral("暂无"), QStringLiteral("None")},
        {QStringLiteral("%1 等 %2 项"), QStringLiteral("%1 and %2 items")},
        {QStringLiteral("状态：未检测到 X-AnyLabeling。可放到 .deps/annotation-tools/X-AnyLabeling，或设置 AITRAIN_XANYLABELING_EXE。"), QStringLiteral("Status: X-AnyLabeling was not detected. Place it under .deps/annotation-tools/X-AnyLabeling or set AITRAIN_XANYLABELING_EXE.")},
        {QStringLiteral("状态：已安装 | %1"), QStringLiteral("Status: Installed | %1")},
        {QStringLiteral("%1：%2 个结果，%3 ms\n结果文件：%4"), QStringLiteral("%1: %2 results, %3 ms\nResult file: %4")},
        {QStringLiteral("预测结果 JSON 无法解析：%1"), QStringLiteral("Prediction JSON could not be parsed: %1")},
        {QStringLiteral("未识别出文本"), QStringLiteral("No text recognized")},
        {QStringLiteral("文本 \"%1\""), QStringLiteral("Text \"%1\"")},
        {QStringLiteral("，置信度 %1"), QStringLiteral(", confidence %1")},
        {QStringLiteral("首个 %1，置信度 %2"), QStringLiteral("First %1, confidence %2")},
        {QStringLiteral("无结果"), QStringLiteral("No results")},
        {QStringLiteral("%1：%2 个结果，%3，%4 ms\n结果文件：%5"), QStringLiteral("%1: %2 results, %3, %4 ms\nResult file: %5")},
        {QStringLiteral("标注工具"), QStringLiteral("Annotation Tool")},
        {QStringLiteral("请先选择数据集目录。"), QStringLiteral("Choose a dataset folder first.")},
        {QStringLiteral("未找到 X-AnyLabeling。请确保 xanylabeling 在 PATH 中，或将 X-AnyLabeling.exe 放到程序目录 / tools/x-anylabeling / .deps/annotation-tools/X-AnyLabeling。"), QStringLiteral("X-AnyLabeling was not found. Make sure xanylabeling is in PATH, or place X-AnyLabeling.exe in the app folder, tools/x-anylabeling, or .deps/annotation-tools/X-AnyLabeling.")},
        {QStringLiteral("已启动 X-AnyLabeling：%1"), QStringLiteral("Started X-AnyLabeling: %1")},
        {QStringLiteral("X-AnyLabeling 启动失败：%1"), QStringLiteral("Failed to start X-AnyLabeling: %1")},
        {QStringLiteral("全部类别"), QStringLiteral("All Categories")},
        {QStringLiteral("全部状态"), QStringLiteral("All Statuses")},
        {QStringLiteral("checkpoint 或 ONNX"), QStringLiteral("checkpoint or ONNX")},
        {QStringLiteral("未检测"), QStringLiteral("Unchecked")},
        {QStringLiteral("点击执行环境自检。"), QStringLiteral("Click Run Environment Check.")},
        {QStringLiteral("项目"), QStringLiteral("Project")},
        {QStringLiteral("项目名称和目录不能为空。"), QStringLiteral("Project name and folder cannot be empty.")},
        {QStringLiteral("上次会话结束时任务未正常完成，已标记为失败。"), QStringLiteral("Tasks that did not finish cleanly in the previous session were marked as failed.")},
        {QStringLiteral("当前项目：%1"), QStringLiteral("Current project: %1")},
        {QStringLiteral("项目已打开：%1"), QStringLiteral("Project opened: %1")},
        {QStringLiteral("选择数据集目录"), QStringLiteral("Choose Dataset Folder")},
        {QStringLiteral("数据集校验"), QStringLiteral("Dataset Validation")},
        {QStringLiteral("Worker 正在执行任务，稍后再校验数据集。"), QStringLiteral("Worker is running a task. Validate the dataset later.")},
        {QStringLiteral("请选择数据集目录和格式。"), QStringLiteral("Choose a dataset folder and format.")},
        {QStringLiteral("正在通过 Worker 校验数据集。"), QStringLiteral("Validating the dataset through Worker.")},
        {QStringLiteral("等待校验结果。"), QStringLiteral("Waiting for validation results.")},
        {QStringLiteral("数据集校验中。"), QStringLiteral("Dataset validation in progress.")},
        {QStringLiteral("无法启动数据集校验：%1"), QStringLiteral("Failed to start dataset validation: %1")},
        {QStringLiteral("数据集校验中"), QStringLiteral("Validating dataset")},
        {QStringLiteral("数据集划分"), QStringLiteral("Dataset Split")},
        {QStringLiteral("Worker 正在执行任务，稍后再划分数据集。"), QStringLiteral("Worker is running a task. Split the dataset later.")},
        {QStringLiteral("请先选择数据集目录和格式。"), QStringLiteral("Choose a dataset folder and format first.")},
        {QStringLiteral("当前划分支持 YOLO 检测、YOLO 分割、PaddleOCR Det 和 PaddleOCR Rec 格式。"), QStringLiteral("Current split supports YOLO detection, YOLO segmentation, PaddleOCR Det, and PaddleOCR Rec formats.")},
        {QStringLiteral("请先通过当前格式的数据集校验。"), QStringLiteral("Validate the dataset with the current format first.")},
        {QStringLiteral("数据集划分中。"), QStringLiteral("Dataset split in progress.")},
        {QStringLiteral("数据集划分中"), QStringLiteral("Splitting dataset")},
        {QStringLiteral("正在划分数据集"), QStringLiteral("Splitting dataset")},
        {QStringLiteral("模型导出"), QStringLiteral("Model Export")},
        {QStringLiteral("Worker 正在执行任务，稍后再导出模型。"), QStringLiteral("Worker is running a task. Export the model later.")},
        {QStringLiteral("请选择模型输入。"), QStringLiteral("Choose model input.")},
        {QStringLiteral("等待 Worker 导出模型。"), QStringLiteral("Waiting for Worker to export the model.")},
        {QStringLiteral("模型导出中。"), QStringLiteral("Model export in progress.")},
        {QStringLiteral("正在导出：%1"), QStringLiteral("Exporting: %1")},
        {QStringLiteral("模型导出中"), QStringLiteral("Exporting model")},
        {QStringLiteral("推理"), QStringLiteral("Inference")},
        {QStringLiteral("Worker 正在执行任务，稍后再推理。"), QStringLiteral("Worker is running a task. Run inference later.")},
        {QStringLiteral("请选择模型文件和图片。"), QStringLiteral("Choose a model file and image.")},
        {QStringLiteral("推理中。"), QStringLiteral("Inference in progress.")},
        {QStringLiteral("正在推理：%1"), QStringLiteral("Running inference: %1")},
        {QStringLiteral("推理中"), QStringLiteral("Running inference")},
        {QStringLiteral("请选择可用插件和任务类型。"), QStringLiteral("Choose an available plugin and task type.")},
        {QStringLiteral("请先选择并校验数据集。"), QStringLiteral("Choose and validate a dataset first.")},
        {QStringLiteral("当前训练插件不支持所选数据集格式。"), QStringLiteral("The current training plugin does not support the selected dataset format.")},
        {QStringLiteral("数据集未通过当前格式校验，不能启动训练。"), QStringLiteral("The dataset has not passed validation for the current format, so training cannot start.")},
        {QStringLiteral("等待当前任务完成。"), QStringLiteral("Waiting for the current task to complete.")},
        {QStringLiteral("等待 Worker 启动。"), QStringLiteral("Waiting for Worker to start.")},
        {QStringLiteral("任务已排队"), QStringLiteral("Task queued")},
        {QStringLiteral("任务已加入队列：%1"), QStringLiteral("Task added to queue: %1")},
        {QStringLiteral("任务队列"), QStringLiteral("Task Queue")},
        {QStringLiteral("请先选择一个任务。"), QStringLiteral("Select a task first.")},
        {QStringLiteral("用户取消排队任务。"), QStringLiteral("User canceled queued task.")},
        {QStringLiteral("只能取消排队任务或当前 Worker 正在运行的任务。"), QStringLiteral("Only queued tasks or the task currently running in Worker can be canceled.")},
        {QStringLiteral("环境自检"), QStringLiteral("Environment Check")},
        {QStringLiteral("Worker 正在执行任务，稍后再运行环境自检。"), QStringLiteral("Worker is running a task. Run the environment check later.")},
        {QStringLiteral("检测中"), QStringLiteral("Checking")},
        {QStringLiteral("等待 Worker 返回结果。"), QStringLiteral("Waiting for Worker results.")},
        {QStringLiteral("环境自检中"), QStringLiteral("Checking environment")},
        {QStringLiteral("产物：%1"), QStringLiteral("Artifact: %1")},
        {QStringLiteral("最新 checkpoint：%1"), QStringLiteral("Latest checkpoint: %1")},
        {QStringLiteral("最新预览：%1"), QStringLiteral("Latest preview: %1")},
        {QStringLiteral("预览图加载失败"), QStringLiteral("Failed to load preview image")},
        {QStringLiteral("Worker 上报产物"), QStringLiteral("Artifact reported by Worker")},
        {QStringLiteral("任务已暂停"), QStringLiteral("Task paused")},
        {QStringLiteral("训练运行中"), QStringLiteral("Training running")},
        {QStringLiteral("任务已取消"), QStringLiteral("Task canceled")},
        {QStringLiteral("任务已取消：%1"), QStringLiteral("Task canceled: %1")},
        {QStringLiteral("导出完成：%1；报告：%2"), QStringLiteral("Export complete: %1; report: %2")},
        {QStringLiteral("暂无插件"), QStringLiteral("No plugins")},
        {QStringLiteral("重新扫描或检查 plugins/models 目录。"), QStringLiteral("Rescan or check the plugins/models folder.")},
        {QStringLiteral("需要"), QStringLiteral("Required")},
        {QStringLiteral("否"), QStringLiteral("No")},
        {QStringLiteral("暂无数据集记录"), QStringLiteral("No dataset records")},
        {QStringLiteral("未通过"), QStringLiteral("Failed")},
        {QStringLiteral("标注后刷新"), QStringLiteral("Refresh After Annotation")},
        {QStringLiteral("暂无任务记录"), QStringLiteral("No task records")},
        {QStringLiteral("未记录类型"), QStringLiteral("Type not recorded")},
        {QStringLiteral("未记录后端"), QStringLiteral("Backend not recorded")},
        {QStringLiteral("任务 %1：%2 / %3 / %4，%5 个产物，%6 个指标点，%7 条导出记录"), QStringLiteral("Task %1: %2 / %3 / %4, %5 artifacts, %6 metric points, %7 export records")},
        {QStringLiteral("\n失败摘要：%1\n建议：优先查看 report/log 产物；若提示缺环境或缺 Python 包，进入“环境”页自检；若提示数据错误，回到“数据集”页重新校验。"), QStringLiteral("\nFailure summary: %1\nSuggestion: check report/log artifacts first. If it mentions missing environment or Python packages, run Environment self-check; if it mentions data errors, revalidate on the Dataset page.")},
        {QStringLiteral("Worker 未返回详细消息。"), QStringLiteral("Worker did not return a detailed message.")},
        {QStringLiteral("\n最新消息：%1"), QStringLiteral("\nLatest message: %1")},
        {QStringLiteral("暂无产物"), QStringLiteral("No artifacts")},
        {QStringLiteral("暂无指标"), QStringLiteral("No metrics")},
        {QStringLiteral("暂无导出"), QStringLiteral("No exports")},
        {QStringLiteral("暂无图片预览"), QStringLiteral("No image preview")},
        {QStringLiteral("请选择一个产物。"), QStringLiteral("Select an artifact.")},
        {QStringLiteral("产物不存在：%1"), QStringLiteral("Artifact does not exist: %1")},
        {QStringLiteral("目录产物\n路径：%1\n修改时间：%2"), QStringLiteral("Directory artifact\nPath: %1\nModified: %2")},
        {QStringLiteral("图片产物\n路径：%1\n尺寸：%2 x %3\n大小：%4 bytes"), QStringLiteral("Image artifact\nPath: %1\nSize: %2 x %3\nFile size: %4 bytes")},
        {QStringLiteral("ONNX 模型\n路径：%1\n模型族：%2\n大小：%3 bytes"), QStringLiteral("ONNX model\nPath: %1\nModel family: %2\nFile size: %3 bytes")},
        {QStringLiteral("模型产物\n路径：%1\n类型：%2\n大小：%3 bytes"), QStringLiteral("Model artifact\nPath: %1\nType: %2\nFile size: %3 bytes")},
        {QStringLiteral("无法读取文本产物：%1"), QStringLiteral("Failed to read text artifact: %1")},
        {QStringLiteral("\n\n[文件超过 256KB，仅显示前部内容]\n路径：%1"), QStringLiteral("\n\n[File exceeds 256 KB; showing the first part only]\nPath: %1")},
        {QStringLiteral("不支持内联预览的产物\n路径：%1\n大小：%2 bytes"), QStringLiteral("Inline preview is not supported for this artifact\nPath: %1\nFile size: %2 bytes")},
        {QStringLiteral("产物路径已复制"), QStringLiteral("Artifact path copied")},
        {QStringLiteral("已加载 %1 个 AITrain 插件。"), QStringLiteral("%1 AITrain plugins loaded.")},
        {QStringLiteral("未加载 AITrain 插件，请检查 plugins/models 目录。"), QStringLiteral("No AITrain plugins loaded. Check the plugins/models folder.")},
        {QStringLiteral("AITrain Plugins"), QStringLiteral("AITrain Plugins")},
        {QStringLiteral("环境缺失"), QStringLiteral("Environment Missing")},
        {QStringLiteral("环境警告"), QStringLiteral("Environment Warning")},
        {QStringLiteral("环境通过"), QStringLiteral("Environment Passed")},
        {QStringLiteral("GPU / 运行时：%1"), QStringLiteral("GPU / Runtime: %1")},
        {QStringLiteral("环境自检完成"), QStringLiteral("Environment check complete")},
        {QStringLiteral("校验通过：%1 个样本。"), QStringLiteral("Validation passed: %1 samples.")},
        {QStringLiteral("校验失败：发现 %1 个问题，训练已被阻止。"), QStringLiteral("Validation failed: %1 issues found. Training is blocked.")},
        {QStringLiteral("暂无样本预览"), QStringLiteral("No sample preview")},
        {QStringLiteral("标注文件已匹配"), QStringLiteral("Annotation file matched")},
        {QStringLiteral("未发现数据集问题。"), QStringLiteral("No dataset issues found.")},
        {QStringLiteral("数据集校验通过"), QStringLiteral("Dataset validation passed")},
        {QStringLiteral("数据集校验失败"), QStringLiteral("Dataset validation failed")},
        {QStringLiteral("数据质量报告"), QStringLiteral("Dataset Quality Report")},
        {QStringLiteral("Worker 正在执行任务，稍后再生成数据质量报告。"), QStringLiteral("Worker is running a task. Generate the dataset quality report later.")},
        {QStringLiteral("数据质量报告生成中。"), QStringLiteral("Generating dataset quality report.")},
        {QStringLiteral("数据质量报告生成中"), QStringLiteral("Generating quality report")},
        {QStringLiteral("问题清单"), QStringLiteral("Issue List")},
        {QStringLiteral("请先生成数据质量报告。"), QStringLiteral("Generate the dataset quality report first.")},
        {QStringLiteral("问题清单：%1"), QStringLiteral("Issue list: %1")},
        {QStringLiteral("已启动 X-AnyLabeling，请按问题清单修复样本。"), QStringLiteral("X-AnyLabeling started. Fix samples according to the issue list.")},
        {QStringLiteral("数据集快照"), QStringLiteral("Dataset Snapshot")},
        {QStringLiteral("Worker 正在执行任务，稍后再创建数据集快照。"), QStringLiteral("Worker is running a task. Create the dataset snapshot later.")},
        {QStringLiteral("数据集快照创建中。"), QStringLiteral("Creating dataset snapshot.")},
        {QStringLiteral("数据集快照创建中"), QStringLiteral("Creating dataset snapshot")},
        {QStringLiteral("划分完成：train %1 / val %2 / test %3。"), QStringLiteral("Split complete: train %1 / val %2 / test %3.")},
        {QStringLiteral("划分失败：%1 个错误。"), QStringLiteral("Split failed: %1 errors.")},
        {QStringLiteral("数据集已复制到标准划分目录。"), QStringLiteral("Dataset copied to the standard split folder.")},
        {QStringLiteral("数据集划分完成"), QStringLiteral("Dataset split complete")},
        {QStringLiteral("数据集划分失败"), QStringLiteral("Dataset split failed")},
        {QStringLiteral("已打开：%1"), QStringLiteral("Opened: %1")},
        {QStringLiteral("未打开项目。"), QStringLiteral("No project open.")},
        {QStringLiteral("未打开"), QStringLiteral("Not open")},
        {QStringLiteral("已连接"), QStringLiteral("Connected")},
        {QStringLiteral("未连接"), QStringLiteral("Not connected")},
        {QStringLiteral("未加载插件，请检查插件目录。"), QStringLiteral("No plugins loaded. Check the plugin folder.")},
        {QStringLiteral("已加载 %1 个插件。"), QStringLiteral("%1 plugins loaded.")},
        {QStringLiteral("插件搜索路径：%1"), QStringLiteral("Plugin search paths: %1")},
        {QStringLiteral("发现 %1 项缺失，相关能力会被阻塞。"), QStringLiteral("%1 missing item(s) found; related capabilities will be blocked.")},
        {QStringLiteral("发现 %1 项警告，可继续但需要关注。"), QStringLiteral("%1 warning item(s) found; you can continue, but review them.")},
        {QStringLiteral("尚有 %1 项未检测。"), QStringLiteral("%1 item(s) unchecked.")},
        {QStringLiteral("环境自检通过。"), QStringLiteral("Environment check passed.")},
        {QStringLiteral("先创建或打开一个本地项目。项目目录会集中保存数据集索引、任务历史、训练报告和模型产物。"), QStringLiteral("Create or open a local project first. The project folder stores dataset indexes, task history, training reports, and model artifacts.")},
        {QStringLiteral("下一步：导入并校验 detection、segmentation 或 OCR Rec 数据集。只有通过校验的数据集会进入训练主流程。"), QStringLiteral("Next: import and validate a detection, segmentation, or OCR Rec dataset. Only validated datasets enter the main training flow.")},
        {QStringLiteral("下一步：进入训练实验，选择已校验数据集。平台会按任务类型优先选择官方 YOLO / OCR 后端。"), QStringLiteral("Next: go to Training Runs and select a validated dataset. The platform prefers official YOLO / OCR backends by task type.")},
        {QStringLiteral("下一步：在任务与产物中查看 checkpoint / report / ONNX，再进入模型导出或推理验证。"), QStringLiteral("Next: inspect checkpoint / report / ONNX under Tasks and Artifacts, then continue to Model Export or Inference Check.")},
        {QStringLiteral("下一步：在任务与产物中查看 checkpoint / report / ONNX，注册到模型库后再做评估、基准、导出或推理验证。"), QStringLiteral("Next: inspect checkpoint / report / ONNX under Tasks and Artifacts, register it in the model library, then continue with evaluation, benchmarking, export, or inference validation.")},
        {QStringLiteral("项目已具备可复验闭环：数据集、任务历史和模型导出均已记录。可继续运行推理验证或追加实验。"), QStringLiteral("This project now has a reproducible loop: datasets, task history, and model exports are recorded. Continue with inference validation or add another run.")},
        {QStringLiteral("已校验"), QStringLiteral("Validated")},
        {QStringLiteral("待校验"), QStringLiteral("Pending validation")},
        {QStringLiteral("等待自动创建数据快照。"), QStringLiteral("Waiting for an automatic dataset snapshot.")},
        {QStringLiteral("数据快照"), QStringLiteral("Data Snapshot")},
        {QStringLiteral("自动数据快照创建中"), QStringLiteral("Automatic dataset snapshot in progress")},
        {QStringLiteral("为训练自动创建数据快照。"), QStringLiteral("Creating a dataset snapshot automatically for training.")},
        {QStringLiteral("无法为训练创建数据快照：数据集记录缺失。"), QStringLiteral("Cannot create a dataset snapshot for training: dataset record is missing.")},
        {QStringLiteral("训练任务 %1 正在等待自动数据快照 %2。"), QStringLiteral("Training task %1 is waiting for automatic dataset snapshot %2.")},
        {QStringLiteral("自动数据快照已取消，训练未启动。"), QStringLiteral("Automatic dataset snapshot was canceled. Training did not start.")},
        {QStringLiteral("自动数据快照失败：%1"), QStringLiteral("Automatic dataset snapshot failed: %1")},
        {QStringLiteral("快照：未选择数据集"), QStringLiteral("Snapshot: no dataset selected")},
        {QStringLiteral("快照：#%1 | %2 文件 | hash %3 | %4"), QStringLiteral("Snapshot: #%1 | %2 files | hash %3 | %4")},
        {QStringLiteral("快照：暂无，启动训练时将自动创建。"), QStringLiteral("Snapshot: none yet; one will be created automatically when training starts.")},
        {QStringLiteral("当前数据集：未选择。请先在数据集页导入并通过校验。"), QStringLiteral("Current dataset: none selected. Import and validate a dataset on the Datasets page first.")},
        {QStringLiteral("当前数据集：%1 | %2 | %3\n%4"), QStringLiteral("Current dataset: %1 | %2 | %3\n%4")},
        {QStringLiteral("当前数据集：%1 | %2 | %3"), QStringLiteral("Current dataset: %1 | %2 | %3")},
        {QStringLiteral("选择或导入数据集后显示格式、样本数、校验状态和最近报告。"), QStringLiteral("After selecting or importing a dataset, format, sample count, validation status, and latest report are shown here.")},
        {QStringLiteral("格式：%1 | 状态：%2 | 路径：%3\n%4"), QStringLiteral("Format: %1 | Status: %2 | Path: %3\n%4")},
        {QStringLiteral("格式：%1 | 状态：%2 | 路径：%3"), QStringLiteral("Format: %1 | Status: %2 | Path: %3")},
        {QStringLiteral("运行摘要：%1 | 后端 %2 | 模型 %3 | epoch %4 / batch %5 / image %6"), QStringLiteral("Run summary: %1 | backend %2 | model %3 | epoch %4 / batch %5 / image %6")},
        {QStringLiteral("默认"), QStringLiteral("Default")},
        {QStringLiteral("中"), QStringLiteral("ZH")},
        {QStringLiteral("中文"), QStringLiteral("Chinese")},
        {QStringLiteral("工作台"), QStringLiteral("Workbench")},
        {QStringLiteral("总览"), QStringLiteral("Overview")},
        {QStringLiteral("数据与训练"), QStringLiteral("Data and Training")},
        {QStringLiteral("模型交付"), QStringLiteral("Model Delivery")},
        {QStringLiteral("系统"), QStringLiteral("System")},
        {QStringLiteral("模型库"), QStringLiteral("Model Library")},
        {QStringLiteral("模型版本"), QStringLiteral("Model Versions")},
        {QStringLiteral("模型库 / 导出产物"), QStringLiteral("Model library / exported artifacts")},
        {QStringLiteral("刷新模型库"), QStringLiteral("Refresh Model Library")},
        {QStringLiteral("查看评估报告"), QStringLiteral("View Evaluation Reports")},
        {QStringLiteral("评估报告"), QStringLiteral("Evaluation Reports")},
        {QStringLiteral("刷新评估报告"), QStringLiteral("Refresh Evaluation Reports")},
        {QStringLiteral("查看模型库"), QStringLiteral("View Model Library")},
        {QStringLiteral("选中模型用于推理"), QStringLiteral("Use Selected Model for Inference")},
        {QStringLiteral("选中模型用于导出"), QStringLiteral("Use Selected Model for Export")},
        {QStringLiteral("执行本地流水线"), QStringLiteral("Run Local Pipeline")},
        {QStringLiteral("训练产物可从“任务与产物”注册为模型版本；评估报告已拆分到独立页面，模型库聚焦版本管理、导出和推理入口。"), QStringLiteral("Training artifacts can be registered as model versions from Tasks and Artifacts; evaluation reports now live on a dedicated page, while the model library focuses on version management, export, and inference entry points.")},
        {QStringLiteral("来源任务"), QStringLiteral("Source Task")},
        {QStringLiteral("最近评估报告"), QStringLiteral("Recent Evaluation Reports")},
        {QStringLiteral("评估报告详情"), QStringLiteral("Evaluation Report Details")},
        {QStringLiteral("流水线记录"), QStringLiteral("Pipeline Records")},
        {QStringLiteral("模板"), QStringLiteral("Template")},
        {QStringLiteral("管理模型版本、来源 lineage、评估报告和部署基准"), QStringLiteral("Manage model versions, source lineage, evaluation reports, and deployment benchmarks")},
        {QStringLiteral("管理模型版本、来源 lineage，以及导出、推理和流水线入口"), QStringLiteral("Manage model versions, source lineage, and entry points for export, inference, and pipelines")},
        {QStringLiteral("集中查看最近评估报告、任务类型、报告路径和详细可视化结果"), QStringLiteral("Review recent evaluation reports, task types, report paths, and detailed visualized results in one place")},
        {QStringLiteral("集中查看最近评估报告、任务类型、报告路径和详细可视化结果；模型版本管理保留在“模型库”。"), QStringLiteral("Review recent evaluation reports, task types, report paths, and detailed visualized results in one place; model version management remains in the Model Library.")},
        {QStringLiteral("复现实验"), QStringLiteral("Reproduce Run")},
        {QStringLiteral("请先打开项目。"), QStringLiteral("Open a project first.")},
        {QStringLiteral("请先打开项目"), QStringLiteral("Open a project first")},
        {QStringLiteral("请先打开或创建项目。"), QStringLiteral("Open or create a project first.")},
        {QStringLiteral("请先选择一个训练任务。"), QStringLiteral("Select a training task first.")},
        {QStringLiteral("只能复现历史训练任务。"), QStringLiteral("Only historical training tasks can be reproduced.")},
        {QStringLiteral("该训练任务没有可复现的 request 记录。"), QStringLiteral("This training task has no reproducible request record.")},
        {QStringLiteral("原训练 request JSON 无法解析：%1"), QStringLiteral("The original training request JSON could not be parsed: %1")},
        {QStringLiteral("原实验的数据快照 manifest 缺失，无法按同一快照复现。请重新创建快照或选择其他训练任务。"), QStringLiteral("The original run's dataset snapshot manifest is missing, so it cannot be reproduced from the same snapshot. Create a new snapshot or choose another training task.")},
        {QStringLiteral("复现实验已排队。"), QStringLiteral("Reproduction run queued.")},
        {QStringLiteral("复现实验等待 Worker 启动。"), QStringLiteral("Reproduction run is waiting for Worker to start.")},
        {QStringLiteral("复现实验已排队"), QStringLiteral("Reproduction queued")},
        {QStringLiteral("本地闭环流水线"), QStringLiteral("Local End-to-End Pipeline")},
        {QStringLiteral("Phase 35 自动记录的本地复现实验。"), QStringLiteral("Local reproduction run recorded automatically in Phase 35.")},
        {QStringLiteral("模型版本：%1；评估报告：%2；流水线记录：%3。评估、基准和报告 v1 通过 Worker 生成 artifact，完整质量分析仍按 scaffold 标注。"), QStringLiteral("Model versions: %1; evaluation reports: %2; pipeline records: %3. Evaluation, benchmarking, and report v1 artifacts are generated through the Worker; full quality analysis remains explicitly marked as scaffold.")},
        {QStringLiteral("暂无模型版本"), QStringLiteral("No model versions")},
        {QStringLiteral("暂无评估报告"), QStringLiteral("No evaluation reports")},
        {QStringLiteral("暂无流水线记录"), QStringLiteral("No pipeline records")},
        {QStringLiteral("评估报告摘要"), QStringLiteral("Evaluation Report Summary")},
        {QStringLiteral("任务类型：%1"), QStringLiteral("Task type: %1")},
        {QStringLiteral("真实评估：%1"), QStringLiteral("Real evaluation: %1")},
        {QStringLiteral("否，scaffold"), QStringLiteral("No, scaffold")},
        {QStringLiteral("是"), QStringLiteral("Yes")},
        {QStringLiteral("错误样本：%1；低置信样本：%2"), QStringLiteral("Error samples: %1; low-confidence samples: %2")},
        {QStringLiteral("数据质量报告摘要"), QStringLiteral("Dataset Quality Report Summary")},
        {QStringLiteral("格式：%1；真实分析：%2"), QStringLiteral("Format: %1; real analysis: %2")},
        {QStringLiteral("error=%1 warning=%2 info=%3 问题样本=%4 重复图片=%5"), QStringLiteral("error=%1 warning=%2 info=%3 problem samples=%4 duplicate images=%5")},
        {QStringLiteral("问题样本摘要"), QStringLiteral("Problem Sample Summary")},
        {QStringLiteral("问题样本数：%1"), QStringLiteral("Problem sample count: %1")},
        {QStringLiteral("质量报告完成：error %1 / warning %2 / info %3，问题样本 %4，重复图片 %5。"), QStringLiteral("Quality report complete: error %1 / warning %2 / info %3, problem samples %4, duplicate images %5.")},
        {QStringLiteral("未发现需要修复的问题样本。"), QStringLiteral("No problem samples requiring fixes were found.")},
        {QStringLiteral("修复清单：%1"), QStringLiteral("Fix list: %1")},
        {QStringLiteral("数据集快照完成：%1 个文件，hash %2。"), QStringLiteral("Dataset snapshot complete: %1 files, hash %2.")},
        {QStringLiteral("模型注册"), QStringLiteral("Model Registration")},
        {QStringLiteral("请先选择一个 checkpoint、ONNX 或 engine 产物。"), QStringLiteral("Select a checkpoint, ONNX, or engine artifact first.")},
        {QStringLiteral("模型名称"), QStringLiteral("Model Name")},
        {QStringLiteral("版本号"), QStringLiteral("Version")},
        {QStringLiteral("从任务产物手动注册。"), QStringLiteral("Registered manually from a task artifact.")},
        {QStringLiteral("已注册模型版本：%1:%2"), QStringLiteral("Registered model version: %1:%2")},
        {QStringLiteral("模型评估"), QStringLiteral("Model Evaluation")},
        {QStringLiteral("Worker 正在执行任务，稍后再评估模型。"), QStringLiteral("Worker is running a task. Evaluate the model later.")},
        {QStringLiteral("请先选择模型产物，并在数据集页选择评估数据集。"), QStringLiteral("Select a model artifact first, then choose an evaluation dataset on the Datasets page.")},
        {QStringLiteral("模型评估报告生成中。"), QStringLiteral("Generating model evaluation report.")},
        {QStringLiteral("模型评估中"), QStringLiteral("Evaluating model")},
        {QStringLiteral("部署基准"), QStringLiteral("Deployment Benchmark")},
        {QStringLiteral("Worker 正在执行任务，稍后再运行部署基准。"), QStringLiteral("Worker is running a task. Run the deployment benchmark later.")},
        {QStringLiteral("请先选择一个模型产物。"), QStringLiteral("Select a model artifact first.")},
        {QStringLiteral("部署基准报告生成中。"), QStringLiteral("Generating deployment benchmark report.")},
        {QStringLiteral("部署基准运行中"), QStringLiteral("Running deployment benchmark")},
        {QStringLiteral("交付报告"), QStringLiteral("Delivery Report")},
        {QStringLiteral("Worker 正在执行任务，稍后再生成交付报告。"), QStringLiteral("Worker is running a task. Generate the delivery report later.")},
        {QStringLiteral("请先选择一个模型或报告产物。"), QStringLiteral("Select a model or report artifact first.")},
        {QStringLiteral("训练交付报告生成中。"), QStringLiteral("Generating training delivery report.")},
        {QStringLiteral("交付报告生成中"), QStringLiteral("Generating delivery report")},
        {QStringLiteral("本地流水线"), QStringLiteral("Local Pipeline")},
        {QStringLiteral("Worker 正在执行任务，稍后再执行流水线。"), QStringLiteral("Worker is running a task. Run the pipeline later.")},
        {QStringLiteral("训练->评估->导出->注册->报告"), QStringLiteral("Train -> Evaluate -> Export -> Register -> Report")},
        {QStringLiteral("导出->推理->基准->报告"), QStringLiteral("Export -> Inference -> Benchmark -> Report")},
        {QStringLiteral("选择流水线模板"), QStringLiteral("Choose Pipeline Template")},
        {QStringLiteral("本地流水线执行中。"), QStringLiteral("Running local pipeline.")},
        {QStringLiteral("本地流水线执行中"), QStringLiteral("Running local pipeline")},
        {QStringLiteral("模型产物"), QStringLiteral("Model Artifacts")},
        {QStringLiteral("打开项目"), QStringLiteral("Open Project")},
        {QStringLiteral("导入 / 校验数据"), QStringLiteral("Import / Validate Data")},
        {QStringLiteral("启动训练实验"), QStringLiteral("Start Training Run")},
        {QStringLiteral("查看任务与产物"), QStringLiteral("View Tasks and Artifacts")},
        {QStringLiteral("类型"), QStringLiteral("Type")},
        {QStringLiteral("消息"), QStringLiteral("Message")},
        {QStringLiteral("统一管理本机训练项目、SQLite 元数据、数据集目录、运行目录和模型产物目录。"), QStringLiteral("Manage local training projects, SQLite metadata, dataset folders, run folders, and model artifact folders in one place.")},
        {QStringLiteral("状态"), QStringLiteral("Status")},
        {QStringLiteral("目录"), QStringLiteral("Folder")},
        {QStringLiteral("项目会生成 datasets、runs、models 和 project.sqlite。"), QStringLiteral("The project creates datasets, runs, models, and project.sqlite.")},
        {QStringLiteral("项目设置"), QStringLiteral("Project Settings")},
        {QStringLiteral("本地训练项目"), QStringLiteral("Local Training Project")},
        {QStringLiteral("选择项目目录"), QStringLiteral("Choose Project Folder")},
        {QStringLiteral("项目名称"), QStringLiteral("Project Name")},
        {QStringLiteral("项目目录"), QStringLiteral("Project Folder")},
        {QStringLiteral("打开项目后，数据集、任务、导出记录和环境检查都会写入 project.sqlite。"), QStringLiteral("After opening a project, datasets, tasks, export records, and environment checks are written to project.sqlite.")},
        {QStringLiteral("当前项目根目录"), QStringLiteral("Current project root")},
        {QStringLiteral("项目元数据状态"), QStringLiteral("Project metadata status")},
        {QStringLiteral("训练、校验、导出、推理"), QStringLiteral("Training, validation, export, inference")},
        {QStringLiteral("已记录导出产物"), QStringLiteral("Recorded export artifacts")},
        {QStringLiteral("项目页只负责创建和打开工作区；训练、导出和推理仍通过 Worker 执行。"), QStringLiteral("The Project page only creates and opens workspaces; training, export, and inference still run through the Worker.")},
        {QStringLiteral("数据集操作"), QStringLiteral("Dataset Actions")},
        {QStringLiteral("选择数据集"), QStringLiteral("Choose Dataset")},
        {QStringLiteral("默认输出到当前项目 datasets/normalized"), QStringLiteral("Default output: current project datasets/normalized")},
        {QStringLiteral("输出目录"), QStringLiteral("Output Folder")},
        {QStringLiteral("所选数据集详情"), QStringLiteral("Selected Dataset Details")},
        {QStringLiteral("请选择数据集目录和格式，然后执行校验。"), QStringLiteral("Choose a dataset folder and format, then run validation.")},
        {QStringLiteral("级别"), QStringLiteral("Level")},
        {QStringLiteral("代码"), QStringLiteral("Code")},
        {QStringLiteral("文件"), QStringLiteral("File")},
        {QStringLiteral("行号"), QStringLiteral("Line")},
        {QStringLiteral("说明"), QStringLiteral("Description")},
        {QStringLiteral("校验报告 JSON 会显示在这里。"), QStringLiteral("The validation report JSON is shown here.")},
        {QStringLiteral("数据集库与样本预览"), QStringLiteral("Dataset Library and Sample Preview")},
        {QStringLiteral("样本"), QStringLiteral("Sample")},
        {QStringLiteral("路径"), QStringLiteral("Path")},
        {QStringLiteral("标签 / 说明"), QStringLiteral("Label / Notes")},
        {QStringLiteral("外部标注工具"), QStringLiteral("External Annotation Tool")},
        {QStringLiteral("默认使用 X-AnyLabeling。推荐导出：检测使用 YOLO bbox，分割使用 YOLO polygon；PaddleOCR Det 使用 det_gt.txt，PaddleOCR Rec 使用 rec_gt.txt + dict.txt。"), QStringLiteral("X-AnyLabeling is used by default. Recommended exports: YOLO bbox for detection, YOLO polygon for segmentation, det_gt.txt for PaddleOCR Det, and rec_gt.txt + dict.txt for PaddleOCR Rec.")},
        {QStringLiteral("启动 X-AnyLabeling"), QStringLiteral("Launch X-AnyLabeling")},
        {QStringLiteral("检测状态"), QStringLiteral("Check Status")},
        {QStringLiteral("标注后刷新 / 重新校验"), QStringLiteral("Refresh / Revalidate After Annotation")},
        {QStringLiteral("打开数据目录"), QStringLiteral("Open Dataset Folder")},
        {QStringLiteral("样本预览"), QStringLiteral("Sample Preview")},
        {QStringLiteral("划分会复制到新目录，不修改原始数据；支持 YOLO 检测、YOLO 分割、PaddleOCR Det 和 PaddleOCR Rec。"), QStringLiteral("Splitting copies data to a new folder without changing the original data; supports YOLO detection, YOLO segmentation, PaddleOCR Det, and PaddleOCR Rec.")},
        {QStringLiteral("可选：选择已有 checkpoint 继续训练"), QStringLiteral("Optional: choose an existing checkpoint to resume training")},
        {QStringLiteral("水平翻转增强"), QStringLiteral("Horizontal flip augmentation")},
        {QStringLiteral("亮度扰动增强"), QStringLiteral("Brightness jitter augmentation")},
        {QStringLiteral("官方后端会由 Worker 启动独立 Python 进程；scaffold 后端只用于高级诊断。"), QStringLiteral("Official backends start independent Python processes through the Worker; scaffold backends are only for advanced diagnostics.")},
        {QStringLiteral("等待配置训练实验。"), QStringLiteral("Waiting for training run configuration.")},
        {QStringLiteral("启动训练"), QStringLiteral("Start Training")},
        {QStringLiteral("暂停任务"), QStringLiteral("Pause Task")},
        {QStringLiteral("继续任务"), QStringLiteral("Resume Task")},
        {QStringLiteral("取消任务"), QStringLiteral("Cancel Task")},
        {QStringLiteral("按数据集类型优先选择官方 YOLO / OCR 后端；运行结果沉淀到任务与产物。"), QStringLiteral("Prefer official YOLO / OCR backends by dataset type; run results are saved to Tasks and Artifacts.")},
        {QStringLiteral("摘要"), QStringLiteral("Summary")},
        {QStringLiteral("实验参数"), QStringLiteral("Run Parameters")},
        {QStringLiteral("当前模型能力说明"), QStringLiteral("Current Model Capability")},
        {QStringLiteral("高级 / 诊断后端"), QStringLiteral("Advanced / Diagnostic Backend")},
        {QStringLiteral("能力插件"), QStringLiteral("Capability Plugin")},
        {QStringLiteral("训练监控"), QStringLiteral("Training Monitor")},
        {QStringLiteral("运行后会记录 checkpoint、训练报告、ONNX、预览图和请求参数。完整产物浏览请进入“任务与产物”。"), QStringLiteral("After a run, checkpoints, training reports, ONNX files, previews, and request parameters are recorded. Open Tasks and Artifacts for the full artifact browser.")},
        {QStringLiteral("主流程优先使用官方 YOLO / PaddleOCR 后端；PaddleOCR System 产物来自官方工具链，不代表 C++ DB 后处理已经接入。"), QStringLiteral("The main flow prefers official YOLO / PaddleOCR backends. PaddleOCR System artifacts come from the official toolchain and do not mean C++ DB postprocess is integrated.")},
        {QStringLiteral("最新 checkpoint：暂无"), QStringLiteral("Latest checkpoint: none")},
        {QStringLiteral("最新预览：暂无"), QStringLiteral("Latest preview: none")},
        {QStringLiteral("暂无预览图"), QStringLiteral("No preview image")},
        {QStringLiteral("训练日志"), QStringLiteral("Training Log")},
        {QStringLiteral("历史操作"), QStringLiteral("History Actions")},
        {QStringLiteral("刷新历史"), QStringLiteral("Refresh History")},
        {QStringLiteral("取消选中任务"), QStringLiteral("Cancel Selected Task")},
        {QStringLiteral("搜索任务、后端、消息"), QStringLiteral("Search tasks, backends, messages")},
        {QStringLiteral("更新时间"), QStringLiteral("Updated At")},
        {QStringLiteral("请选择一个任务查看产物、指标和导出记录。"), QStringLiteral("Select a task to view artifacts, metrics, and export records.")},
        {QStringLiteral("时间"), QStringLiteral("Time")},
        {QStringLiteral("指标"), QStringLiteral("Metric")},
        {QStringLiteral("值"), QStringLiteral("Value")},
        {QStringLiteral("暂无产物预览"), QStringLiteral("No artifact preview")},
        {QStringLiteral("选择一个产物后显示摘要。"), QStringLiteral("Select an artifact to show its summary.")},
        {QStringLiteral("用作推理模型"), QStringLiteral("Use as Inference Model")},
        {QStringLiteral("用作导出输入"), QStringLiteral("Use as Export Input")},
        {QStringLiteral("开始导出"), QStringLiteral("Start Export")},
        {QStringLiteral("从训练产物生成 ONNX、NCNN、TensorRT 或诊断 JSON，并把报告写入任务与产物。"), QStringLiteral("Generate ONNX, NCNN, TensorRT, or diagnostic JSON from training artifacts and write the report to Tasks and Artifacts.")},
        {QStringLiteral("输入"), QStringLiteral("Input")},
        {QStringLiteral("策略"), QStringLiteral("Policy")},
        {QStringLiteral("优先从“任务与产物”选择 checkpoint、ONNX 或官方训练产物。"), QStringLiteral("Prefer choosing checkpoints, ONNX files, or official training artifacts from Tasks and Artifacts.")},
        {QStringLiteral("导出只通过 Worker 执行；TensorRT 仍需 RTX / SM 75+ 外部验收。"), QStringLiteral("Export only runs through the Worker. TensorRT still needs external acceptance on RTX / SM 75+ hardware.")},
        {QStringLiteral("导出设置"), QStringLiteral("Export Settings")},
        {QStringLiteral("从任务产物带入，或选择 checkpoint / ONNX / AITrain export"), QStringLiteral("Bring in from task artifacts, or choose checkpoint / ONNX / AITrain export")},
        {QStringLiteral("选择模型产物"), QStringLiteral("Choose Model Artifact")},
        {QStringLiteral("留空则输出到项目 models/exported；未打开项目时使用输入同目录"), QStringLiteral("Leave empty to output to project models/exported; without a project, use the input folder")},
        {QStringLiteral("选择输出"), QStringLiteral("Choose Output")},
        {QStringLiteral("选择导出路径"), QStringLiteral("Choose Export Path")},
        {QStringLiteral("模型输入"), QStringLiteral("Model Input")},
        {QStringLiteral("目标格式"), QStringLiteral("Target Format")},
        {QStringLiteral("输出路径"), QStringLiteral("Output Path")},
        {QStringLiteral("从“任务与产物”中选中 best.onnx、checkpoint 或官方导出目录后，可点击“用作导出输入”自动带入这里。"), QStringLiteral("After selecting best.onnx, a checkpoint, or an official export folder in Tasks and Artifacts, click Use as Export Input to fill this field.")},
        {QStringLiteral("导出任务会记录到任务历史，完成后可直接作为推理输入。"), QStringLiteral("Export tasks are recorded in task history and can be used directly as inference input after completion.")},
        {QStringLiteral("格式矩阵"), QStringLiteral("Format Matrix")},
        {QStringLiteral("产物"), QStringLiteral("Artifact")},
        {QStringLiteral("运行状态"), QStringLiteral("Run Status")},
        {QStringLiteral("暂无导出任务。"), QStringLiteral("No export task yet.")},
        {QStringLiteral("ONNX 会写入 AITrain sidecar；NCNN 依赖本机 onnx2ncnn；TensorRT 在当前 GTX 1060 / SM 61 上保持 hardware-blocked。"), QStringLiteral("ONNX writes an AITrain sidecar. NCNN depends on local onnx2ncnn. TensorRT remains hardware-blocked on the current GTX 1060 / SM 61.")},
        {QStringLiteral("推理输入"), QStringLiteral("Inference Input")},
        {QStringLiteral("推理验证工作台"), QStringLiteral("Inference Validation Workbench")},
        {QStringLiteral("选择模型和图片，运行 Worker 推理并查看 prediction JSON 与 overlay。这里不直接承载模型后处理逻辑。"), QStringLiteral("Select a model and image, run Worker inference, and inspect prediction JSON plus overlay output. Model postprocess logic is not hosted here.")},
        {QStringLiteral("验证输入"), QStringLiteral("Validation Input")},
        {QStringLiteral("从任务产物带入，或选择 ONNX / AITrain export / TensorRT engine"), QStringLiteral("Bring in from task artifacts, or choose ONNX / AITrain export / TensorRT engine")},
        {QStringLiteral("选择验证图片"), QStringLiteral("Choose validation image")},
        {QStringLiteral("输出目录；留空则输出到模型同目录 inference"), QStringLiteral("Output folder; leave empty to use an inference folder beside the model")},
        {QStringLiteral("选择模型文件"), QStringLiteral("Choose Model File")},
        {QStringLiteral("选择图片"), QStringLiteral("Choose Image")},
        {QStringLiteral("选择输出目录"), QStringLiteral("Choose Output Folder")},
        {QStringLiteral("选择推理输出目录"), QStringLiteral("Choose Inference Output Folder")},
        {QStringLiteral("推理运行中\n等待 Worker 写入 overlay 产物。"), QStringLiteral("Inference running\nWaiting for the Worker to write the overlay artifact.")},
        {QStringLiteral("边界"), QStringLiteral("Boundary")},
        {QStringLiteral("Worker 隔离执行推理；任务、产物和失败原因写入任务历史。"), QStringLiteral("The Worker runs inference in isolation; tasks, artifacts, and failure reasons are written to task history.")},
        {QStringLiteral("C++ ONNX 支持 detection / segmentation / OCR Rec；PaddleOCR System 仍查看官方工具链产物。"), QStringLiteral("C++ ONNX supports detection / segmentation / OCR Rec; PaddleOCR System results are still viewed through official toolchain artifacts.")},
        {QStringLiteral("模型"), QStringLiteral("Model")},
        {QStringLiteral("图片"), QStringLiteral("Image")},
        {QStringLiteral("输出"), QStringLiteral("Output")},
        {QStringLiteral("模型路径"), QStringLiteral("Model Path")},
        {QStringLiteral("图片路径"), QStringLiteral("Image Path")},
        {QStringLiteral("推理输出"), QStringLiteral("Inference Output")},
        {QStringLiteral("从“任务与产物”选中 ONNX、AITrain export 或 engine 后，可点击“用作推理模型”自动带入这里。输出目录留空会写到模型同目录 inference。"), QStringLiteral("After selecting an ONNX, AITrain export, or engine under Tasks and Artifacts, click Use as Inference Model to fill this field. Leaving the output folder empty writes to an inference folder beside the model.")},
        {QStringLiteral("推理任务会记录到任务历史，完成后可在产物详情中复查 JSON、overlay 和耗时。"), QStringLiteral("Inference tasks are recorded in task history; after completion, review JSON, overlay, and latency in artifact details.")},
        {QStringLiteral("可解析结果"), QStringLiteral("Parseable Results")},
        {QStringLiteral("推理验证会根据 ONNX 模型族或 AITrain 导出信息选择后处理；scaffold / 官方工具链边界保持显式。"), QStringLiteral("Inference validation chooses postprocess from the ONNX model family or AITrain export metadata; scaffold / official toolchain boundaries stay explicit.")},
        {QStringLiteral("box、类别、置信度、NMS 与 overlay。"), QStringLiteral("Boxes, classes, confidence, NMS, and overlay.")},
        {QStringLiteral("box、mask、mask area 与半透明 overlay。"), QStringLiteral("Boxes, masks, mask area, and translucent overlay.")},
        {QStringLiteral("CTC greedy decode、文本与置信度摘要。"), QStringLiteral("CTC greedy decode, text, and confidence summary.")},
        {QStringLiteral("端到端结果仍通过官方工具链任务产物查看。"), QStringLiteral("End-to-end results are still viewed through official toolchain task artifacts.")},
        {QStringLiteral("验证链路"), QStringLiteral("Validation Flow")},
        {QStringLiteral("样本图片"), QStringLiteral("Sample Image")},
        {QStringLiteral("单张验证图进入预处理"), QStringLiteral("Single validation image enters preprocessing")},
        {QStringLiteral("Worker 推理"), QStringLiteral("Worker Inference")},
        {QStringLiteral("隔离执行，不阻塞 GUI"), QStringLiteral("Runs in isolation without blocking the GUI")},
        {QStringLiteral("结果归档"), QStringLiteral("Result Archive")},
        {QStringLiteral("prediction JSON + overlay"), QStringLiteral("prediction JSON + overlay")},
        {QStringLiteral("结果摘要"), QStringLiteral("Result Summary")},
        {QStringLiteral("Worker 返回的 prediction JSON 会压缩显示任务类型、结果数量、首个类别 / 文本、耗时和结果文件路径。"), QStringLiteral("Prediction JSON returned by the Worker is summarized with task type, result count, first class / text, latency, and result file path.")},
        {QStringLiteral("尚未推理。"), QStringLiteral("No inference yet.")},
        {QStringLiteral("完整原始 JSON 可在“任务与产物”的产物详情中查看。"), QStringLiteral("The full raw JSON is available in artifact details under Tasks and Artifacts.")},
        {QStringLiteral("Overlay 预览"), QStringLiteral("Overlay Preview")},
        {QStringLiteral("完成后显示检测框、分割 mask 或 OCR 可视化图；失败时保留明确状态文本。"), QStringLiteral("After completion, shows detection boxes, segmentation masks, or OCR visualization; clear status text remains on failure.")},
        {QStringLiteral("暂无 overlay\n运行推理后显示可视化产物。"), QStringLiteral("No overlay\nRun inference to show the visualization artifact.")},
        {QStringLiteral("推理验证会根据 ONNX 模型族或 AITrain 导出信息选择 detection、segmentation 或 OCR Rec 后处理；完整 PaddleOCR System 推理通过官方工具链任务产物查看。"), QStringLiteral("Inference validation chooses detection, segmentation, or OCR Rec postprocess from the ONNX model family or AITrain export metadata. Full PaddleOCR System inference is viewed through official toolchain task artifacts.")},
        {QStringLiteral("输入预览"), QStringLiteral("Input Preview")},
        {QStringLiteral("原图、视频帧或 OCR 输入图像。"), QStringLiteral("Original image, video frame, or OCR input image.")},
        {QStringLiteral("结果预览"), QStringLiteral("Result Preview")},
        {QStringLiteral("检测框、分割 mask、OCR 文本和耗时会显示在这里。"), QStringLiteral("Detection boxes, segmentation masks, OCR text, and latency appear here.")},
        {QStringLiteral("暂无推理结果。"), QStringLiteral("No inference result yet.")},
        {QStringLiteral("暂无 overlay"), QStringLiteral("No overlay")},
        {QStringLiteral("扫描模型、数据集、导出和推理扩展能力；插件仍只通过公共接口暴露能力。"), QStringLiteral("Scan model, dataset, export, and inference extension capabilities. Plugins still expose capabilities only through public interfaces.")},
        {QStringLiteral("扫描"), QStringLiteral("Scan")},
        {QStringLiteral("等待插件扫描。"), QStringLiteral("Waiting for plugin scan.")},
        {QStringLiteral("插件搜索路径：未初始化"), QStringLiteral("Plugin search paths: not initialized")},
        {QStringLiteral("manifest 已加载"), QStringLiteral("manifest loaded")},
        {QStringLiteral("可识别 / 校验格式"), QStringLiteral("Recognizable / validatable formats")},
        {QStringLiteral("插件声明的导出目标"), QStringLiteral("Export targets declared by plugins")},
        {QStringLiteral("GPU 需求"), QStringLiteral("GPU Requirement")},
        {QStringLiteral("声明需要 GPU 的插件"), QStringLiteral("Plugins declaring GPU requirement")},
        {QStringLiteral("名称"), QStringLiteral("Name")},
        {QStringLiteral("版本"), QStringLiteral("Version")},
        {QStringLiteral("执行环境自检"), QStringLiteral("Run Environment Check")},
        {QStringLiteral("检查 NVIDIA 驱动、CUDA、TensorRT、ONNX Runtime、Qt 插件和 Worker 可用性。"), QStringLiteral("Check NVIDIA driver, CUDA, TensorRT, ONNX Runtime, Qt plugins, and Worker availability.")},
        {QStringLiteral("尚未执行环境自检。"), QStringLiteral("Environment check has not run.")},
        {QStringLiteral("可用依赖"), QStringLiteral("Available dependencies")},
        {QStringLiteral("可继续但需关注"), QStringLiteral("Can continue, needs attention")},
        {QStringLiteral("会阻塞相关能力"), QStringLiteral("Blocks related capabilities")},
        {QStringLiteral("等待 Worker 自检"), QStringLiteral("Waiting for Worker self-check")},
        {QStringLiteral("检查明细"), QStringLiteral("Check Details")},
        {QStringLiteral("检查项"), QStringLiteral("Check Item")},
        {QStringLiteral("TensorRT 真机验收仍需要 RTX / SM 75+。当前 GTX 1060 / SM 61 只能记录为 hardware-blocked。"), QStringLiteral("Real TensorRT acceptance still requires RTX / SM 75+. The current GTX 1060 / SM 61 can only be recorded as hardware-blocked.")},
        {QStringLiteral("任务已启动：%1"), QStringLiteral("Task started: %1")},
        {QStringLiteral("打开项目后，按 数据集 -> 训练实验 -> 任务与产物 -> 模型导出 -> 推理验证 的顺序完成本机训练闭环。"), QStringLiteral("After opening a project, complete the local training loop in this order: Datasets -> Training Runs -> Tasks and Artifacts -> Model Export -> Inference Check.")},
        {QStringLiteral("这里统一追踪训练、校验、划分、导出和推理任务；运行产物在下方详情区集中查看。"), QStringLiteral("Track training, validation, split, export, and inference tasks here; run artifacts are shown in the details area below.")}
    };

    return translations.value(source, QString());
}

} // namespace

QString languageSettingsKey()
{
    return QStringLiteral("settings/language");
}

QString defaultLanguageCode()
{
    const QString system = QLocale::system().name();
    if (system.startsWith(QStringLiteral("en"))) {
        return QStringLiteral("en_US");
    }
    return QStringLiteral("zh_CN");
}

QString configuredLanguageCode()
{
    QSettings settings;
    return normalizeLanguageCode(settings.value(languageSettingsKey(), defaultLanguageCode()).toString());
}

void storeLanguageCode(const QString& languageCode)
{
    QSettings settings;
    settings.setValue(languageSettingsKey(), normalizeLanguageCode(languageCode));
}

QString languageDisplayName(const QString& languageCode)
{
    const QString normalized = normalizeLanguageCode(languageCode);
    if (normalized == QStringLiteral("en_US")) {
        return QStringLiteral("English");
    }
    return QStringLiteral("中文");
}

bool loadTranslator(QApplication& app, QTranslator* translator, const QString& languageCode)
{
    if (!translator) {
        return false;
    }

    const QString normalized = normalizeLanguageCode(languageCode);
    if (normalized == QStringLiteral("zh_CN")) {
        return false;
    }

    const QString fileName = QStringLiteral("aitrain_%1.qm").arg(normalized);
    for (const QString& root : translationSearchRoots()) {
        if (translator->load(fileName, root)) {
            app.installTranslator(translator);
            return true;
        }
    }
    return false;
}

QString translateText(const char* context, const QString& text)
{
    if (text.isEmpty() || configuredLanguageCode() == QStringLiteral("zh_CN")) {
        return text;
    }
    const QByteArray source = text.toUtf8();
    const QString translated = QCoreApplication::translate(context, source.constData());
    if (translated != text) {
        return translated;
    }

    const QString fallback = fallbackEnglishTranslation(text);
    return fallback.isEmpty() ? text : fallback;
}

void translateWidgetTree(QWidget* root, const char* context)
{
    if (!root || configuredLanguageCode() == QStringLiteral("zh_CN")) {
        return;
    }

    const auto translate = [context](const QString& text) {
        return translateText(context, text);
    };

    const auto widgets = root->findChildren<QWidget*>();
    for (QWidget* widget : widgets) {
        if (auto* label = qobject_cast<QLabel*>(widget)) {
            label->setText(translate(label->text()));
        } else if (auto* button = qobject_cast<QAbstractButton*>(widget)) {
            button->setText(translate(button->text()));
        } else if (auto* groupBox = qobject_cast<QGroupBox*>(widget)) {
            groupBox->setTitle(translate(groupBox->title()));
        } else if (auto* lineEdit = qobject_cast<QLineEdit*>(widget)) {
            lineEdit->setPlaceholderText(translate(lineEdit->placeholderText()));
        } else if (auto* plainTextEdit = qobject_cast<QPlainTextEdit*>(widget)) {
            plainTextEdit->setPlaceholderText(translate(plainTextEdit->placeholderText()));
            if (plainTextEdit->isReadOnly()) {
                plainTextEdit->setPlainText(translate(plainTextEdit->toPlainText()));
            }
        } else if (auto* textEdit = qobject_cast<QTextEdit*>(widget)) {
            textEdit->setPlaceholderText(translate(textEdit->placeholderText()));
            if (textEdit->isReadOnly()) {
                textEdit->setPlainText(translate(textEdit->toPlainText()));
            }
        } else if (auto* comboBox = qobject_cast<QComboBox*>(widget)) {
            for (int index = 0; index < comboBox->count(); ++index) {
                comboBox->setItemText(index, translate(comboBox->itemText(index)));
            }
        }

        if (auto* table = qobject_cast<QTableWidget*>(widget)) {
            for (int column = 0; column < table->columnCount(); ++column) {
                if (auto* item = table->horizontalHeaderItem(column)) {
                    item->setText(translate(item->text()));
                }
            }
            for (int row = 0; row < table->rowCount(); ++row) {
                if (auto* item = table->verticalHeaderItem(row)) {
                    item->setText(translate(item->text()));
                }
                for (int column = 0; column < table->columnCount(); ++column) {
                    if (auto* item = table->item(row, column)) {
                        item->setText(translate(item->text()));
                    }
                }
            }
        }
    }
}

} // namespace aitrain_app
