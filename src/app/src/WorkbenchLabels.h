#pragma once
#include "WorkbenchTranslation.h"
#include <QHash>
#include <QString>
namespace aitrain_app {
inline QString taskDisplayName(const QString& type)
{
    static const QHash<QString, QString> names{
        {QStringLiteral("detection"), aitrain_app::workbenchText(QStringLiteral("目标检测训练"))},
        {QStringLiteral("segmentation"), aitrain_app::workbenchText(QStringLiteral("实例分割训练"))},
        {QStringLiteral("obb_detection"), aitrain_app::workbenchText(QStringLiteral("旋转框检测训练"))},
        {QStringLiteral("semantic_segmentation"), aitrain_app::workbenchText(QStringLiteral("语义分割训练"))},
        {QStringLiteral("anomaly_detection"), aitrain_app::workbenchText(QStringLiteral("异常检测训练"))},
        {QStringLiteral("ocr_detection"), aitrain_app::workbenchText(QStringLiteral("OCR 检测训练"))},
        {QStringLiteral("ocr_recognition"), aitrain_app::workbenchText(QStringLiteral("OCR 识别训练"))},
        {QStringLiteral("dataset_snapshot_import"), aitrain_app::workbenchText(QStringLiteral("数据导入"))},
        {QStringLiteral("data_quality"), aitrain_app::workbenchText(QStringLiteral("质量检查"))},
        {QStringLiteral("dataset_quality"), aitrain_app::workbenchText(QStringLiteral("质量检查"))},
        {QStringLiteral("dataset_conversion"), aitrain_app::workbenchText(QStringLiteral("格式转换"))},
        {QStringLiteral("dataset_split"), aitrain_app::workbenchText(QStringLiteral("数据划分"))},
        {QStringLiteral("annotation_session_create"), aitrain_app::workbenchText(QStringLiteral("创建标注会话"))},
        {QStringLiteral("annotation_session_sync"), aitrain_app::workbenchText(QStringLiteral("同步标注结果"))},
        {QStringLiteral("model_import"), aitrain_app::workbenchText(QStringLiteral("模型导入"))},
        {QStringLiteral("runtime_delivery"), aitrain_app::workbenchText(QStringLiteral("验证与交付"))},
        {QStringLiteral("environment_check"), aitrain_app::workbenchText(QStringLiteral("环境检查"))},
        {QStringLiteral("diagnostics"), aitrain_app::workbenchText(QStringLiteral("诊断包"))},
        {QStringLiteral("ocr_official_report_import"), aitrain_app::workbenchText(QStringLiteral("OCR 报告导入"))},
        {QStringLiteral("ocr_acceptance"), aitrain_app::workbenchText(QStringLiteral("OCR 验收"))},
        {QStringLiteral("external_acceptance_evidence"), aitrain_app::workbenchText(QStringLiteral("外部验收证据"))}};
    return names.value(type, type);
}
inline QString artifactDisplayName(const QString& kind)
{
    static const QHash<QString, QString> names{
        {QStringLiteral("dataset_quality_report"), aitrain_app::workbenchText(QStringLiteral("质量报告"))},
        {QStringLiteral("dataset_quality_analysis"), aitrain_app::workbenchText(QStringLiteral("质量问题分析"))},
        {QStringLiteral("dataset_repair_manifest"), aitrain_app::workbenchText(QStringLiteral("标注修复清单"))},
        {QStringLiteral("annotation_session"), aitrain_app::workbenchText(QStringLiteral("标注会话"))},
        {QStringLiteral("annotation_sync_report"), aitrain_app::workbenchText(QStringLiteral("标注同步报告"))},
        {QStringLiteral("dataset_snapshot"), aitrain_app::workbenchText(QStringLiteral("数据快照"))},
        {QStringLiteral("paddleocr_det_official_report"), aitrain_app::workbenchText(QStringLiteral("OCR 检测官方报告"))},
        {QStringLiteral("paddleocr_rec_official_report"), aitrain_app::workbenchText(QStringLiteral("OCR 识别官方报告"))},
        {QStringLiteral("paddleocr_system_official_report"), aitrain_app::workbenchText(QStringLiteral("OCR 系统官方报告"))},
        {QStringLiteral("runtime_delivery_report"), aitrain_app::workbenchText(QStringLiteral("交付报告"))},
        {QStringLiteral("runtime_deployment_validation"), aitrain_app::workbenchText(QStringLiteral("部署检查结果"))},
        {QStringLiteral("runtime_benchmark"), aitrain_app::workbenchText(QStringLiteral("样本计时结果"))},
        {QStringLiteral("environment_check"), aitrain_app::workbenchText(QStringLiteral("环境检查报告"))},
        {QStringLiteral("diagnostics_bundle"), aitrain_app::workbenchText(QStringLiteral("诊断包"))},
        {QStringLiteral("delivery_report"), aitrain_app::workbenchText(QStringLiteral("交付报告"))}};
    return names.value(kind, kind);
}
}
