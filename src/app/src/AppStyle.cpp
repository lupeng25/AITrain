#include "AppStyle.h"

#include <QApplication>
#include <QFont>
#include <QPalette>
#include <QSettings>
#include <QRegularExpression>

namespace AppStyle {

QString configuredTheme()
{
    return QSettings().value(QStringLiteral("settings/theme"), QStringLiteral("light")).toString() == QStringLiteral("dark")
        ? QStringLiteral("dark") : QStringLiteral("light");
}

void apply(QApplication& app, const QString& theme)
{
    QFont font(QStringLiteral("Microsoft YaHei UI"));
    font.setPointSize(10);
    app.setFont(font);

    QString sheet = QStringLiteral(R"(
        QMainWindow {
            background: #F4F6F8;
        }

        QWidget {
            color: #111827;
            font-family: "Microsoft YaHei UI";
            font-size: 10pt;
        }

        QScrollBar:vertical {
            background: #EEF2F7;
            border: none;
            border-radius: 5px;
            margin: 0;
            width: 10px;
        }

        QScrollBar::handle:vertical {
            background: #B8C2D0;
            border: 2px solid #EEF2F7;
            border-radius: 5px;
            min-height: 32px;
        }

        QScrollBar::handle:vertical:hover {
            background: #9AA6B2;
        }

        QScrollBar::handle:vertical:pressed {
            background: #6B7280;
        }

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            background: transparent;
            border: none;
            height: 0;
        }

        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {
            background: transparent;
        }

        QScrollBar:horizontal {
            background: #EEF2F7;
            border: none;
            border-radius: 5px;
            height: 10px;
            margin: 0;
        }

        QScrollBar::handle:horizontal {
            background: #B8C2D0;
            border: 2px solid #EEF2F7;
            border-radius: 5px;
            min-width: 32px;
        }

        QScrollBar::handle:horizontal:hover {
            background: #9AA6B2;
        }

        QScrollBar::handle:horizontal:pressed {
            background: #6B7280;
        }

        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            background: transparent;
            border: none;
            width: 0;
        }

        QScrollBar::add-page:horizontal,
        QScrollBar::sub-page:horizontal {
            background: transparent;
        }

        QFrame#TopBar {
            background: #FFFFFF;
            border-bottom: 1px solid #D8DEE6;
        }

        QFrame#Sidebar {
            background: #071B34;
            border: none;
        }

        QLabel#BrandTitle {
            color: #FFFFFF;
            font-size: 12pt;
            font-weight: 700;
        }

        QLabel#BrandSubtitle {
            color: #8FA7C2;
            font-size: 7pt;
        }

        QLabel#SidebarSection {
            color: #7F98B5;
            font-size: 8pt;
            font-weight: 700;
            padding: 6px 4px 2px 4px;
        }

        QPushButton#SidebarButton {
            color: #B8C9DC;
            background: transparent;
            border: none;
            border-radius: 6px;
            padding: 8px 10px;
            text-align: left;
            min-height: 34px;
            icon-size: 18px;
        }

        QPushButton#SidebarButton:hover {
            background: #102B4C;
            color: #FFFFFF;
        }

        QPushButton#SidebarButton:checked {
            background: #15365D;
            color: #FFFFFF;
            border-left: 3px solid #5E92FF;
            padding-left: 7px;
        }

        QLabel#PageTitle {
            color: #111827;
            font-size: 12pt;
            font-weight: 700;
        }

        QLabel#PageCaption {
            color: #6B7280;
            font-size: 8pt;
        }

        QFrame#Panel {
            background: #FFFFFF;
            border: 1px solid #D8DEE6;
            border-radius: 7px;
        }

        QFrame#CompactMetricPanel,
        QFrame[trainingLiveRole="panel"] {
            background: #FFFFFF;
            border: 1px solid #D8DEE6;
            border-radius: 6px;
        }

        QLabel#PanelTitle {
            color: #111827;
            font-size: 9pt;
            font-weight: 700;
        }

        QLabel#MutedText {
            color: #6B7280;
        }

        QLabel#FieldErrorText {
            color: #DC2626;
            font-size: 8pt;
        }

        QLabel#EmptyState {
            color: #6B7280;
            background: #F9FAFB;
            border: 1px dashed #C9D1DB;
            border-radius: 6px;
            padding: 12px;
        }

        QLabel#InlineStatus {
            color: #374151;
            background: #F3F5F8;
            border: 1px solid #D8DEE6;
            border-radius: 5px;
            padding: 5px 9px;
            min-height: 24px;
        }

        QLabel#InlineStatus a {
            color: #2563EB;
        }

        QLabel#TrainingPhaseStatus {
            color: #0F172A;
            background: #EEF6E8;
            border: 1px solid #B9D99A;
            border-left: 3px solid #76B900;
            border-radius: 5px;
            padding: 6px 10px;
            min-height: 26px;
            font-weight: 600;
        }

        QLabel#TaskDetailSummary {
            color: #374151;
            background: #F9FAFB;
            border: 1px solid #D8DEE6;
            border-radius: 5px;
            padding: 8px 10px;
            min-height: 28px;
        }

        QLabel#DarkInlineStatus {
            color: #F9FAFB;
            background: #1F2937;
            border: 1px solid #374151;
            border-left: 3px solid #76B900;
            border-radius: 2px;
            padding: 6px 10px;
            min-height: 26px;
        }

        QLabel#ExperimentKicker {
            color: #76B900;
            font-size: 8pt;
            font-weight: 700;
        }

        QLabel#ExperimentTitle {
            color: #FFFFFF;
            font-size: 13pt;
            font-weight: 700;
        }

        QLabel#ExperimentMeta {
            color: #A7B0BD;
            font-size: 9pt;
        }

        QLabel#MetricValue {
            color: #1F2937;
            font-size: 14pt;
            font-weight: 700;
        }

        QLabel#MetricLabel {
            color: #6B7280;
            font-size: 8pt;
        }

        QLabel#CompactMetricValue,
        QLabel[trainingLiveRole="value"] {
            color: #111827;
            font-size: 11pt;
            font-weight: 700;
            min-height: 22px;
        }

        QLabel#CompactMetricCaption,
        QLabel[trainingLiveRole="caption"] {
            color: #6B7280;
            font-size: 8pt;
            min-height: 18px;
        }

        QLabel#StatusPill {
            border-radius: 10px;
            padding: 3px 8px;
            font-size: 8pt;
            font-weight: 600;
        }

        QFrame#LanguageSwitch {
            background: #F3F5F8;
            border: 1px solid #D8DEE6;
            border-radius: 10px;
        }

        QToolButton#LanguageSwitchButton {
            background: transparent;
            border: none;
            border-radius: 8px;
            color: #6B7280;
            font-size: 8pt;
            font-weight: 700;
            min-width: 30px;
            min-height: 20px;
            padding: 1px 6px;
        }

        QToolButton#LanguageSwitchButton:hover {
            color: #111827;
            background: #E8EDF5;
        }

        QToolButton#LanguageSwitchButton:checked {
            color: #FFFFFF;
            background: #2563EB;
        }

        QPushButton {
            background: #FFFFFF;
            border: 1px solid #C9D1DB;
            border-radius: 5px;
            padding: 5px 12px;
            min-height: 28px;
        }

        QPushButton:hover {
            border-color: #9AA6B2;
            background: #F9FAFB;
        }

        QPushButton:pressed {
            background: #EEF2F7;
        }

        QPushButton:disabled {
            color: #9CA3AF;
            background: #F3F5F8;
            border-color: #D8DEE6;
        }

        QPushButton#PrimaryButton {
            background: #2563EB;
            border-color: #2563EB;
            color: #FFFFFF;
            font-weight: 600;
        }

        QPushButton#PrimaryButton:hover {
            background: #1D4ED8;
            border-color: #1D4ED8;
        }

        QPushButton#PrimaryButton:disabled {
            background: #D8DEE6;
            border-color: #D8DEE6;
            color: #6B7280;
        }

        QPushButton#DangerButton {
            color: #B91C1C;
            border-color: #F1B4B4;
            background: #FFF7F7;
        }

        QPushButton#DangerButton:disabled {
            color: #9CA3AF;
            border-color: #D8DEE6;
            background: #F3F5F8;
        }

        QPushButton#GreenButton {
            background: transparent;
            border: 2px solid #76B900;
            border-radius: 2px;
            color: #FFFFFF;
            font-weight: 700;
            padding: 5px 13px;
            min-height: 28px;
        }

        QPushButton#GreenButton:hover {
            background: #1F2937;
            border-color: #BFF230;
        }

        QPushButton#GreenButton:disabled {
            color: #9CA3AF;
            border-color: #4B5563;
            background: transparent;
        }

        QFrame#ExperimentHeader {
            background: #FFFFFF;
            border: 1px solid #D8DEE6;
            border-radius: 7px;
        }

        QFrame#ExperimentHeader QLabel#ExperimentTitle {
            color: #111827;
        }

        QFrame#ExperimentHeader QLabel#ExperimentKicker {
            color: #2563EB;
        }

        QFrame#ExperimentHeader QLabel#ExperimentMeta {
            color: #6B7280;
        }

        QFrame#ActionStrip {
            background: #F9FAFB;
            border: 1px solid #E5E7EB;
            border-radius: 6px;
        }

        QFrame#ActionStrip QFrame#Panel {
            border-color: #E5E7EB;
        }

        QTabWidget {
            background: transparent;
        }

        QTabWidget::pane {
            border: 1px solid #D8DEE6;
            border-radius: 7px;
            background: #FFFFFF;
            top: 0;
        }

        QTabWidget::tab-bar {
            left: 0;
        }

        QTabWidget QTabBar::tab {
            background: transparent;
            color: #6B7280;
            border: none;
            border-bottom: 2px solid transparent;
            margin-right: 2px;
            min-height: 28px;
            min-width: 74px;
            padding: 6px 12px;
        }

        QTabWidget QTabBar::tab:hover {
            background: #F7F9FC;
            color: #111827;
        }

        QTabWidget QTabBar::tab:selected {
            background: #FFFFFF;
            color: #2563EB;
            border-bottom-color: #2563EB;
            font-weight: 700;
        }

        QTabWidget QTabBar::tab:disabled {
            color: #9CA3AF;
            background: #F3F5F8;
        }

        QFrame#TaskControlStrip {
            background: #F9FAFB;
            border: 1px solid #E5E7EB;
            border-radius: 6px;
        }

        QFrame#TaskControlStrip QPushButton {
            min-height: 30px;
            padding: 6px 12px;
        }

        QLabel#TaskFilterLabel {
            color: #374151;
            font-size: 8pt;
            font-weight: 700;
            padding: 0 4px;
        }

        QLabel#ArtifactPreviewCanvas {
            color: #6B7280;
            background: #F9FAFB;
            border: 1px dashed #C9D1DB;
            border-radius: 8px;
            padding: 16px;
        }

        QPlainTextEdit#ArtifactPreviewText {
            background: #0B1020;
            color: #D1D5DB;
            border: 1px solid #1F2937;
            border-radius: 6px;
            font-family: "Consolas";
            font-size: 9pt;
        }

        QFrame#InferenceHeader {
            background: #FFFFFF;
            border: 1px solid #D8DEE6;
            border-radius: 7px;
        }

        QFrame#InferenceHeader QPushButton#PrimaryButton {
            background: #2563EB;
            border-color: #2563EB;
            color: #FFFFFF;
            font-weight: 600;
        }

        QFrame#InferenceHeader QPushButton#PrimaryButton:hover {
            background: #1D4ED8;
            border-color: #1D4ED8;
        }

        QLabel#InferenceKicker {
            color: #2563EB;
            font-size: 8pt;
            font-weight: 700;
        }

        QLabel#InferenceTitle {
            color: #111827;
            font-size: 13pt;
            font-weight: 700;
        }

        QLabel#InferenceMeta {
            color: #6B7280;
            font-size: 9pt;
        }

        QLabel#InferenceBadge {
            color: #1D4ED8;
            background: #EAF1FF;
            border: 1px solid #CFE0FF;
            border-radius: 11px;
            padding: 3px 9px;
            font-size: 8pt;
            font-weight: 700;
            min-height: 18px;
        }

        QFrame#InferenceStep {
            background: #FFFFFF;
            border: 1px solid #D8DEE6;
            border-radius: 6px;
        }

        QLabel#InferenceStepIndex {
            color: #FFFFFF;
            background: #2563EB;
            border: 1px solid #2563EB;
            border-radius: 12px;
            font-weight: 700;
        }

        QLabel#InferenceStepTitle {
            color: #111827;
            font-weight: 700;
        }

        QLabel#InferenceStepCaption {
            color: #6B7280;
            font-size: 8pt;
        }

        QFrame#InferenceCapability {
            background: #F8FAFC;
            border: 1px solid #D8DEE6;
            border-left: 3px solid #2563EB;
            border-radius: 5px;
        }

        QLabel#InferenceCapabilityTitle {
            color: #111827;
            font-weight: 700;
        }

        QLabel#InferenceCapabilityCaption {
            color: #6B7280;
            font-size: 8pt;
        }

        QLabel#InferenceResultSummary {
            color: #D1D5DB;
            background: #0B1020;
            border: 1px solid #1F2937;
            border-left: 3px solid #76B900;
            border-radius: 4px;
            padding: 10px 12px;
            min-height: 54px;
            font-family: "Consolas";
            font-size: 9pt;
        }

        QLabel#InferenceOverlayCanvas {
            color: #6B7280;
            background: #F9FAFB;
            border: 1px dashed #C9D1DB;
            border-radius: 8px;
            padding: 16px;
        }
    )") + QStringLiteral(R"(

        QFrame#Inspector {
            background: #FFFFFF;
            border-left: 1px solid #D8DEE6;
        }

        QLabel#InspectorTitle {
            color: #111827;
            font-size: 10pt;
            font-weight: 700;
        }

        QLabel#InspectorSubtitle,
        QLabel#InspectorFooter {
            color: #6B7280;
            font-size: 8pt;
        }

        QFrame#InspectorIdentity,
        QFrame#InspectorSection {
            background: #F8FAFC;
            border: 1px solid #E1E7EF;
            border-radius: 6px;
        }

        QLabel#InspectorProject {
            color: #111827;
            font-size: 10pt;
            font-weight: 700;
        }

        QLabel#InspectorDetail {
            color: #5E6C80;
            font-size: 8pt;
        }

        QLabel#InspectorSectionTitle {
            color: #334155;
            font-size: 8pt;
            font-weight: 700;
        }

        QPushButton#InspectorShortcut {
            color: #475569;
            background: transparent;
            border: none;
            border-bottom: 1px solid #E1E7EF;
            border-radius: 0;
            padding: 5px 2px;
            text-align: left;
            min-height: 26px;
        }

        QPushButton#InspectorShortcut:hover {
            color: #2563EB;
            background: transparent;
            border-color: #CFE0FF;
        }

        QGroupBox {
            border: 1px solid #D8DEE6;
            border-radius: 6px;
            margin-top: 12px;
            padding: 10px;
            font-weight: 600;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #374151;
        }

        QLineEdit, QComboBox {
            background: #FFFFFF;
            border: 1px solid #C9D1DB;
            border-radius: 5px;
            padding: 2px 8px;
            min-height: 28px;
            selection-background-color: #2563EB;
        }

        QPlainTextEdit, QTextEdit {
            background: #FFFFFF;
            border: 1px solid #C9D1DB;
            border-radius: 5px;
            padding: 6px 8px;
            selection-background-color: #2563EB;
        }

        QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {
            border-color: #2563EB;
        }

        QComboBox::drop-down {
            border: none;
            width: 24px;
        }

        QCheckBox {
            min-height: 24px;
            spacing: 6px;
        }

        QTableWidget {
            background: #FFFFFF;
            gridline-color: #E5E7EB;
            border: 1px solid #D8DEE6;
            border-radius: 5px;
            alternate-background-color: #F9FAFB;
            selection-background-color: #DBEAFE;
            selection-color: #111827;
        }

        QHeaderView::section {
            background: #F3F5F8;
            color: #374151;
            border: none;
            border-bottom: 1px solid #D8DEE6;
            padding: 7px 8px;
            font-weight: 600;
        }

        QProgressBar {
            background: #EEF2F7;
            border: none;
            border-radius: 5px;
            min-height: 10px;
            text-align: center;
        }

        QProgressBar::chunk {
            background: #2563EB;
            border-radius: 5px;
        }

        QFrame#Sidebar {
            background: #071B34;
        }

        QWidget#SidebarBrand {
            background: transparent;
        }

        QLabel#BrandIcon {
            background: #FFFFFF;
            border-radius: 6px;
        }

        QLabel#SidebarAvatar {
            background: #385574;
            color: #FFFFFF;
            border-radius: 15px;
            font-size: 8pt;
            font-weight: 700;
        }

        QLabel#SidebarUserText {
            color: #D5E1EF;
            font-size: 8pt;
        }

        QFrame#SidebarFooter {
            background: transparent;
            border-top: 1px solid #1B3654;
        }

        QPushButton#SidebarButton[compact="true"] {
            font-size: 0px;
            padding: 8px 0;
            text-align: center;
        }

        QFrame#TopBar {
            background: #FFFFFF;
            border-bottom: 1px solid #DCE3EB;
        }

        QLabel#TopbarCaption {
            color: #7A8798;
            font-size: 8pt;
        }

        QLabel#TopbarProject {
            color: #172033;
            font-size: 9pt;
            font-weight: 700;
        }

        QToolButton#InspectorToggle {
            background: #F4F6F9;
            border: 1px solid #D6DEE8;
            border-radius: 5px;
            min-width: 30px;
            min-height: 28px;
        }

        QToolButton#InspectorToggle:checked {
            color: #2563EB;
            background: #EEF4FF;
            border-color: #AFC6F4;
        }

        QFrame#PageHeading {
            background: #F4F7FA;
            border: none;
        }

        QLabel#PageEyebrow {
            color: #718096;
            font-size: 8pt;
        }

        QLabel#PageTitle {
            color: #152033;
            font-size: 15pt;
            font-weight: 700;
        }

        QFrame#Inspector {
            background: #FFFFFF;
            border-left: 1px solid #DCE3EB;
        }

        QLabel#InspectorTitle {
            color: #172033;
            font-size: 10pt;
            font-weight: 700;
        }

        QFrame#InspectorIdentity,
        QFrame#InspectorSection {
            background: #FFFFFF;
            border: none;
            border-top: 1px solid #E5EAF0;
            border-radius: 0;
        }

        QPushButton#InspectorShortcut {
            background: #FFFFFF;
            border: none;
            border-bottom: 1px solid #E8EDF2;
            border-radius: 0;
            text-align: left;
            padding-left: 4px;
        }

        QFrame#FlowRail {
            background: #FFFFFF;
            border: 1px solid #DCE3EB;
            border-radius: 7px;
            min-height: 60px;
        }

        QFrame#FlowStep,
        QFrame#FlowStepDone,
        QFrame#FlowStepActive {
            background: transparent;
            border: none;
            border-right: 1px solid #E5EAF0;
        }

        QFrame#FlowStepActive {
            background: #F1F6FF;
        }

        QLabel#FlowStepNumber {
            color: #607086;
            background: #F0F3F7;
            border: 1px solid #D6DEE8;
            border-radius: 12px;
            font-size: 8pt;
            font-weight: 700;
        }

        QFrame#FlowStepDone QLabel#FlowStepNumber {
            color: #15803D;
            background: #ECF9F0;
            border-color: #A9DFB8;
        }

        QFrame#FlowStepActive QLabel#FlowStepNumber {
            color: #FFFFFF;
            background: #2563EB;
            border-color: #2563EB;
        }

        QLabel#FlowStepTitle {
            color: #1E293B;
            font-size: 9pt;
            font-weight: 700;
        }

        QLabel#FlowStepDetail {
            color: #7A8798;
            font-size: 7pt;
        }

        QFrame#TrainingRunHeader {
            background: #FFFFFF;
            border: 1px solid #DCE3EB;
            border-radius: 7px;
            min-height: 54px;
        }

        QFrame#WorkspaceToolbar {
            background: #FFFFFF;
            border: 1px solid #DCE3EB;
            border-radius: 6px;
            min-height: 34px;
        }

        QLabel#WorkspaceToolbarContext,
        QLabel#WorkspaceToolbarTitle {
            color: #4B5B70;
            font-size: 8pt;
            font-weight: 600;
        }

        QLabel#WorkspaceToolbarStatus {
            color: #2563EB;
            background: #EEF4FF;
            border-radius: 8px;
            padding: 2px 7px;
            font-size: 7pt;
        }

        QLabel#WorkspaceToolbarMeta {
            color: #758297;
            font-size: 8pt;
        }

        QLabel#RunStatus {
            color: #2563EB;
            background: #EEF4FF;
            border-radius: 8px;
            padding: 2px 7px;
            font-size: 7pt;
        }

        QLabel#TrainingRunTitle {
            color: #182235;
            font-size: 11pt;
            font-weight: 700;
        }

        QLabel#TrainingRunMeta {
            color: #758297;
            font-size: 7pt;
        }

        QLabel#TrainingDatasetNote,
        QLabel#TrainingRunNote {
            color: #526175;
            background: #F7F9FC;
            border: 1px solid #E3E8EF;
            border-radius: 5px;
            padding: 6px 8px;
            font-size: 8pt;
        }

        QPushButton#AdvancedToggle {
            color: #526175;
            background: #F7F9FB;
            border-color: #DCE3EB;
            text-align: left;
        }

        QWidget#ExportAdvancedArgs {
            background: #F7F9FC;
            border: 1px solid #E3E8EF;
            border-radius: 6px;
        }

        QTabWidget#TrainingDetailTabs::pane {
            border: 1px solid #DCE3EB;
            background: #FFFFFF;
        }

        QTextEdit#LogView {
            background: #0B1020;
            color: #D1D5DB;
            border: 1px solid #1F2937;
            font-family: "Consolas";
            font-size: 9pt;
        }
        /* 三工作区界面：中性导航与统一表格，覆盖历史检查器布局样式。 */
        QFrame#Sidebar { background: #EEF1F5; border-right: 1px solid #DCE1E9; }
        QLabel#BrandTitle { color: #172235; font-size: 10pt; font-weight: 500; }
        QLabel#BrandSubtitle, QLabel#SidebarSection { color: #5B6575; }
        QPushButton#SidebarButton, QPushButton#SidebarToolButton {
            color: #273447; background: transparent; border: none;
            font-size: 10pt;
            border-radius: 6px; text-align: left; padding: 8px 10px; min-height: 28px;
        }
        QPushButton#SidebarButton:hover, QPushButton#SidebarToolButton:hover { background: #E2E8F2; color: #172235; }
        QPushButton#SidebarButton:checked { background: #E0EAFE; color: #245EDB; border: none; padding-left: 10px; }
        QPushButton[primaryAction="true"] { background: #245EDB; color: white; border: 1px solid #245EDB; }
        QPushButton[primaryAction="true"]:hover { background: #1D4FC0; }
        QPushButton[primaryAction="true"]:disabled { background: #DBE1EA; color: #758095; border-color: #DBE1EA; }
        QLabel#WorkspaceModeTitle { color: #172235; font-size: 11pt; font-weight: 500; }
        QLabel#TrainingDatasetNote, QLabel#TrainingRunNote { font-size: 9pt; }
        QLabel#PageTitle { font-size: 16pt; font-weight: 500; }
        QTableView { background: #FFFFFF; color: #172235; border: 1px solid #DCE1E9; border-radius: 5px; selection-background-color: #E7EFFF; selection-color: #172235; }
        QTableView::item { padding: 7px 9px; border-bottom: 1px solid #EDF0F5; }
        QHeaderView::section { background: #F7F8FA; color: #5B6575; border: none; border-bottom: 1px solid #DCE1E9; padding: 9px; }
        QLabel#DatasetSampleImage { background: #E9EDF3; border: 1px solid #DCE1E9; border-radius: 6px; }
        QStatusBar { background: #FFFFFF; border-top: 1px solid #DCE1E9; min-height: 30px; }
        QStatusBar::item { border: none; }
        QToolButton#ProjectMenuButton { background: white; border: 1px solid #C9D3E2; border-radius: 5px; padding: 7px 12px; min-height: 22px; }
        QListWidget { background: white; border: 1px solid #DCE1E9; border-radius: 5px; outline: 0; }
        QListWidget::item { padding: 9px 10px; min-height: 20px; color: #273447; }
        QListWidget::item:selected { color: #245EDB; background: #E7EFFF; }
        QPushButton#ActivityTaskButton { border: none; color: #245EDB; background: transparent; padding: 2px 10px; min-height: 24px; }
    )");
    const bool dark = (theme.isEmpty() ? configuredTheme() : theme) == QStringLiteral("dark");
    QPalette palette;
    const QColor window(dark ? QStringLiteral("#171d27") : QStringLiteral("#f4f6f8"));
    const QColor surface(dark ? QStringLiteral("#202836") : QStringLiteral("#ffffff"));
    const QColor text(dark ? QStringLiteral("#e4eaf3") : QStringLiteral("#172235"));
    const QColor muted(dark ? QStringLiteral("#aab7ca") : QStringLiteral("#5b6575"));
    palette.setColor(QPalette::Window, window);
    palette.setColor(QPalette::Base, surface);
    palette.setColor(QPalette::AlternateBase, dark ? QColor(QStringLiteral("#252f3e")) : QColor(QStringLiteral("#f7f8fa")));
    palette.setColor(QPalette::Button, surface);
    palette.setColor(QPalette::WindowText, text);
    palette.setColor(QPalette::Text, text);
    palette.setColor(QPalette::ButtonText, text);
    palette.setColor(QPalette::ToolTipBase, surface);
    palette.setColor(QPalette::ToolTipText, text);
    palette.setColor(QPalette::Highlight, dark ? QColor(QStringLiteral("#304c78")) : QColor(QStringLiteral("#e7efff")));
    palette.setColor(QPalette::HighlightedText, text);
    palette.setColor(QPalette::Mid, dark ? QColor(QStringLiteral("#425066")) : QColor(QStringLiteral("#dce1e9")));
    palette.setColor(QPalette::Disabled, QPalette::Text, muted);
    palette.setColor(QPalette::Disabled, QPalette::WindowText, muted);
    palette.setColor(QPalette::Disabled, QPalette::ButtonText, muted);
    if (dark) {
        // 按声明角色映射现有浅色样式，避免将按钮白字与白色面板混为一类。
        const QRegularExpression declaration(QStringLiteral("([a-z-]+)\\s*:[^;{}]+"));
        const QRegularExpression color(QStringLiteral("#[0-9A-Fa-f]{6}|\\bwhite\\b"));
        auto matches = declaration.globalMatch(sheet);
        QVector<QPair<int, QPair<int, QString>>> replacements;
        while (matches.hasNext()) {
            const auto match = matches.next(); const QString role = match.captured(1);
            QString value = match.captured(); auto colors = color.globalMatch(value);
            QVector<QPair<int, QPair<int, QString>>> mapped;
            while (colors.hasNext()) {
                const auto item = colors.next(); const QColor source(item.captured()); QColor target = source;
                if (role.contains(QStringLiteral("background"))) {
                    if (source.lightnessF() > 0.92) target = surface;
                    else if (source.lightnessF() > 0.72) target = QColor(QStringLiteral("#29384e"));
                    else if (source.saturationF() < 0.35) target = QColor(QStringLiteral("#37465a"));
                } else if (role.startsWith(QStringLiteral("border"))) {
                    if (source.lightnessF() > 0.65 || source.saturationF() < 0.4) target = QColor(QStringLiteral("#425066"));
                } else if (role.endsWith(QStringLiteral("color"))) {
                    if (source.lightnessF() < 0.25) target = text;
                    else if (source.saturationF() < 0.4) target = source.lightnessF() > 0.85 ? text : muted;
                    else target = QColor::fromHslF(source.hslHueF(), 0.8, 0.74);
                }
                mapped.prepend({item.capturedStart(), {item.capturedLength(), target.name()}});
            }
            for (const auto& item : mapped) value.replace(item.first, item.second.first, item.second.second);
            replacements.prepend({match.capturedStart(), {match.capturedLength(), value}});
        }
        for (const auto& item : replacements) sheet.replace(item.first, item.second.first, item.second.second);
    }
    app.setPalette(palette);
    sheet += QStringLiteral(R"(
        QMenu { background: palette(window); color: palette(window-text); }
        QMenu::item:selected { background: palette(highlight); }
        QToolTip { background: palette(tool-tip-base); color: palette(tool-tip-text); border: 1px solid palette(mid); padding: 5px; }
        QLabel#StatusPill[tone="0"] { background: %1; color: %2; }
        QLabel#StatusPill[tone="1"] { background: %3; color: %4; }
        QLabel#StatusPill[tone="2"] { background: %5; color: %6; }
        QLabel#StatusPill[tone="3"] { background: %7; color: %8; }
        QLabel#StatusPill[tone="4"] { background: %9; color: %10; }
    )").arg(dark ? QStringLiteral("#29384e") : QStringLiteral("#eef2f7"), dark ? QStringLiteral("#c0cadd") : QStringLiteral("#4b5563"),
        dark ? QStringLiteral("#183d30") : QStringLiteral("#e8f7ea"), dark ? QStringLiteral("#95e0ad") : QStringLiteral("#166534"),
        dark ? QStringLiteral("#49371c") : QStringLiteral("#fff7e6"), dark ? QStringLiteral("#f7d18c") : QStringLiteral("#9a5b00"),
        dark ? QStringLiteral("#4d262f") : QStringLiteral("#fdecec"), dark ? QStringLiteral("#ffabb3") : QStringLiteral("#b91c1c"),
        dark ? QStringLiteral("#233c62") : QStringLiteral("#eaf1ff")).arg(dark ? QStringLiteral("#a6c7ff") : QStringLiteral("#1d4ed8"));
    app.setStyleSheet(sheet);
}

} // namespace AppStyle
