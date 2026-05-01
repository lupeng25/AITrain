#include "AppStyle.h"

#include <QApplication>
#include <QFont>

namespace AppStyle {

void apply(QApplication& app)
{
    QFont font(QStringLiteral("Microsoft YaHei UI"));
    font.setPointSize(9);
    app.setFont(font);

    app.setStyleSheet(QStringLiteral(R"(
        QMainWindow {
            background: #F4F6F8;
        }

        QWidget {
            color: #111827;
            font-family: "Microsoft YaHei UI";
            font-size: 9pt;
        }

        QFrame#TopBar {
            background: #FFFFFF;
            border-bottom: 1px solid #D8DEE6;
        }

        QFrame#Sidebar {
            background: #111827;
            border: none;
        }

        QLabel#BrandTitle {
            color: #FFFFFF;
            font-size: 14pt;
            font-weight: 700;
        }

        QLabel#BrandSubtitle {
            color: #9CA3AF;
            font-size: 8pt;
        }

        QLabel#SidebarSection {
            color: #6B7280;
            font-size: 8pt;
            font-weight: 700;
            padding: 6px 2px 2px 2px;
        }

        QPushButton#SidebarButton {
            color: #D1D5DB;
            background: transparent;
            border: none;
            border-radius: 5px;
            padding: 8px 12px;
            text-align: left;
            min-height: 30px;
        }

        QPushButton#SidebarButton:hover {
            background: #1F2937;
            color: #FFFFFF;
        }

        QPushButton#SidebarButton:checked {
            background: #263244;
            color: #FFFFFF;
            border-left: 3px solid #76B900;
            padding-left: 9px;
        }

        QLabel#PageTitle {
            color: #111827;
            font-size: 15pt;
            font-weight: 700;
        }

        QLabel#PageCaption {
            color: #6B7280;
            font-size: 9pt;
        }

        QFrame#Panel {
            background: #FFFFFF;
            border: 1px solid #D8DEE6;
            border-radius: 6px;
        }

        QFrame#CompactMetricPanel {
            background: #FFFFFF;
            border: 1px solid #D8DEE6;
            border-radius: 6px;
        }

        QLabel#PanelTitle {
            color: #111827;
            font-size: 10pt;
            font-weight: 700;
        }

        QLabel#MutedText {
            color: #6B7280;
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
            color: #111827;
            font-size: 15pt;
            font-weight: 700;
        }

        QLabel#MetricLabel {
            color: #6B7280;
            font-size: 8pt;
        }

        QLabel#CompactMetricValue {
            color: #111827;
            font-size: 11pt;
            font-weight: 700;
            min-height: 22px;
        }

        QLabel#CompactMetricCaption {
            color: #6B7280;
            font-size: 8pt;
            min-height: 18px;
        }

        QLabel#StatusPill {
            border-radius: 10px;
            padding: 3px 9px;
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
            background: #111827;
        }

        QPushButton {
            background: #FFFFFF;
            border: 1px solid #C9D1DB;
            border-radius: 5px;
            padding: 5px 12px;
            min-height: 26px;
        }

        QPushButton:hover {
            border-color: #9AA6B2;
            background: #F9FAFB;
        }

        QPushButton:pressed {
            background: #EEF2F7;
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

        QPushButton#DangerButton {
            color: #B91C1C;
            border-color: #F1B4B4;
            background: #FFF7F7;
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

        QFrame#ExperimentHeader {
            background: #0B1020;
            border: 1px solid #1F2937;
            border-radius: 3px;
        }

        QFrame#ActionStrip {
            background: #F9FAFB;
            border: 1px solid #E5E7EB;
            border-radius: 6px;
        }

        QFrame#ActionStrip QFrame#Panel {
            border-color: #E5E7EB;
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
            background: #76B900;
            border-radius: 5px;
        }

        QTextEdit#LogView {
            background: #0B1020;
            color: #D1D5DB;
            border: 1px solid #1F2937;
            font-family: "Consolas";
            font-size: 9pt;
        }
    )"));
}

} // namespace AppStyle
