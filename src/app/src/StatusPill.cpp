#include "StatusPill.h"

StatusPill::StatusPill(QWidget* parent)
    : QLabel(parent)
{
    setObjectName(QStringLiteral("StatusPill"));
    setStatus(tr("未检测"), Tone::Neutral);
}

void StatusPill::setStatus(const QString& text, Tone tone)
{
    setText(text);

    QString background;
    QString foreground;
    switch (tone) {
    case Tone::Success:
        background = QStringLiteral("#E8F7EA");
        foreground = QStringLiteral("#166534");
        break;
    case Tone::Warning:
        background = QStringLiteral("#FFF7E6");
        foreground = QStringLiteral("#9A5B00");
        break;
    case Tone::Error:
        background = QStringLiteral("#FDECEC");
        foreground = QStringLiteral("#B91C1C");
        break;
    case Tone::Info:
        background = QStringLiteral("#EAF1FF");
        foreground = QStringLiteral("#1D4ED8");
        break;
    case Tone::Neutral:
    default:
        background = QStringLiteral("#EEF2F7");
        foreground = QStringLiteral("#4B5563");
        break;
    }

    setStyleSheet(QStringLiteral("QLabel#StatusPill { background: %1; color: %2; }").arg(background, foreground));
}
