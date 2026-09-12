#include "StatusPill.h"
#include <QStyle>
#include <QVariant>

StatusPill::StatusPill(QWidget* parent)
    : QLabel(parent)
{
    setObjectName(QStringLiteral("StatusPill"));
    setStatus(tr("未检测"), Tone::Neutral);
}

void StatusPill::setStatus(const QString& text, Tone tone)
{
    setText(text);

    setProperty("tone", static_cast<int>(tone));
    style()->unpolish(this);
    style()->polish(this);
    update();
}
