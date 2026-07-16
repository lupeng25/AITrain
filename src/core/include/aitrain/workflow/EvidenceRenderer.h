#pragma once

#include "aitrain/workflow/EvidenceBundle.h"

#include <QByteArray>
#include <QString>

namespace aitrain {

class EvidenceRenderer final {
public:
    static QByteArray renderJson(const EvidenceBundle& bundle, QString* error = nullptr);
    static QString renderMarkdown(const EvidenceBundle& bundle, QString* error = nullptr);
    static QString renderHtml(const EvidenceBundle& bundle, QString* error = nullptr);
    static QByteArray renderModelCard(const EvidenceBundle& bundle, QString* error = nullptr);
};

} // namespace aitrain
