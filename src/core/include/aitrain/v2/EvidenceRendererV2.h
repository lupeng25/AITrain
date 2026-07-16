#pragma once

#include "aitrain/v2/EvidenceBundleV2.h"

#include <QByteArray>
#include <QString>

namespace aitrain::v2 {

class EvidenceRendererV2 final {
public:
    static QByteArray renderJson(const EvidenceBundleV2& bundle, QString* error = nullptr);
    static QString renderMarkdown(const EvidenceBundleV2& bundle, QString* error = nullptr);
    static QString renderHtml(const EvidenceBundleV2& bundle, QString* error = nullptr);
    static QByteArray renderModelCard(const EvidenceBundleV2& bundle, QString* error = nullptr);
};

} // namespace aitrain::v2
