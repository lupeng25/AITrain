#include "aitrain/v2/BuiltinDatasetDriversV2.h"

#include "aitrain/v2/AnomalyFolderDatasetDriverV2.h"
#include "aitrain/v2/PaddleOcrDetDatasetDriverV2.h"
#include "aitrain/v2/PaddleOcrRecDatasetDriverV2.h"
#include "aitrain/v2/YoloDetectionDatasetDriverV2.h"
#include "aitrain/v2/YoloObbDatasetDriverV2.h"
#include "aitrain/v2/YoloSegmentationDatasetDriverV2.h"

namespace aitrain::v2 {

bool registerBuiltinDatasetDriversV2(DatasetDriverRegistryV2* registry, QString* error)
{
    if (!registry) {
        if (error) {
            *error = QStringLiteral("dataset_driver_registry_missing");
        }
        return false;
    }
    if (!registry->formats().isEmpty()) {
        if (error) {
            *error = QStringLiteral("dataset_driver_registry_not_empty");
        }
        return false;
    }
    static const YoloDetectionDatasetDriverV2 detection;
    static const YoloSegmentationDatasetDriverV2 segmentation;
    static const YoloObbDatasetDriverV2 obb;
    static const SemanticMaskDatasetDriverV2 semanticMask;
    static const AnomalyFolderDatasetDriverV2 anomaly;
    static const PaddleOcrDetDatasetDriverV2 ocrDet;
    static const PaddleOcrRecDatasetDriverV2 ocrRec;
    const DatasetDriverV2* drivers[] = {&detection, &segmentation, &obb, &semanticMask, &anomaly, &ocrDet, &ocrRec};
    for (const DatasetDriverV2* driver : drivers) {
        if (!registry->registerDriver(driver, error)) {
            return false;
        }
    }
    return true;
}

} // namespace aitrain::v2
