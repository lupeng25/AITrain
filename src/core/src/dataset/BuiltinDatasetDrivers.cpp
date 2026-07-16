#include "aitrain/dataset/BuiltinDatasetDrivers.h"

#include "aitrain/dataset/AnomalyFolderDatasetDriver.h"
#include "aitrain/dataset/PaddleOcrDetDatasetDriver.h"
#include "aitrain/dataset/PaddleOcrRecDatasetDriver.h"
#include "aitrain/dataset/YoloDetectionDatasetDriver.h"
#include "aitrain/dataset/YoloObbDatasetDriver.h"
#include "aitrain/dataset/YoloSegmentationDatasetDriver.h"

namespace aitrain {

bool registerBuiltinDatasetDrivers(DatasetDriverRegistry* registry, QString* error)
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
    static const YoloDetectionDatasetDriver detection;
    static const YoloSegmentationDatasetDriver segmentation;
    static const YoloObbDatasetDriver obb;
    static const SemanticMaskDatasetDriver semanticMask;
    static const AnomalyFolderDatasetDriver anomaly;
    static const PaddleOcrDetDatasetDriver ocrDet;
    static const PaddleOcrRecDatasetDriver ocrRec;
    const DatasetDriver* drivers[] = {&detection, &segmentation, &obb, &semanticMask, &anomaly, &ocrDet, &ocrRec};
    for (const DatasetDriver* driver : drivers) {
        if (!registry->registerDriver(driver, error)) {
            return false;
        }
    }
    return true;
}

} // namespace aitrain
