Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$scanRoots = @(
    "src",
    "tests",
    "python_trainers",
    "tools",
    ".vscode",
    "CMakeLists.txt"
)

$files = @()
foreach ($scanRoot in $scanRoots) {
    if (-not (Test-Path -LiteralPath $scanRoot)) {
        continue
    }
    $item = Get-Item -LiteralPath $scanRoot
    if ($item.PSIsContainer) {
        $files += Get-ChildItem -LiteralPath $scanRoot -Recurse -File |
            Where-Object { $_.Extension -in @('.c', '.cc', '.cpp', '.cxx', '.h', '.hpp', '.py', '.ps1', '.cmake', '.md', '.json', '.txt', '.yml', '.yaml', '.vcxproj', '.sln', '.qrc') }
    } else {
        $files += $item
    }
}
$selfPath = [IO.Path]::GetFullPath($PSCommandPath)
$files = @($files | Where-Object { [IO.Path]::GetFullPath($_.FullName) -ne $selfPath })

function Assert-CleanPattern {
    param(
        [string]$Pattern,
        [string]$Message
    )

    $findings = @($files | Select-String -CaseSensitive -Pattern $Pattern)
    if ($findings.Count -gt 0) {
        $findings | Select-Object -First 20 | ForEach-Object { Write-Host $_ -ForegroundColor Red }
        throw $Message
    }
}

Write-Host "Architecture check: removed legacy entry points" -ForegroundColor Cyan
foreach ($pattern in @(
        'aitrain_core',
        'aitrain_foundation',
        'ProjectRepository',
        'DetectionTrainingOptions',
        'DetectionTrainingMetrics',
        'DetectionTrainingResult',
        'resumeCheckpointPath',
        'tiny_linear_detector',
        'python_mock',
        'QPluginLoader',
        '(?<![A-Za-z0-9])startTrain(?![A-Za-z0-9])',
        'runCustomerOcrAcceptance',
        'semantic-onnx-smoke',
        'phase-p1-yolo-full-matrix-smoke')) {
    Assert-CleanPattern $pattern 'Removed legacy target, protocol, or script entry detected.'
}

Write-Host "Architecture check: dependency direction" -ForegroundColor Cyan
$coreCmake = Join-Path $root "src/core/CMakeLists.txt"
if (Test-Path -LiteralPath $coreCmake) {
    $content = Get-Content -LiteralPath $coreCmake -Encoding UTF8 -Raw
    $storageBlock = [regex]::Match($content, 'target_link_libraries\(aitrain_storage(?s:.*?)\)')
    $applicationBlock = [regex]::Match($content, 'target_link_libraries\(aitrain_application(?s:.*?)\)')
    $reverseDependency = $storageBlock.Success -and $storageBlock.Value -match 'aitrain_application'
    if ($reverseDependency) {
        throw 'Reverse application/storage dependency detected.'
    }
}

Write-Host "Architecture check: MainWindow app-shell boundary" -ForegroundColor Cyan
$mainWindowHeader = Join-Path $root "src/app/src/MainWindow.h"
if (Test-Path -LiteralPath $mainWindowHeader) {
    $mainWindowContent = Get-Content -LiteralPath $mainWindowHeader -Encoding UTF8 -Raw
    foreach ($forbiddenMember in @(
            'taskQueueTable_',
            'taskListTableModel_',
            'taskListFilterModel_',
            'taskLoadMoreButton_',
            'taskCancelButton_',
            'taskArtifactPanel_',
            'taskKindFilterCombo_',
            'taskStateFilterCombo_',
            'taskSearchEdit_',
            'taskArtifactPresenter_',
            'dashboardProjectValue_',
            'dashboardTaskValue_',
            'dashboardDatasetValue_',
            'dashboardNextStepLabel_',
            'recentTasksTable_',
            'projectConsoleStatusLabel_',
            'projectPathSummaryLabel_',
            'projectSqliteSummaryLabel_',
            'projectDatasetSummaryLabel_',
            'projectTaskSummaryLabel_',
            'projectExportSummaryLabel_',
            'projectNameEdit_',
            'projectRootEdit_',
            'projectSummaryPresenter_',
            'systemSettingsTabs_',
            'settingsZhLanguageButton_',
            'settingsEnLanguageButton_',
            'settingsDefaultProjectPathStatusLabel_',
            'settingsDefaultProjectPathEdit_',
            'capabilityConsoleStatusLabel_',
            'capabilitySourceLabel_',
            'capabilityCountSummaryLabel_',
            'capabilityDatasetFormatSummaryLabel_',
            'capabilityExportFormatSummaryLabel_',
            'capabilityGpuSummaryLabel_',
            'capabilityTable_',
            'modelRegistryPresenter_',
            'modelRegistrySummaryLabel_',
            'ModelPackageTable_',
            'modelImportSourceEdit_',
            'modelImportManifestEdit_',
            'modelImportResultLabel_',
            'modelWorkspaceTabs_',
            'modelImportInProgress_',
            'environmentCheckPresenter_',
            'environmentConsoleStatusLabel_',
            'environmentOkSummaryLabel_',
            'environmentWarningSummaryLabel_',
            'environmentMissingSummaryLabel_',
            'environmentUncheckedSummaryLabel_',
            'environmentTable_',
            'deploymentTabs_',
            'deploymentModelPackageCombo_',
            'deploymentSampleDatasetIdEdit_',
            'deploymentSampleDatasetVersionIdEdit_',
            'deploymentSampleSnapshotIdEdit_',
            'deploymentSampleSnapshotArtifactIdEdit_',
            'deploymentSampleRelativePathEdit_',
            'deploymentValidationResultLabel_',
            'inferenceModelPackageCombo_',
            'inferenceSampleDatasetIdEdit_',
            'inferenceSampleDatasetVersionIdEdit_',
            'inferenceSampleSnapshotIdEdit_',
            'inferenceSampleSnapshotArtifactIdEdit_',
            'inferenceSampleRelativePathEdit_',
            'inferenceResultLabel_',
            'inferenceOverlayLabel_',
            'capabilityCombo_',
            'taskTypeCombo_',
            'trainingBackendCombo_',
            'modelPresetCombo_',
            'trainingDatasetSummaryLabel_',
            'trainingRunSummaryLabel_',
            'epochsEdit_',
            'batchEdit_',
            'imageSizeEdit_',
            'progressBar_',
            'trainingPhaseLabel_',
            'logEdit_',
            'metricsWidget_',
            'datasetTabs_',
            'datasetListTable_',
            'datasetPathEdit_',
            'datasetFormatCombo_',
            'dataQualityDatasetIdEdit_',
            'dataQualityDatasetVersionIdEdit_',
            'dataQualitySnapshotIdEdit_',
            'dataQualitySnapshotArtifactIdEdit_',
            'splitSourceDatasetIdEdit_',
            'splitSourceSnapshotIdEdit_',
            'datasetConversionSourceFormatCombo_',
            'datasetConversionTargetFormatCombo_',
            'datasetConversionInputEdit_',
            'datasetConversionProgressBar_',
            'validationIssuesTable_',
            'datasetRepairLoopTable_',
            'sampleReviewTable_',
            'sampleReviewSummaryLabel_',
            'customerOcrDetReportEdit_',
            'customerOcrRecReportEdit_',
            'customerOcrSystemReportEdit_',
            'customerOcrCohortIdEdit_',
            'customerOcrDomainIdEdit_',
            'customerOcrStatusLabel_',
            'diagnosticsStatusLabel_',
            'deliveryAcceptanceSummaryLabel_',
            'deliveryAcceptanceTable_',
            'deliveryEvidencePresenter_')) {
        if ($mainWindowContent.Contains($forbiddenMember)) {
            throw "Page business state leaked back into MainWindow: $forbiddenMember"
        }
    }
}

$activeAppSources = @(Get-ChildItem -LiteralPath (Join-Path $root "src/app/src") -Recurse -File |
    Where-Object { $_.Extension -in @('.cpp', '.h', '.hpp') })
$combinedProjectEntry = @($activeAppSources |
    Select-String -CaseSensitive -SimpleMatch -Pattern '创建 / 打开项目')
if ($combinedProjectEntry.Count -gt 0) {
    $combinedProjectEntry | Select-Object -First 20 |
        ForEach-Object { Write-Host $_ -ForegroundColor Red }
    throw 'Project create/open/rebuild must remain explicit operations.'
}

Write-Host "Architecture check: Storage repository boundaries" -ForegroundColor Cyan
foreach ($repository in @(
        'ProjectMetaRepository',
        'TaskEventRepository',
        'WorkflowRepository',
        'WorkflowTerminalizationStore',
        'ArtifactCatalogRepository',
        'DatasetCatalogRepository',
        'ModelCatalogRepository',
        'ProjectReadRepository')) {
    $header = Join-Path $root "src/core/include/aitrain/storage/$repository.h"
    $source = Join-Path $root "src/core/src/storage/$repository.cpp"
    $headerExists = Test-Path -LiteralPath $header
    $sourceExists = Test-Path -LiteralPath $source
    if (-not $headerExists -or -not $sourceExists) {
        throw "Required Storage repository is missing: $repository"
    }
}

Write-Host "Architecture check: Artifact path boundary" -ForegroundColor Cyan
$activeSourceFiles = @(Get-ChildItem -LiteralPath (Join-Path $root "src") -Recurse -File |
    Where-Object { $_.Extension -in @('.cpp', '.h', '.hpp') })
$committedPathLeaks = @($activeSourceFiles |
    Select-String -CaseSensitive -SimpleMatch -Pattern 'artifacts/committed')
if ($committedPathLeaks.Count -gt 0) {
    $committedPathLeaks | Select-Object -First 20 |
        ForEach-Object { Write-Host $_ -ForegroundColor Red }
    throw 'Upper modules must not reconstruct the committed Artifact path.'
}
$artifactPathCallLeaks = @($activeSourceFiles |
    Where-Object {
        $_.FullName -notlike '*\src\core\src\artifact\ArtifactStore.cpp' -and
        $_.FullName -notlike '*\src\core\include\aitrain\artifact\ArtifactStore.h'
    } |
    Select-String -CaseSensitive -Pattern 'artifactPath\s*\(')
if ($artifactPathCallLeaks.Count -gt 0) {
    $artifactPathCallLeaks | Select-Object -First 20 |
        ForEach-Object { Write-Host $_ -ForegroundColor Red }
    throw 'Upper modules must obtain committed Artifact access through openVerified().'
}

Write-Host "Architecture check passed." -ForegroundColor Green
