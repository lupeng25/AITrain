# Copilot Instructions for AITrain Studio

This project uses a harness-first workflow. Before generating substantial code, use these files as the source of truth:

- `HARNESS.md`
- `docs/harness/current-status.md`
- `docs/harness/project-context.md`
- `docs/harness/implementation-checklist.md`
- `docs/harness/quality-gates.md`

For UI work, follow:

- `docs/harness/ui-guidelines.md`

## Project Rules

- C++20, CMake, Qt Widgets.
- Qt 6 preferred, but current verified environment is Qt 5.12.9.
- Long-running tasks belong in `aitrain_worker`, not the GUI thread.
- GUI should orchestrate and display state only.
- SQLite access should go through `ProjectRepository`.
- Model, dataset, validation, export, and inference behavior should be plugin-based.
- Preserve the left-sidebar workbench UI.
- Use UTF-8 and `QStringLiteral` for Chinese UI text.
- Do not claim real LibTorch/CUDA YOLO/OCR training exists while it is still scaffolded.

## Verification

Use:

```powershell
.\tools\harness-check.ps1
```

Expected successful result:

```text
Harness check passed.
```
