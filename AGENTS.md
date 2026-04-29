# AITrain Studio Agent Instructions

This repository uses a harness-first workflow. In every new AI coding conversation, read these files before making a plan or editing code:

1. `HARNESS.md`
2. `docs/harness/project-context.md`
3. `docs/harness/implementation-checklist.md`
4. `docs/harness/quality-gates.md`

If the task touches UI, also read:

5. `docs/harness/ui-guidelines.md`

If the task is broad or implementation-heavy, also inspect:

6. `AITrainStudio_后续实施方案.md`

## Operating Rules

- Do not start implementation from memory. Ground in the harness files first.
- Keep changes scoped to the task.
- Preserve the current Qt Widgets workbench architecture:
  - left sidebar
  - top status bar
  - central `QStackedWidget`
  - `AppStyle`, `Sidebar`, `InfoPanel`, `StatusPill`
- Do not put long-running work in the GUI thread.
- Do not put model training logic in `MainWindow`.
- Use Worker messages for long tasks.
- Use `ProjectRepository` for SQLite metadata.
- Use plugin interfaces for model, dataset, export, inference, and validation extensions.
- Keep Qt 5.12+ compatibility unless explicitly asked to upgrade.
- Use `QStringLiteral` for UI text.
- Avoid Chinese mojibake. Source files must compile with UTF-8.
- If a feature is a scaffold, label it as scaffold. Do not claim real YOLO/OCR training exists until it does.

## Verification

For code changes, run:

```powershell
.\tools\harness-check.ps1
```

For context inspection, run:

```powershell
.\tools\harness-context.ps1
```

Final responses should include:

- What changed.
- Key files changed.
- Verification command and result.
- Any known scaffold or unfinished part.

