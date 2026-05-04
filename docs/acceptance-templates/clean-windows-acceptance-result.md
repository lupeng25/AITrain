# Clean Windows Package Acceptance Result

## Run Identity

- Date/time:
- Tester:
- Organization/team:
- Package name:
- Package hash:
- Source commit:
- Package root:

## Machine

- Windows edition/version:
- CPU:
- RAM:
- GPU(s):
- NVIDIA driver:
- PowerShell version:
- Notes about fresh machine state:

## Command

```powershell
.\tools\acceptance-smoke.ps1 -Package
```

Optional source-side preflight, if run:

```powershell
.\tools\local-rc-closeout.ps1
.\tools\acceptance-smoke.ps1 -LocalBaseline -Package -SkipBuild
```

## Result

- Status: pass / fail
- Start time:
- End time:
- Exit code:
- `acceptance_summary.json` path:
- Summary status:
- Failure reason, if any:

## Evidence Attached

- Full console output:
- `acceptance_summary.json`:
- Worker self-check JSON:
- Worker plugin smoke JSON:
- Package root layout:
- Screenshots, if any:

## Notes And Follow-Ups

- Optional runtimes reported missing:
- Package layout gaps:
- Plugin load issues:
- Documentation/script presence issues:
- Follow-up owner:
- Follow-up due date:
