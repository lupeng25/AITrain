# Example AITrain Plugin Package

Place your built Qt plugin DLL under:

```text
payload/plugins/models/YourPlugin.dll
```

Update `plugin.json` so the package id matches the DLL plugin manifest id, then replace the placeholder SHA256 digest with the real DLL digest.

After replacing `YourPlugin.dll` with a real built plugin, you can generate the zip and refresh hashes with:

```powershell
.\tools\create-plugin-package.ps1 -PackageRoot .\examples\plugin-package-template -OutputPath .\.deps\Example.aitrain-plugin.zip -Force
```
