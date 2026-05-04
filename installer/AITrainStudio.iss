; AITrain Studio Inno Setup installer script.
; Build with tools\build-inno-installer.ps1 so SourceDir points at the verified package layout.

#define AppName "AITrain Studio"
#define AppPublisher "AITrain"
#define AppExeName "AITrainStudio.exe"
#define WorkerExeName "aitrain_worker.exe"
#ifndef AppVersion
#define AppVersion "0.1.0"
#endif
#ifndef SourceDir
#define SourceDir "..\build-vscode\package-smoke"
#endif
#ifndef OutputDir
#define OutputDir "..\build-vscode\inno"
#endif
#ifndef PackageExcludes
#define PackageExcludes "*.pdb,*.ilk,*.exp,*.lib,installer\*,tools\build-inno-installer.ps1"
#endif
#ifndef InstallerCompression
#define InstallerCompression "lzma/normal"
#endif
#ifndef InstallerSolidCompression
#define InstallerSolidCompression "no"
#endif

#if !DirExists(SourceDir)
#error SourceDir does not exist. Run tools\package-smoke.ps1 first or pass /DSourceDir=<package root>.
#endif
#if !FileExists(AddBackslash(SourceDir) + AppExeName)
#error SourceDir is missing AITrainStudio.exe.
#endif
#if !FileExists(AddBackslash(SourceDir) + WorkerExeName)
#error SourceDir is missing aitrain_worker.exe.
#endif

[Setup]
AppId={{7E9CFBD7-7CC5-4C1D-A842-18B7F3E48C9D}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\AITrain Studio
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
OutputDir={#OutputDir}
OutputBaseFilename=AITrainStudio-{#AppVersion}-Setup
Compression={#InstallerCompression}
SolidCompression={#InstallerSolidCompression}
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeName}
CloseApplications=yes
CloseApplicationsFilter={#AppExeName},{#WorkerExeName}
RestartApplications=no
SetupLogging=yes
UsePreviousAppDir=yes
UsePreviousTasks=yes
MinVersion=10.0.17763

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "{#PackageExcludes}"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{group}\Acceptance Runbook"; Filename: "{app}\docs\acceptance-runbook.md"; WorkingDir: "{app}\docs"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: dirifempty; Name: "{app}\runtimes\onnxruntime"
Type: dirifempty; Name: "{app}\runtimes\tensorrt"
Type: dirifempty; Name: "{app}\runtimes"
Type: dirifempty; Name: "{app}\plugins\models"
Type: dirifempty; Name: "{app}\plugins"
