; AITrain Studio Python AI environment installer script.
; Build with tools\build-inno-installer.ps1 and pass a prepared PythonEnvSourceDir.

#define AppName "AITrain Studio Python AI Environment"
#define AppPublisher "AITrain"
#ifndef AppVersion
#define AppVersion "0.1.0"
#endif
#ifndef PythonEnvSourceDir
#define PythonEnvSourceDir "..\build-vscode\python_env"
#endif
#ifndef PaddleOcrSourceDir
#define PaddleOcrSourceDir ""
#endif
#ifndef OutputDir
#define OutputDir "..\build-vscode\inno"
#endif
#ifndef PythonEnvExcludes
#define PythonEnvExcludes "__pycache__\*,*.pyc,*.pyo"
#endif
#ifndef PaddleOcrExcludes
#define PaddleOcrExcludes ".git\*,__pycache__\*,*.pyc,*.pyo"
#endif
#ifndef InstallerCompression
#define InstallerCompression "lzma/normal"
#endif
#ifndef InstallerSolidCompression
#define InstallerSolidCompression "no"
#endif
#ifndef OutputBaseFilename
#define OutputBaseFilename "AITrainStudio-" + AppVersion + "-PythonEnv-Setup"
#endif

#if !DirExists(PythonEnvSourceDir)
#error PythonEnvSourceDir does not exist. Pass /DPythonEnvSourceDir=<prepared-python-env-root>.
#endif
#if !FileExists(AddBackslash(PythonEnvSourceDir) + "python.exe") && !FileExists(AddBackslash(PythonEnvSourceDir) + "Scripts\python.exe")
#error PythonEnvSourceDir must contain python.exe or Scripts\python.exe.
#endif

[Setup]
AppId={{CB144F7F-CA08-47D5-A13D-7C95D539CB6F}
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
OutputBaseFilename={#OutputBaseFilename}
Compression={#InstallerCompression}
SolidCompression={#InstallerSolidCompression}
WizardStyle=modern
CloseApplications=yes
CloseApplicationsFilter=AITrainStudio.exe,aitrain_worker.exe
RestartApplications=no
SetupLogging=yes
UsePreviousAppDir=yes
UsePreviousTasks=yes
MinVersion=10.0.17763

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Files]
Source: "{#PythonEnvSourceDir}\*"; DestDir: "{app}\python_env"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "{#PythonEnvExcludes}"
#if Len(PaddleOcrSourceDir) > 0 && DirExists(PaddleOcrSourceDir)
Source: "{#PaddleOcrSourceDir}\*"; DestDir: "{app}\python_env\PaddleOCR"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "{#PaddleOcrExcludes}"
#endif

[Icons]
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"

[UninstallDelete]
Type: dirifempty; Name: "{app}\python_env\PaddleOCR"
Type: dirifempty; Name: "{app}\python_env"
