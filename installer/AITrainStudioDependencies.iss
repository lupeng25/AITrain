; AITrain Studio dependency installer script.
; Build with tools\build-inno-installer.ps1 so SourceDir points at the verified package layout.

#define AppName "AITrain Studio Dependencies"
#define AppPublisher "AITrain"
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
#define PackageExcludes ""
#endif
#ifndef InstallerCompression
#define InstallerCompression "lzma/normal"
#endif
#ifndef InstallerSolidCompression
#define InstallerSolidCompression "no"
#endif
#ifndef OutputBaseFilename
#define OutputBaseFilename "AITrainStudio-" + AppVersion + "-Dependencies-Setup"
#endif

#if !DirExists(SourceDir)
#error SourceDir does not exist. Run tools\package-smoke.ps1 first or pass /DSourceDir=<package root>.
#endif

[Setup]
AppId={{A178B006-9C3A-4F8B-88B4-6C64A3594AC5}
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
Source: "{#SourceDir}\runtimes\*"; DestDir: "{app}\runtimes"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist; Excludes: "{#PackageExcludes}"
Source: "{#SourceDir}\platforms\*"; DestDir: "{app}\platforms"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "{#SourceDir}\imageformats\*"; DestDir: "{app}\imageformats"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "{#SourceDir}\iconengines\*"; DestDir: "{app}\iconengines"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "{#SourceDir}\sqldrivers\*"; DestDir: "{app}\sqldrivers"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "{#SourceDir}\styles\*"; DestDir: "{app}\styles"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "{#SourceDir}\bearer\*"; DestDir: "{app}\bearer"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "{#SourceDir}\Qt5*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\Qt6*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\libEGL*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\libGLES*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\opengl32sw.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\d3dcompiler_47.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\vcruntime*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\msvcp*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\concrt*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\vccorlib*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\api-ms-win-*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\ucrtbase*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\onnxruntime*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourceDir}\ncnn.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"

[UninstallDelete]
Type: dirifempty; Name: "{app}\runtimes\onnxruntime"
Type: dirifempty; Name: "{app}\runtimes\ncnn"
Type: dirifempty; Name: "{app}\runtimes\tensorrt"
Type: dirifempty; Name: "{app}\runtimes"
Type: dirifempty; Name: "{app}\platforms"
Type: dirifempty; Name: "{app}\imageformats"
Type: dirifempty; Name: "{app}\iconengines"
Type: dirifempty; Name: "{app}\sqldrivers"
Type: dirifempty; Name: "{app}\styles"
Type: dirifempty; Name: "{app}\bearer"
