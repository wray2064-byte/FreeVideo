param([string]$Python = 'python', [string]$DependencyRoot = (Split-Path $PSScriptRoot -Parent))
$ErrorActionPreference = 'Stop'
if (-not $IsWindows -and $PSVersionTable.PSEdition -eq 'Core') { throw 'This experimental setup currently targets Windows x64.' }
& $Python -c "import sys; assert sys.version_info[:2] == (3,12), 'Python 3.12 is required'"
if ($LASTEXITCODE -ne 0) { throw 'Select a Python 3.12 executable with -Python.' }
function Check-Native([string]$Action) {
    if ($LASTEXITCODE -ne 0) { throw "$Action failed (exit $LASTEXITCODE)." }
}
function Clone-Pinned([string]$Url, [string]$Path, [string]$Commit) {
    if (-not (Test-Path -LiteralPath $Path)) {
        git -c core.autocrlf=false clone --filter=blob:none --no-checkout $Url $Path
        Check-Native 'Clone'
        git -C $Path -c core.autocrlf=false checkout $Commit
        Check-Native 'Pinned checkout'
    }
    $actual = git -C $Path rev-parse HEAD
    Check-Native 'Dependency inspection'
    if ($actual -ne $Commit) { throw "Existing dependency $Path has a different revision; it was left unchanged." }
}
$DependencyRoot = [IO.Path]::GetFullPath($DependencyRoot)
New-Item -ItemType Directory -Force -Path $DependencyRoot | Out-Null
$pins = Get-Content (Join-Path $PSScriptRoot 'freevideo_engine/dependencies.json') -Raw | ConvertFrom-Json
$vdn = Join-Path $DependencyRoot 'vdn-minimax-h3'
$diffusers = Join-Path $vdn 'diffusers'
$comfy = Join-Path $DependencyRoot 'ComfyUI'
Clone-Pinned $pins.vdn.url $vdn $pins.vdn.commit
Clone-Pinned $pins.diffusers.url $diffusers $pins.diffusers.base
Clone-Pinned $pins.encoder.comfy_url $comfy $pins.encoder.comfy_commit
foreach ($patch in Get-ChildItem (Join-Path $vdn 'diffusers_patches') -Filter '*.patch' | Sort-Object Name) {
    git -C $diffusers apply --check $patch.FullName 2>$null
    if ($LASTEXITCODE -eq 0) {
        git -C $diffusers apply $patch.FullName
        Check-Native 'Official Diffusers patch'
    } else {
        git -C $diffusers apply --reverse --check $patch.FullName 2>$null
        Check-Native 'Previously applied official Diffusers patch'
    }
}
$pythonExe = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    & $Python -m venv (Join-Path $PSScriptRoot '.venv')
    Check-Native 'Virtual environment creation'
}
& $pythonExe -m pip install 'torch==2.14.1+cpu' 'torchaudio==2.11.0+cpu' --index-url https://download.pytorch.org/whl/cpu
Check-Native 'CPU PyTorch installation'
& $pythonExe -m pip install -r (Join-Path $PSScriptRoot 'requirements-vulkan-experimental.txt')
Check-Native 'Experimental dependencies'
& $pythonExe -m pip install -e $diffusers
Check-Native 'Pinned Diffusers installation'
$data = Join-Path $PSScriptRoot 'experimental-data'
$runtime = Join-Path $data 'runtime'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$archive = Join-Path $data 'sd-master-d637cfb-bin-win-vulkan-x64.zip'
$digest = 'B28606D62B8B819BF9D91BD7224A04513AA33EDAC30F9487D5172AF5A59785F2'
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest 'https://github.com/leejet/stable-diffusion.cpp/releases/download/master-959-d637cfb/sd-master-d637cfb-bin-win-vulkan-x64.zip' -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $digest) { throw 'Pinned Vulkan runtime archive failed SHA-256 verification.' }
Expand-Archive -LiteralPath $archive -DestinationPath $runtime -Force
$baseDll = Get-ChildItem -LiteralPath $runtime -Filter 'ggml-base.dll' -Recurse | Select-Object -First 1
if (-not $baseDll) { throw 'The pinned archive did not contain ggml-base.dll.' }
$runtime = $baseDll.DirectoryName
@{vdn=$vdn;comfy=$comfy;runtime=$runtime} | ConvertTo-Json | Set-Content (Join-Path $data 'experimental-paths.json')
$env:FREEVIDEO_HOME = $data
$env:FREEVIDEO_VDN_ROOT = $vdn
$env:FREEVIDEO_COMFY_ROOT = $comfy
$env:FREEVIDEO_VULKAN_LIBRARY_DIR = $runtime
Push-Location $PSScriptRoot
try {
    & $pythonExe -m pytest tests/test_vulkan_experimental.py -q
    Check-Native 'Actual Vulkan numerical tests'
} finally { Pop-Location }
Write-Host 'Experimental setup finished. Model weights are a separate ~46 GiB download; see docs/Vulkan-experimental.md.'
