param([ValidateSet('init','encode','sample','decode','all')][string]$Stage = 'all',
      [ValidateSet('vulkan','cpu-reference')][string]$Compute = 'cpu-reference')
$ErrorActionPreference = 'Stop'
$data = Join-Path $PSScriptRoot 'experimental-data'
$configuration = Join-Path $data 'experimental-paths.json'
if (Test-Path -LiteralPath $configuration) {
    $paths = Get-Content -LiteralPath $configuration -Raw | ConvertFrom-Json
    $env:FREEVIDEO_VDN_ROOT = $paths.vdn
    $env:FREEVIDEO_COMFY_ROOT = $paths.comfy
    $env:FREEVIDEO_VULKAN_LIBRARY_DIR = $paths.runtime
} else {
    $parent = Split-Path $PSScriptRoot -Parent
    $env:FREEVIDEO_VDN_ROOT = Join-Path $parent 'vdn-minimax-h3'
    $env:FREEVIDEO_COMFY_ROOT = Join-Path $parent 'ComfyUI'
    $env:FREEVIDEO_VULKAN_LIBRARY_DIR = Join-Path $parent 'runtime'
}
$env:FREEVIDEO_HOME = $data
$output = Join-Path $PSScriptRoot 'experimental-output'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$stages = if ($Stage -eq 'all') { @('encode','sample','decode') } else { @($Stage) }
Push-Location $PSScriptRoot
try {
    foreach ($item in $stages) {
        $name = if ($item -eq 'encode') { 'vulkan' } else { $Compute }
        & '.\.venv\Scripts\python.exe' -u -m scripts.test_vulkan_video $item --compute $Compute 2>&1 |
            Tee-Object -FilePath (Join-Path $output "$name-$item.log")
        if ($LASTEXITCODE -ne 0) { throw "Experimental $name $item failed; see experimental-output/$name-$item.log." }
    }
} finally { Pop-Location }
