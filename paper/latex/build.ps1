param([Parameter(Mandatory = $true)][string]$PaperRoot)
$ErrorActionPreference = 'Stop'
$paper = (Resolve-Path -LiteralPath $PaperRoot).Path
Push-Location -LiteralPath $paper
try {
    $main = '中文初稿'
    $supp = '中文初稿_补充材料'
    $flags = @('-synctex=1','-interaction=nonstopmode','-halt-on-error','-file-line-error')
    $log = Join-Path $paper 'audit\compile'
    New-Item -ItemType Directory -Path $log -Force | Out-Null
    function Invoke-Checked([string]$Program, [string[]]$Options, [string]$Phase) {
        $path = Join-Path $log "$Phase.txt"
        & $Program @Options *> $path
        if ($LASTEXITCODE -ne 0) { Get-Content -LiteralPath $path -Tail 25; throw "$Phase failed" }
    }
    foreach ($n in 1,2) { Invoke-Checked xelatex ($flags + "$supp.tex") "supplement_$n" }
    Copy-Item -LiteralPath "$supp.aux" -Destination (Join-Path $log 'supplement_refs.aux') -Force
    Invoke-Checked xelatex ($flags + "$main.tex") 'main_1'
    Invoke-Checked bibtex @($main) 'bibtex'
    foreach ($n in 2,3) { Invoke-Checked xelatex ($flags + "$main.tex") "main_$n" }

    foreach ($job in $main,$supp) {
        $bad = Select-String -LiteralPath "$job.log" -Pattern 'undefined references|Citation .* undefined|Reference .* undefined|multiply defined|LaTeX Error|Invalid page number|not found on input'
        if ($bad) { $bad | Write-Output; throw "Unresolved references or resources in $job" }
        foreach ($ext in 'log','out','spl','blg') {
            $path = "$job.$ext"
            if (Test-Path -LiteralPath $path) { Move-Item -LiteralPath $path -Destination (Join-Path $log $path) -Force }
        }
    }
    Write-Output "$main.pdf and $supp.pdf compiled."
} finally { Pop-Location }
