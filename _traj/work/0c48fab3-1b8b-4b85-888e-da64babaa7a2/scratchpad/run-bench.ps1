# The full 3060M-style benchmark on the engine with TODO 2's changes, for the Coder and UD-IQ4_XS configs:
# server start timed to the first /v1/models answer ("model load to listening"), then bench_halo.py.
Set-Location "E:\work\AI\Strata"
$scratch = "C:\Users\WEILU~1\AppData\Local\Temp\claude\E--work-AI-Strata\0c48fab3-1b8b-4b85-888e-da64babaa7a2\scratchpad"
$env:STRATA_ENGRAM_SRC = "E:\work\AI\EngramHalo.cpp\src"
$runs = @(
    @{ cfg = "strata-coder-iq1_m.json";       name = "coder-iq1_m-todo2";  label = "halo-8060S-coder-iq1_m-arena-todo2" },
    @{ cfg = "strata-unsloth-ud-iq4_xs.json"; name = "ud-iq4_xs-todo2";    label = "halo-8060S-ud-iq4_xs-32-96-cache10000-todo2" }
)
$only = $args
foreach ($r in $runs) {
    if ($only.Count -gt 0 -and $only -notcontains $r.name) { continue }
    $cfg = Get-Content $r.cfg -Raw | ConvertFrom-Json
    if (-not ($cfg.env.PSObject.Properties.Name -contains "STRATA_PREFILL_TIMING")) { $cfg.env | Add-Member -NotePropertyName STRATA_PREFILL_TIMING -NotePropertyValue "1" }
    $log = "E:\work\AI\Strata\docs\benchmarks\2026-10-05-halo-$($r.name).log"
    $cfg.log = $log
    $cfgPath = "$scratch\bench-$($r.name).json"
    $cfg | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 $cfgPath
    if (Test-Path $log) { Remove-Item $log -Force -Confirm:$false }
    Get-Process | Where-Object { $_.ProcessName -match '^strata|^python$' } | ForEach-Object { try { Stop-Process -Id $_.Id -Force -Confirm:$false } catch {} }
    Start-Sleep -Seconds 5
    $t0 = Get-Date
    $p = Start-Process -FilePath "C:\conda_envs\strata\python.exe" -ArgumentList @("E:\work\AI\Strata\serve\server.py", "--engine", "strata", "--config", $cfgPath, "--port", "8080") -WorkingDirectory "E:\work\AI\Strata" -RedirectStandardOutput "$scratch\bench-$($r.name)-stdout.txt" -RedirectStandardError "$scratch\bench-$($r.name)-stderr.txt" -PassThru
    $deadline = (Get-Date).AddSeconds(600); $up = $false
    while ((Get-Date) -lt $deadline) { try { $x = Invoke-WebRequest -Uri "http://127.0.0.1:8080/v1/models" -UseBasicParsing -TimeoutSec 2; if ($x.StatusCode -eq 200) { $up = $true; break } } catch {}; Start-Sleep -Milliseconds 500 }
    $load = ((Get-Date) - $t0).TotalSeconds
    "== $($r.name): up $up after {0:N1} s (pid $($p.Id))" -f $load
    if (-not $up) { continue }
    & "C:\conda_envs\strata\python.exe" "docs\benchmarks\2026-10-02-3060m-bench_halo.py" $r.label "docs\benchmarks\2026-10-05-halo-$($r.name).json"
    Get-Content $log | Select-String "expert cache|hit rate|VRAM free|prefill timing: 4|prefill timing: 16|prefill timing: 20" | ForEach-Object { $_.Line.Substring(0, [Math]::Min(200, $_.Line.Length)) }
    Get-Process | Where-Object { $_.ProcessName -match '^strata|^python$' } | ForEach-Object { try { Stop-Process -Id $_.Id -Force -Confirm:$false } catch {} }
    Start-Sleep -Seconds 5
}
"done"
