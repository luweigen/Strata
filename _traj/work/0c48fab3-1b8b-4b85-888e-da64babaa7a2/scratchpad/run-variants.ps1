# Runs the Coder server once per variant (extra env on top of strata-coder-iq1_m.json + STRATA_PREFILL_TIMING=1),
# three pp4k requests each and one deterministic generation, logs to docs/benchmarks.
Set-Location "E:\work\AI\Strata"
$scratch = "C:\Users\WEILU~1\AppData\Local\Temp\claude\E--work-AI-Strata\0c48fab3-1b8b-4b85-888e-da64babaa7a2\scratchpad"
$env:STRATA_ENGRAM_SRC = "E:\work\AI\EngramHalo.cpp\src"
$variants = @(
    @{ name = "sorted";     env = @{} },
    @{ name = "unsorted";   env = @{ STRATA_PREFILL_SORT_EXPERTS = "0" } },
    @{ name = "mmq0-table"; env = @{ STRATA_PREFILL_MMQ = "0"; STRATA_HIPBLASLT_VERBOSE = "1" } },
    @{ name = "optmax";     env = @{ STRATA_PREFILL_MMQ_OPT = "max" } },
    @{ name = "optmean";    env = @{ STRATA_PREFILL_MMQ_OPT = "mean" } }
)
$only = $args
foreach ($v in $variants) {
    if ($only.Count -gt 0 -and $only -notcontains $v.name) { continue }
    $name = $v.name
    $cfg = Get-Content "strata-coder-iq1_m.json" -Raw | ConvertFrom-Json
    $cfg.env | Add-Member -NotePropertyName STRATA_PREFILL_TIMING -NotePropertyValue "1"
    foreach ($k in $v.env.Keys) { $cfg.env | Add-Member -NotePropertyName $k -NotePropertyValue $v.env[$k] }
    $suffix = if ($env:PP_TOKENS) { "-" + $env:PP_TOKENS } else { "" }
    $log = "E:\work\AI\Strata\docs\benchmarks\2026-10-05-halo-expert-gemm-$name$suffix.log"
    $cfg.log = $log
    $cfgPath = "$scratch\coder-$name.json"
    $cfg | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 $cfgPath
    if (Test-Path $log) { Remove-Item $log -Force -Confirm:$false }
    Get-Process | Where-Object { $_.ProcessName -match '^strata|^python$' } | ForEach-Object { try { Stop-Process -Id $_.Id -Force -Confirm:$false } catch {} }
    Start-Sleep -Seconds 3
    $p = Start-Process -FilePath "C:\conda_envs\strata\python.exe" -ArgumentList @("E:\work\AI\Strata\serve\server.py", "--engine", "strata", "--config", $cfgPath, "--port", "8080") -WorkingDirectory "E:\work\AI\Strata" -RedirectStandardOutput "$scratch\server-$name-stdout.txt" -RedirectStandardError "$scratch\server-$name-stderr.txt" -PassThru
    $deadline = (Get-Date).AddSeconds(180); $up = $false
    while ((Get-Date) -lt $deadline) { try { $r = Invoke-WebRequest -Uri "http://127.0.0.1:8080/v1/models" -UseBasicParsing -TimeoutSec 3; if ($r.StatusCode -eq 200) { $up = $true; break } } catch {}; Start-Sleep -Seconds 3 }
    "== $name : server pid $($p.Id), up $up"
    if (-not $up) { continue }
    $tokens = if ($env:PP_TOKENS) { $env:PP_TOKENS } else { "4096" }
    & "C:\conda_envs\strata\python.exe" "docs\benchmarks\2026-10-05-halo-pp4k.py" "$name-$tokens" 3 $tokens
    # a deterministic generation over a ~1K-token prompt (the prompt path runs the MoE through MMQ): the text is
    # saved for a sorted-vs-unsorted comparison
    $text = (Get-Content "E:\work\AI\EngramHalo.cpp\src\llama-model-loader.cpp" -Raw -Encoding UTF8).Substring(0, 3500)
    $body = @{ model = "x"; temperature = 0; max_tokens = 96; messages = @(@{ role = "user"; content = "Summarize what this code does in three sentences, then list its functions.`n`n" + $text }) } | ConvertTo-Json -Depth 5 -Compress
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/chat/completions" -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 600
    $out = $r.choices[0].message.content
    [System.IO.File]::WriteAllText("$scratch\gen-$name.txt", $out, (New-Object System.Text.UTF8Encoding $false))
    [System.IO.File]::WriteAllText("$scratch\gen-$name.json", ($r | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding $false))
    "generation ($name): $($out.Length) chars, timings: $($r.timings | ConvertTo-Json -Compress)"
    Get-Content $log | Select-String "prefill timing" | ForEach-Object { $_.Line }
    Get-Content $log | Select-String "prefill timing: experts" | ForEach-Object { $_.Line }
    Get-Process | Where-Object { $_.ProcessName -match '^strata|^python$' } | ForEach-Object { try { Stop-Process -Id $_.Id -Force -Confirm:$false } catch {} }
    Start-Sleep -Seconds 3
}
"done"
