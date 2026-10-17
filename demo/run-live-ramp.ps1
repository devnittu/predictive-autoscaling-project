param(
    [string]$BaseUrl = "http://148.100.79.148:8000",
    [int]$TargetRps = 1000,
    [int]$RampDurationSeconds = 180,
    [int]$Concurrency = 100
)

$ErrorActionPreference = "Stop"

& "$PSScriptRoot\..\scripts\run_remote_experiment.ps1" `
    -BaseUrl $BaseUrl `
    -Concurrency $Concurrency `
    -RampToRps $TargetRps `
    -RampDurationSeconds $RampDurationSeconds
