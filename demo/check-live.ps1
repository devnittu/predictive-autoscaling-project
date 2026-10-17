param(
    [string]$BaseUrl = "http://148.100.79.148:8000"
)

$ErrorActionPreference = "Stop"
$base = $BaseUrl.TrimEnd("/")

Write-Host "Checking $base"
Write-Host "`nHealth:"
Invoke-RestMethod "$base/health" | ConvertTo-Json
Write-Host "`nReadiness:"
Invoke-RestMethod "$base/ready" | ConvertTo-Json
Write-Host "`nController state:"
Invoke-RestMethod "$base/api/state" | ConvertTo-Json
Write-Host "`nDashboard:"
Write-Host "$base/"
Write-Host "$base/details"
