$env:APP_ROLE = if ($env:APP_ROLE) { $env:APP_ROLE } else { "simulator" }
$env:CONTROL_INTERVAL_SECONDS = if ($env:CONTROL_INTERVAL_SECONDS) { $env:CONTROL_INTERVAL_SECONDS } else { "2" }
python -m uvicorn src.ppo_k8s_app:app --host 127.0.0.1 --port 8000
