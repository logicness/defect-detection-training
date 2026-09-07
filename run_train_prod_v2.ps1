# run_train_prod_v2.ps1 - run production v2 training (v11m base, production v1.0 data)
# Usage: powershell -ExecutionPolicy Bypass -File run_train_prod_v2.ps1

$PY = 'E:\Anaconda\envs\yolov11\python.exe'
$SCRIPT = 'D:\RK3588&Orin Nano\ORIN NANO\Model Training\scripts\train\train_production_v2.py'
$LOG = 'D:\RK3588&Orin Nano\ORIN NANO\Model Training\logs\train_prod_v2.log'

for ($b = 1; $b -le 15; $b++) {
    Write-Host "[runner] BLOCK $b / 15 ..."
    & $PY $SCRIPT --block $b *>> $LOG
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[runner] BLOCK $b FAILED (exit $LASTEXITCODE), stopping."
        exit 1
    }
    Write-Host "[runner] BLOCK $b done."
}

Add-Content -Path $LOG -Value "========== TRAIN_PROD_V2 ALL_DONE 2026-09-06 11:12:28 =========="
Write-Host "[runner] ALL DONE."
