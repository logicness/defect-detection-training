$python = "E:\Anaconda\envs\yolov11\python.exe"
$script = "D:\RK3568&Orin Nano\ORIN NANO\Model Training\scripts\run_one_block_45.py"
Set-Location "D:\RK3568&Orin Nano\ORIN NANO\Model Training"
$env:PYTHONPATH = "E:\Anaconda\envs\yolov11\Lib\site-packages"
for ($i = 15; $i -le 20; $i++) {
    Write-Output "========== BLOCK $i START $(Get-Date -Format 'HH:mm:ss') =========="
    & $python $script $i 2>&1 | ForEach-Object { Write-Output $_ }
    if ($LASTEXITCODE -ne 0) {
        Write-Output "BLOCK $i FAILED, RETRY"
        & $python $script $i 2>&1 | ForEach-Object { Write-Output $_ }
    }
    Write-Output "========== BLOCK $i DONE $(Get-Date -Format 'HH:mm:ss') =========="
}
Write-Output "========== ALL FINISHED $(Get-Date -Format 'HH:mm:ss') =========="
