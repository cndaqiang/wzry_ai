& "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate "$env:USERPROFILE\miniconda3"
conda activate wzry_ai
.\scrcpy-win64-v2.0\adb.exe connect 127.0.0.1:5555
python train.py



# 暂停，方便查看结果
$timer=Start-Job {Start-Sleep 20}; Write-Host "Press any key to continue..."; while(-not [console]::KeyAvailable -and (Get-Job -Id $timer.Id).State -eq 'Running'){Start-Sleep 0.1}; Stop-Job $timer


