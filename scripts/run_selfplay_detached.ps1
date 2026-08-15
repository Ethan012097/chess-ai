<#
.SYNOPSIS
    用「工作排程器」啟動自我對弈，讓它完全脫離終端機的生命週期。

.DESCRIPTION
    為什麼不直接 Start-Process：
    自我對弈一代要一個多小時，五代是掛著跑一整晚的事情。用 Start-Process
    啟動的行程雖然不是「發起它的那個 shell」的子行程，但它仍然在**啟動它的
    那個 session 的行程樹**底下。終端機關掉、遠端 session 結束、或是上層用
    Windows Job Object 管理行程樹的工具收工時，都可能把它一起帶走 ——
    而自我對弈被砍掉的代價是好幾個小時的 GPU 時間（見 CLAUDE.md §5-8）。

    工作排程器啟動的任務是由 Task Scheduler 服務拉起來的，不掛在任何終端機
    底下。關掉 VSCode、關掉 PowerShell 都動不到它。

    **限制：使用者登出時任務會停。** 這裡刻意用 Interactive 登入類型，因為
    CUDA 在 WDDM 驅動下需要互動式 session 才拿得到顯示卡。鎖定螢幕沒問題，
    登出不行。跑之前也記得擋掉睡眠（見下方 -BlockSleep）。

.EXAMPLE
    # 接續跑 3 代（代數會自己從 replay buffer 的檔案數接下去）
    powershell -ExecutionPolicy Bypass -File scripts\run_selfplay_detached.ps1 -Iterations 3

.EXAMPLE
    # 每代 500 局、放寬 SPRT 的對立假設
    powershell -ExecutionPolicy Bypass -File scripts\run_selfplay_detached.ps1 `
        -Iterations 5 -Games 500 -TrainSteps 800 -ExtraArgs "--gate-rounds 200"

.EXAMPLE
    # 看狀態 / 停掉
    Get-ScheduledTaskInfo -TaskName chessai_selfplay
    Stop-ScheduledTask   -TaskName chessai_selfplay
#>

[CmdletBinding()]
param(
    [int]    $Iterations = 3,
    [int]    $Games      = 200,
    [int]    $Simulations = 400,
    [int]    $TrainSteps = 1500,
    [int]    $BatchSize  = 256,
    [double] $Lr         = 1e-4,
    # 原封不動接到 src.selfplay 後面的其他參數，例如 "--gate-rounds 200 --from-best"
    [string] $ExtraArgs  = "",
    [string] $TaskName   = "chessai_selfplay",
    # 順便把 AC 電源的睡眠關掉。跑完要自己改回來：powercfg /change standby-timeout-ac 10
    [switch] $BlockSleep
)

$ErrorActionPreference = "Stop"

$root   = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "找不到 $python`n下一步：先建好 venv（python -m venv .venv）並裝好套件"
}

$stamp   = Get-Date -Format "yyyyMMdd_HHmm"
$logDir  = Join-Path $root "logs"
$outLog  = Join-Path $logDir "selfplay_$stamp.log"
$errLog  = Join-Path $logDir "selfplay_$stamp.log.err"   # tqdm 走 stderr，所以進度條在這裡
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

# 已經有一份在跑就不要再開一份：兩個行程會寫到同一個 iter_NNNN.npy 互相蓋掉
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq "Running") {
    throw "任務 $TaskName 正在執行中。`n下一步：Stop-ScheduledTask -TaskName $TaskName，或換一個 -TaskName"
}

# 命令列寫進一個 .cmd 再讓任務去跑它。
# 理由：Register-ScheduledTask 的 -Argument 是單一字串，裡面同時要有路徑引號與
# 重新導向符號時，跳脫規則非常容易出錯；寫成批次檔就完全不用管跳脫。
$cmdFile = Join-Path $logDir "selfplay_task.cmd"
$argLine = "-m src.selfplay --iterations $Iterations --games $Games " +
           "--simulations $Simulations --train-steps $TrainSteps " +
           "--batch-size $BatchSize --lr $Lr $ExtraArgs"
@"
@echo off
cd /d "$root"
rem 不緩衝，否則 log 要等緩衝區滿了才會出現，看起來像卡住
set PYTHONUNBUFFERED=1
"$python" $argLine 1> "$outLog" 2> "$errLog"
"@ | Set-Content -Path $cmdFile -Encoding ASCII

# 再包一層 VBS 的理由：**視窗一定要藏起來。**
# 工作排程器用 Interactive 身分直接跑 cmd.exe 會開一個可見的主控台視窗。
# 那個視窗會一直杵在桌面上，而使用者只要順手把它關掉，Windows 就會送出
# CTRL_CLOSE_EVENT，整個自我對弈直接死掉（結束碼 0xC000013A = STATUS_CONTROL_C_EXIT）。
# 實測就是這樣掉了兩次。WScript.Shell.Run 的第二個參數 0 = 隱藏視窗，
# 第三個參數 True = 等它結束（這樣任務的 State 才會如實反映跑完沒有）。
$vbsFile = Join-Path $logDir "selfplay_task.vbs"
@"
Set sh = CreateObject("WScript.Shell")
sh.Run """$cmdFile""", 0, True
"@ | Set-Content -Path $vbsFile -Encoding ASCII

$action = New-ScheduledTaskAction -Execute "wscript.exe" `
                                  -Argument "//nologo `"$vbsFile`"" `
                                  -WorkingDirectory $root

# Interactive：跑在使用者的 session 裡，CUDA 才拿得到顯示卡
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero)    # 0 = 不限時。預設 3 天，五代跑得完但不要賭

Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal `
                       -Settings $settings -Force | Out-Null

if ($BlockSleep) {
    powercfg /change standby-timeout-ac 0
    powercfg /change hibernate-timeout-ac 0
    Write-Host "已擋掉 AC 電源的睡眠。跑完請自己改回來：powercfg /change standby-timeout-ac 10"
}

Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 5
$info = Get-ScheduledTaskInfo -TaskName $TaskName

Write-Host ""
Write-Host "已啟動 $TaskName（$Iterations 代 × $Games 局，每步 $Simulations 次模擬）"
Write-Host "  狀態    : $((Get-ScheduledTask -TaskName $TaskName).State)  LastResult=0x$('{0:X}' -f $info.LastTaskResult)"
Write-Host "  stdout  : $outLog"
Write-Host "  進度條  : $errLog"
Write-Host ""
Write-Host "監看："
Write-Host "  Get-Content `"$outLog`" -Tail 20"
Write-Host "  dir `"$root\data\selfplay`"    # 每 25 局更新一次，超過 20 分鐘沒動才是真的停了"
Write-Host "停止："
Write-Host "  Stop-ScheduledTask -TaskName $TaskName"
