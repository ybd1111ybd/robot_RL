param(
    [string]$RunName = $(Get-Date -Format "yyyy-MM-dd_HH-mm-ss"),
    [string]$Task = "Isaac-Reach-JZ-Bi-v0",
    [int]$NumEnvs = 2048,
    [int]$MaxIterations = 6000,
    [string]$Checkpoint = "",
    [int]$PollSeconds = 1800,
    [int]$EvalStart = 100,
    [int]$EvalEvery = 100,
    [int]$EvalNumEnvs = 8,
    [int]$EvalSteps = 120,
    [int]$EvalTimeoutSeconds = 900,
    [switch]$StartTensorBoard,
    [string]$ProjectRoot = "",
    [string]$IsaacLabRoot = "",
    [string]$CondaEnvName = "env_isaacsim"
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $scriptRoot "..\..\..")).Path
}
if ([string]::IsNullOrWhiteSpace($IsaacLabRoot)) {
    $IsaacLabRoot = $env:ISAACLAB_PATH
}
if ([string]::IsNullOrWhiteSpace($IsaacLabRoot)) {
    throw "Isaac Lab root is not set. Please pass -IsaacLabRoot or set ISAACLAB_PATH."
}

$projectRoot = (Resolve-Path $ProjectRoot).Path
$isaacLabRoot = (Resolve-Path $IsaacLabRoot).Path
$trainScript = Join-Path $projectRoot "scripts\reinforcement_learning\rl_games\train.py"
$watchScript = Join-Path $projectRoot "scripts\reinforcement_learning\rl_games\watch_training.py"

if (-not (Test-Path $trainScript)) {
    throw "Training script not found: $trainScript"
}
if (-not (Test-Path $watchScript)) {
    throw "Watcher script not found: $watchScript"
}

$condaHookCommand = "& conda shell.powershell hook | Out-String | Invoke-Expression"
$logRoot = Join-Path $isaacLabRoot "logs\rl_games\jz_bi_reach"
$runDir = Join-Path $logRoot $RunName

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$trainStdout = Join-Path $logRoot "train_$RunName.log"
$trainStderr = Join-Path $logRoot "train_$RunName.err.log"

$checkpointArg = ""
if ($Checkpoint -ne "") {
    $checkpointArg = "--checkpoint '$Checkpoint'"
}

$trainCommand = @"
$condaHookCommand
conda activate $CondaEnvName
`$env:ISAACLAB_PATH = '$isaacLabRoot'
Set-Location '$isaacLabRoot'
& '.\isaaclab.bat' -p '$trainScript' --task $Task --num_envs $NumEnvs --max_iterations $MaxIterations --headless $checkpointArg +agent.params.config.full_experiment_name=$RunName
"@

$trainProcess = Start-Process -FilePath powershell.exe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $trainCommand) `
    -RedirectStandardOutput $trainStdout `
    -RedirectStandardError $trainStderr `
    -PassThru

Write-Host "Training started."
Write-Host "Run name : $RunName"
Write-Host "Train PID: $($trainProcess.Id)"
Write-Host "Train log: $trainStdout"

$deadline = (Get-Date).AddMinutes(5)
while (-not (Test-Path $runDir)) {
    if ((Get-Date) -gt $deadline) {
        throw "Run directory was not created in time: $runDir"
    }
    Start-Sleep -Seconds 2
}

$watchStdout = Join-Path $runDir "watch_launcher.log"
$watchStderr = Join-Path $runDir "watch_launcher.err.log"
$watchCommand = @"
$condaHookCommand
conda activate $CondaEnvName
`$env:ISAACLAB_PATH = '$isaacLabRoot'
Set-Location '$projectRoot'
& 'python' '$watchScript' --run $RunName --task Isaac-Reach-JZ-Bi-Play-v0 --poll-seconds $PollSeconds --eval-start $EvalStart --eval-every $EvalEvery --eval-num-envs $EvalNumEnvs --eval-steps $EvalSteps --eval-timeout-seconds $EvalTimeoutSeconds
"@

$watchProcess = Start-Process -FilePath powershell.exe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $watchCommand) `
    -RedirectStandardOutput $watchStdout `
    -RedirectStandardError $watchStderr `
    -PassThru

Write-Host "Watcher started."
Write-Host "Watch PID : $($watchProcess.Id)"
Write-Host "Watch log : $watchStdout"

if ($StartTensorBoard) {
    $tbCommand = @"
$condaHookCommand
conda activate $CondaEnvName
Set-Location '$isaacLabRoot'
tensorboard --logdir '$logRoot' --port 6006
"@

    $tbProcess = Start-Process -FilePath powershell.exe `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $tbCommand) `
        -PassThru

    Write-Host "TensorBoard started."
    Write-Host "TensorBoard PID: $($tbProcess.Id)"
    Write-Host "URL: http://localhost:6006/"
}

Write-Host "Run directory: $runDir"
