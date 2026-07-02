# JZ Bimanual USD

This folder stores the generated Isaac Lab USD for the JZ dual-arm robot.

Generate it with:

```powershell
$env:JZLAB_PROJECT_PATH = (Get-Location).Path
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\tools\convert_jz_bimanual.py" --headless
```
