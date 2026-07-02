@echo off
call "%~dp0platform\windows\run_with_isaac_fixed.bat" %*
exit /b %ERRORLEVEL%
