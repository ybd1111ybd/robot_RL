@echo off
call "%~dp0platform\windows\setup_windows_ros2_fixed.bat" %*
exit /b %ERRORLEVEL%
