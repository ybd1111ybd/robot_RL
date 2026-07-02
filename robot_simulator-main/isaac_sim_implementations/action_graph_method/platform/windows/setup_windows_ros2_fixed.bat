@echo off
REM Configure Windows ROS2/FastDDS environment for Isaac Sim Action Graph.
REM Usage:
REM   1) Configure env in current cmd shell:
REM      call setup_windows_ros2_fixed.bat
REM   2) Run one command with configured env:
REM      setup_windows_ros2_fixed.bat run_with_isaac_fixed.bat jinzhi_ros2_action_graph.py --hold --domain-id 77

setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ACTION_GRAPH_DIR=%%~fI"
for %%I in ("%ACTION_GRAPH_DIR%\..\..\..") do set "REPO_ROOT=%%~fI"

if defined JZ_ROS_DOMAIN_ID (
    set "ROS_DOMAIN_ID=%JZ_ROS_DOMAIN_ID%"
) else (
    set "ROS_DOMAIN_ID=77"
)

if defined JZ_RMW_IMPLEMENTATION (
    set "RMW_IMPLEMENTATION=%JZ_RMW_IMPLEMENTATION%"
) else (
    set "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
)

if defined JZ_ROS_LOCALHOST_ONLY (
    set "ROS_LOCALHOST_ONLY=%JZ_ROS_LOCALHOST_ONLY%"
) else (
    set "ROS_LOCALHOST_ONLY=0"
)

if defined JZ_FASTDDS_PROFILE (
    set "FASTRTPS_DEFAULT_PROFILES_FILE=%JZ_FASTDDS_PROFILE%"
) else (
    if exist "%ACTION_GRAPH_DIR%\fastdds_profile_windows.xml" (
        set "FASTRTPS_DEFAULT_PROFILES_FILE=%ACTION_GRAPH_DIR%\fastdds_profile_windows.xml"
    ) else (
        if exist "%ACTION_GRAPH_DIR%\fastdds_profile.xml" (
            set "FASTRTPS_DEFAULT_PROFILES_FILE=%ACTION_GRAPH_DIR%\fastdds_profile.xml"
        ) else (
            set "FASTRTPS_DEFAULT_PROFILES_FILE="
        )
    )
)
if defined FASTRTPS_DEFAULT_PROFILES_FILE (
    set "FASTDDS_DEFAULT_PROFILES_FILE=%FASTRTPS_DEFAULT_PROFILES_FILE%"
) else (
    set "FASTDDS_DEFAULT_PROFILES_FILE="
)

set "JZ_ROS_PKG=%REPO_ROOT%\jz_descripetion\robot_urdf"
set "JZ_ROS_EXTRA=%JZ_ROS_PKG%"
if exist "%JZ_ROS_PKG%\install\jz_robot_description\share\jz_robot_description" (
    set "JZ_ROS_EXTRA=%JZ_ROS_PKG%\install\jz_robot_description\share;%JZ_ROS_EXTRA%"
)
if exist "%REPO_ROOT%\install\share\jz_robot_description" (
    set "JZ_ROS_EXTRA=%REPO_ROOT%\install\share;%JZ_ROS_EXTRA%"
)
if exist "%REPO_ROOT%\jz_descripetion\jz_robot_description" (
    set "JZ_ROS_EXTRA=%REPO_ROOT%\jz_descripetion;%JZ_ROS_EXTRA%"
)
if defined ROS_PACKAGE_PATH (
    set "ROS_PACKAGE_PATH=%JZ_ROS_EXTRA%;%ROS_PACKAGE_PATH%"
) else (
    set "ROS_PACKAGE_PATH=%JZ_ROS_EXTRA%"
)

echo ==========================================
echo   Windows ROS2 environment configured
echo ==========================================
echo   ROS_DOMAIN_ID=%ROS_DOMAIN_ID%
echo   RMW_IMPLEMENTATION=%RMW_IMPLEMENTATION%
if defined FASTRTPS_DEFAULT_PROFILES_FILE (
    echo   FASTRTPS_DEFAULT_PROFILES_FILE=%FASTRTPS_DEFAULT_PROFILES_FILE%
) else (
    echo   FASTRTPS_DEFAULT_PROFILES_FILE=^(not set^)
)
if defined FASTDDS_DEFAULT_PROFILES_FILE (
    echo   FASTDDS_DEFAULT_PROFILES_FILE=%FASTDDS_DEFAULT_PROFILES_FILE%
)
echo   ROS_PACKAGE_PATH=%ROS_PACKAGE_PATH%
echo   ROS_LOCALHOST_ONLY=%ROS_LOCALHOST_ONLY%
echo ==========================================

if "%~1"=="" goto :set_only

echo Running command with configured environment:
echo   %*
pushd "%ACTION_GRAPH_DIR%"
call %*
set "CMD_EXIT=%ERRORLEVEL%"
popd
endlocal & exit /b %CMD_EXIT%

:set_only
echo To keep these variables in cmd.exe, run:
echo   call setup_windows_ros2_fixed.bat
echo Note: In PowerShell, use `$env:VAR=...` assignments before launching Isaac.

endlocal & (
    set "ROS_DOMAIN_ID=%ROS_DOMAIN_ID%"
    set "RMW_IMPLEMENTATION=%RMW_IMPLEMENTATION%"
    set "FASTRTPS_DEFAULT_PROFILES_FILE=%FASTRTPS_DEFAULT_PROFILES_FILE%"
    set "FASTDDS_DEFAULT_PROFILES_FILE=%FASTDDS_DEFAULT_PROFILES_FILE%"
    set "ROS_PACKAGE_PATH=%ROS_PACKAGE_PATH%"
    set "ROS_LOCALHOST_ONLY=%ROS_LOCALHOST_ONLY%"
)

exit /b 0
