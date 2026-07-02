@echo off
setlocal
REM Isaac Sim launcher for ROS2 bridge (Windows).
REM Optional: set DISABLE_FASTDDS_PROFILE=1 to bypass XML profile loading.

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ACTION_GRAPH_DIR=%%~fI"
for %%I in ("%ACTION_GRAPH_DIR%\..\..\..") do set "REPO_ROOT=%%~fI"
set "ISAAC_SIM_PATH=E:\isaac-sim-standalone-5.1.0-windows-x86_64"

if not exist "%ISAAC_SIM_PATH%\python.bat" (
    echo ERROR: Isaac Sim not found: %ISAAC_SIM_PATH%
    exit /b 1
)

echo Configuring ROS2 environment ...
set "PYTHONNOUSERSITE=1"
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

if /I "%DISABLE_FASTDDS_PROFILE%"=="1" (
    set "FASTRTPS_DEFAULT_PROFILES_FILE="
    set "FASTDDS_DEFAULT_PROFILES_FILE="
    echo   FASTDDS profile disabled by DISABLE_FASTDDS_PROFILE=1
) else (
    if defined JZ_FASTDDS_PROFILE (
        set "FASTRTPS_DEFAULT_PROFILES_FILE=%JZ_FASTDDS_PROFILE%"
        echo   Using JZ_FASTDDS_PROFILE override
    ) else (
        if exist "%ACTION_GRAPH_DIR%\fastdds_profile_windows.xml" (
            set "FASTRTPS_DEFAULT_PROFILES_FILE=%ACTION_GRAPH_DIR%\fastdds_profile_windows.xml"
            echo   Using Windows FastDDS profile
        ) else (
            if exist "%ACTION_GRAPH_DIR%\fastdds_profile.xml" (
                set "FASTRTPS_DEFAULT_PROFILES_FILE=%ACTION_GRAPH_DIR%\fastdds_profile.xml"
                echo   Using fallback FastDDS profile
            ) else (
                set "FASTRTPS_DEFAULT_PROFILES_FILE="
                echo   No FastDDS profile found, using default middleware settings
            )
        )
    )
    if defined FASTRTPS_DEFAULT_PROFILES_FILE (
        set "FASTDDS_DEFAULT_PROFILES_FILE=%FASTRTPS_DEFAULT_PROFILES_FILE%"
    ) else (
        set "FASTDDS_DEFAULT_PROFILES_FILE="
    )
)

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
echo.

pushd "%ACTION_GRAPH_DIR%"
if "%~1"=="" (
    echo No script args provided, using default:
    echo   "%ACTION_GRAPH_DIR%\jinzhi_ros2_action_graph.py" --hold --domain-id %ROS_DOMAIN_ID%
    echo Running with Isaac Sim Python: "%ACTION_GRAPH_DIR%\jinzhi_ros2_action_graph.py" --hold --domain-id %ROS_DOMAIN_ID%
    call "%ISAAC_SIM_PATH%\python.bat" "%ACTION_GRAPH_DIR%\jinzhi_ros2_action_graph.py" --hold --domain-id %ROS_DOMAIN_ID%
) else (
    echo Running with Isaac Sim Python: %*
    call "%ISAAC_SIM_PATH%\python.bat" %*
)
set "CMD_EXIT=%ERRORLEVEL%"
popd
exit /b %CMD_EXIT%
