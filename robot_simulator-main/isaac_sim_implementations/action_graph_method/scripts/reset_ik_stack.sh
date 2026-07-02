#!/usr/bin/env bash
set -euo pipefail

DOMAIN_ID="${1:-77}"

echo "== Resetting WSL IK / RViz bridge stack (domain ${DOMAIN_ID}) =="

pkill -f 'ros2 launch /mnt/e/jz_robot/jz_descripetion/robot_urdf/launch/isaac_rviz2_bridge.launch.py' 2>/dev/null || true
pkill -f '/mnt/e/jz_robot/install/jz_robot_description/lib/jz_robot_description/joint_state_merger.py' 2>/dev/null || true
pkill -f '/opt/ros/humble/lib/robot_state_publisher/robot_state_publisher' 2>/dev/null || true
pkill -f 'static_transform_publisher.*world.*base_link' 2>/dev/null || true
pkill -f '/opt/ros/humble/lib/rviz2/rviz2' 2>/dev/null || true
pkill -f 'interactive_ik_marker.py' 2>/dev/null || true
pkill -f 'ik_phase4_acceptance_check.py' 2>/dev/null || true
pkill -f 'action_graph_topic_io_check.py' 2>/dev/null || true

echo "== Resetting Windows Isaac processes for domain ${DOMAIN_ID} =="
WIN_DOMAIN_ID="${DOMAIN_ID}" powershell.exe -NoProfile -Command '
  $domain = $env:WIN_DOMAIN_ID
  $pattern = "(^|\s)--domain-id\s+" + [regex]::Escape($domain) + "(\s|$)"
  $targets = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match "jinzhi_ros2_action_graph.py" -and
    $_.CommandLine -match $pattern
  }
  if ($targets) {
    $targets | ForEach-Object {
      Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
    $targets | Select-Object ProcessId, Name, CommandLine | Format-List
  } else {
    Write-Output "No matching Windows Isaac processes found."
  }

  $lockDir = Join-Path $env:LOCALAPPDATA "Temp\jz_robot_isaac_singletons"
  if (Test-Path $lockDir) {
    Get-ChildItem -Path $lockDir -Filter "*_domain_$domain.json" -ErrorAction SilentlyContinue |
      Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Output "Cleared Windows singleton lock files from: $lockDir"
  }
'

sleep 2

if [[ -f /mnt/e/jz_robot/env.sh ]]; then
  # shellcheck source=/dev/null
  source /mnt/e/jz_robot/env.sh >/dev/null 2>&1 || true
fi
export ROS_DOMAIN_ID="${DOMAIN_ID}"

echo "== Remaining ROS nodes on domain ${DOMAIN_ID} =="
if command -v ros2 >/dev/null 2>&1; then
  ros2 node list || true
  echo "---"
  ros2 topic info -v /joint_states || true
else
  echo "ros2 not found in current shell."
fi
