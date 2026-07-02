#!/bin/bash
# WSL ROS2 environment setup for cross-host (WSL <-> Windows Isaac Sim).

__jz_restore_shell_opts() {
  if [[ -n "${__JZ_SETUP_CALLER_SET_OPTS:-}" ]]; then
    eval "${__JZ_SETUP_CALLER_SET_OPTS}"
    unset __JZ_SETUP_CALLER_SET_OPTS
  fi
  trap - RETURN
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  IS_SOURCED=0
  set -eo pipefail
else
  IS_SOURCED=1
  # When sourced, avoid leaking shell option changes (especially `set -e`)
  # into the interactive caller session.
  __JZ_SETUP_CALLER_SET_OPTS="$(set +o)"
  trap __jz_restore_shell_opts RETURN
  set -o pipefail
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION_GRAPH_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WSL_PROFILE="${ACTION_GRAPH_DIR}/fastdds_profile.xml"
WIN_PROFILE="${ACTION_GRAPH_DIR}/fastdds_profile_windows.xml"
DEFAULT_DOMAIN="${JZ_ROS_DOMAIN_ID:-77}"
DEFAULT_RMW="${JZ_RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
DEFAULT_LOCALHOST_ONLY="${JZ_ROS_LOCALHOST_ONLY:-0}"
DEFAULT_MAX_INITIAL_PEERS="${JZ_FASTDDS_MAX_INITIAL_PEERS:-32}"
DEFAULT_USE_BUILTIN_TRANSPORTS="${JZ_FASTDDS_USE_BUILTIN_TRANSPORTS:-1}"
if ! [[ "${DEFAULT_MAX_INITIAL_PEERS}" =~ ^[0-9]+$ ]] || [[ "${DEFAULT_MAX_INITIAL_PEERS}" -lt 4 ]]; then
  DEFAULT_MAX_INITIAL_PEERS=32
fi
if [[ "${DEFAULT_USE_BUILTIN_TRANSPORTS}" == "0" ]]; then
  XML_USE_BUILTIN_TRANSPORTS="false"
else
  XML_USE_BUILTIN_TRANSPORTS="true"
fi
DISCOVERY_MCAST_PORT=$((7400 + 250 * DEFAULT_DOMAIN))
DISCOVERY_UCAST_PORT_FIRST=$((DISCOVERY_MCAST_PORT + 10))
DISCOVERY_UCAST_PORT_LAST=$((DISCOVERY_UCAST_PORT_FIRST + 2 * (DEFAULT_MAX_INITIAL_PEERS - 1)))

is_ipv4() {
  local ip="$1"
  [[ "${ip}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]
}

unique_values() {
  local v
  declare -A seen=()
  for v in "$@"; do
    [[ -z "${v}" ]] && continue
    if [[ -z "${seen[${v}]:-}" ]]; then
      seen["${v}"]=1
      printf '%s\n' "${v}"
    fi
  done
}

get_default_gateway() {
  local gw_hex
  gw_hex="$(awk '$2=="00000000" {print $3; exit}' /proc/net/route)"
  if [[ -z "${gw_hex}" ]]; then
    return 1
  fi
  printf "%d.%d.%d.%d" \
    "$((16#${gw_hex:6:2}))" \
    "$((16#${gw_hex:4:2}))" \
    "$((16#${gw_hex:2:2}))" \
    "$((16#${gw_hex:0:2}))"
}

list_local_ipv4_candidates() {
  awk '
    /^[[:space:]]*\|-- / { cand=$2 }
    /\/32 host LOCAL/ { print cand }
  ' /proc/net/fib_trie \
    | grep -E '^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)' \
    || true
}

discover_windows_ips() {
  local ipconfig_out="" line ip
  local -a candidates=()

  if command -v powershell.exe >/dev/null 2>&1; then
    ipconfig_out="$(
      powershell.exe -NoProfile -Command "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; ipconfig" 2>/dev/null \
        | tr -d '\r'
    )"
    while IFS= read -r line; do
      ip="$(printf '%s\n' "${line}" | grep -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -n1 || true)"
      if ! is_ipv4 "${ip}"; then
        continue
      fi
      case "${ip}" in
        127.*|169.254.*|0.*) continue ;;
      esac
      candidates+=("${ip}")
    done < <(printf '%s\n' "${ipconfig_out}" | grep -i "IPv4" || true)
  fi

  if [[ ${#candidates[@]} -eq 0 ]]; then
    return 1
  fi

  unique_values "${candidates[@]}"
}

detect_win_ip() {
  # Prefer querying Windows IPs directly. In WSL mirrored networking mode,
  # WSL IP may equal Windows Wi-Fi IP, so allow localhost fallback later.
  local wsl_ip="$1"
  local prefix gw_ip ip
  local -a candidates=()
  local -a preferred=()

  mapfile -t candidates < <(discover_windows_ips || true)
  if [[ ${#candidates[@]} -gt 0 ]]; then
    if [[ -n "${wsl_ip}" ]]; then
      prefix="${wsl_ip%.*}."
      for ip in "${candidates[@]}"; do
        if [[ "${ip}" == "${prefix}"* ]]; then
          preferred+=("${ip}")
        fi
      done
    fi

    if [[ ${#preferred[@]} -eq 0 ]]; then
      preferred=("${candidates[@]}")
    fi

    # Prefer a same-subnet candidate that is not exactly equal to WSL IP.
    for ip in "${preferred[@]}"; do
      if [[ -n "${wsl_ip}" && "${ip}" == "${wsl_ip}" ]]; then
        continue
      fi
      printf "%s" "${ip}"
      return 0
    done

    # Last choice from Windows adapters.
    printf "%s" "${preferred[0]}"
    return 0
  fi

  # Fallback: parse default gateway from /proc/net/route (little-endian hex).
  gw_ip="$(get_default_gateway || true)"
  if [[ -z "${gw_ip}" ]]; then
    return 1
  fi

  if [[ -n "${wsl_ip}" ]]; then
    local -a local_ips=()
    local wsl_prefix gw_prefix
    mapfile -t local_ips < <(list_local_ipv4_candidates || true)
    if [[ ${#local_ips[@]} -gt 1 ]]; then
      wsl_prefix="${wsl_ip%.*}."
      gw_prefix="${gw_ip%.*}."
      if [[ "${wsl_prefix}" == "${gw_prefix}" ]]; then
        for ip in "${local_ips[@]}"; do
          if [[ "${ip%.*}." != "${gw_prefix}" ]]; then
            printf "%s" "${wsl_ip}"
            return 0
          fi
        done
      fi
    fi
  fi

  printf "%s" "${gw_ip}"
}

detect_wsl_ip() {
  # Prefer RFC1918 private IPv4 from fib_trie (works in restricted envs).
  local ip gw prefix
  local -a candidates=()
  mapfile -t candidates < <(list_local_ipv4_candidates | head -n 20)
  if [[ ${#candidates[@]} -gt 0 ]]; then
    gw="$(get_default_gateway || true)"
    if [[ -n "${gw}" ]]; then
      prefix="${gw%.*}."
      for ip in "${candidates[@]}"; do
        if [[ "${ip}" == "${prefix}"* ]]; then
          printf "%s" "${ip}"
          return 0
        fi
      done
    fi

    for ip in "${candidates[@]}"; do
      if [[ "${ip}" != "10.255.255.254" ]]; then
        printf "%s" "${ip}"
        return 0
      fi
    done

    printf "%s" "${candidates[0]}"
    return 0
  fi

  # Fallback: try iproute2 if available.
  if command -v ip >/dev/null 2>&1; then
    ip="$(
      ip -4 -o addr show scope global 2>/dev/null \
        | awk '{split($4,a,"/"); print a[1]}' \
        | grep -E '^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)' \
        | head -n1
    )"
    if [[ -z "${ip}" ]]; then
      ip="$(
        ip -4 -o addr show scope global 2>/dev/null \
          | awk '{split($4,a,"/"); print a[1]}' \
          | head -n1
      )"
    fi
    if [[ -n "${ip}" ]]; then
      printf "%s" "${ip}"
      return 0
    fi
  fi

  # Last fallback: any LOCAL IPv4 in fib_trie.
  ip="$(
    awk '
      /^[[:space:]]*\|-- / { cand=$2 }
      /\/32 host LOCAL/ { print cand }
    ' /proc/net/fib_trie \
      | grep -E -v '^(127\.|0\.)' \
      | head -n1
  )"
  printf "%s" "${ip}"
}

emit_initial_peers_xml() {
  local -a ips=("$@")
  local ip port participant_id

  for ip in "${ips[@]}"; do
    port="${DISCOVERY_MCAST_PORT}"
    cat <<EOL
            <locator>
              <udpv4>
                <address>${ip}</address>
                <port>${port}</port>
              </udpv4>
            </locator>
EOL

    for ((participant_id = 0; participant_id < DEFAULT_MAX_INITIAL_PEERS; participant_id++)); do
      port=$((DISCOVERY_UCAST_PORT_FIRST + 2 * participant_id))
      cat <<EOL
            <locator>
              <udpv4>
                <address>${ip}</address>
                <port>${port}</port>
              </udpv4>
            </locator>
EOL
    done
  done
}

WSL_IP="${JZ_WSL_IP:-}"
if [[ -z "${WSL_IP}" ]]; then
  WSL_IP="$(detect_wsl_ip || true)"
fi

WIN_IP="${JZ_WIN_IP:-}"
if [[ -z "${WIN_IP}" ]]; then
  WIN_IP="$(detect_win_ip "${WSL_IP}" || true)"
fi

if [[ -z "${WIN_IP}" || -z "${WSL_IP}" ]]; then
  echo "ERROR: failed to detect Windows/WSL IP (WIN_IP='${WIN_IP}', WSL_IP='${WSL_IP}')"
  return 1 2>/dev/null || exit 1
fi

declare -a WSL_PEER_IPS WIN_PEER_IPS
if [[ "${WIN_IP}" == "${WSL_IP}" ]]; then
  # Mirrored networking mode hint: include localhost to improve cross-stack discovery.
  WSL_PEER_IPS=("${WIN_IP}" "127.0.0.1")
  WIN_PEER_IPS=("${WSL_IP}" "127.0.0.1")
  MIRRORED_NET_HINT=1
else
  WSL_PEER_IPS=("${WIN_IP}")
  WIN_PEER_IPS=("${WSL_IP}")
  MIRRORED_NET_HINT=0
fi

if [[ -n "${JZ_EXTRA_PEER_IPS:-}" ]]; then
  IFS=',; ' read -r -a EXTRA_IPS <<< "${JZ_EXTRA_PEER_IPS}"
  WSL_PEER_IPS+=("${EXTRA_IPS[@]}")
  WIN_PEER_IPS+=("${EXTRA_IPS[@]}")
fi

mapfile -t WSL_PEER_IPS < <(unique_values "${WSL_PEER_IPS[@]}")
mapfile -t WIN_PEER_IPS < <(unique_values "${WIN_PEER_IPS[@]}")

if [[ ${#WSL_PEER_IPS[@]} -eq 0 || ${#WIN_PEER_IPS[@]} -eq 0 ]]; then
  echo "ERROR: peer ip list is empty after filtering"
  return 1 2>/dev/null || exit 1
fi

WSL_PEERS_XML="$(emit_initial_peers_xml "${WSL_PEER_IPS[@]}")"
WIN_PEERS_XML="$(emit_initial_peers_xml "${WIN_PEER_IPS[@]}")"

cat > "${WSL_PROFILE}" <<EOF_XML
<?xml version="1.0" encoding="UTF-8" ?>
<dds>
  <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
      <transport_descriptor>
        <transport_id>udp_transport</transport_id>
        <type>UDPv4</type>
        <sendBufferSize>1048576</sendBufferSize>
        <receiveBufferSize>1048576</receiveBufferSize>
        <non_blocking_send>true</non_blocking_send>
        <maxMessageSize>65500</maxMessageSize>
        <maxInitialPeersRange>${DEFAULT_MAX_INITIAL_PEERS}</maxInitialPeersRange>
      </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="default_participant" is_default_profile="true">
      <rtps>
        <builtin>
          <discovery_config>
            <discoveryProtocol>SIMPLE</discoveryProtocol>
            <leaseDuration>
              <sec>DURATION_INFINITY</sec>
            </leaseDuration>
          </discovery_config>
          <initialPeersList>
${WSL_PEERS_XML}
          </initialPeersList>
        </builtin>
        <useBuiltinTransports>${XML_USE_BUILTIN_TRANSPORTS}</useBuiltinTransports>
        <userTransports>
          <transport_id>udp_transport</transport_id>
        </userTransports>
      </rtps>
    </participant>
  </profiles>
</dds>
EOF_XML

cat > "${WIN_PROFILE}" <<EOF_XML
<?xml version="1.0" encoding="UTF-8" ?>
<dds>
  <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
      <transport_descriptor>
        <transport_id>udp_transport</transport_id>
        <type>UDPv4</type>
        <sendBufferSize>1048576</sendBufferSize>
        <receiveBufferSize>1048576</receiveBufferSize>
        <non_blocking_send>true</non_blocking_send>
        <maxMessageSize>65500</maxMessageSize>
        <maxInitialPeersRange>${DEFAULT_MAX_INITIAL_PEERS}</maxInitialPeersRange>
      </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="default_participant" is_default_profile="true">
      <rtps>
        <builtin>
          <discovery_config>
            <discoveryProtocol>SIMPLE</discoveryProtocol>
            <leaseDuration>
              <sec>DURATION_INFINITY</sec>
            </leaseDuration>
          </discovery_config>
          <initialPeersList>
${WIN_PEERS_XML}
          </initialPeersList>
        </builtin>
        <useBuiltinTransports>${XML_USE_BUILTIN_TRANSPORTS}</useBuiltinTransports>
        <userTransports>
          <transport_id>udp_transport</transport_id>
        </userTransports>
      </rtps>
    </participant>
  </profiles>
</dds>
EOF_XML

# Load ROS2 env
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
else
  echo "WARN: /opt/ros/humble/setup.bash not found"
fi

if [ -f /mnt/e/jz_robot/install/setup.bash ]; then
  source /mnt/e/jz_robot/install/setup.bash
fi

export ROS_DOMAIN_ID="${DEFAULT_DOMAIN}"
export RMW_IMPLEMENTATION="${DEFAULT_RMW}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${WSL_PROFILE}"
export FASTDDS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE}"
export ROS_LOCALHOST_ONLY="${DEFAULT_LOCALHOST_ONLY}"

CURRENT_PYTHON3="$(command -v python3 2>/dev/null || true)"
if [[ -n "${CURRENT_PYTHON3}" ]]; then
  if [[ "${CURRENT_PYTHON3}" == *"/miniconda"* ]] || [[ "${CURRENT_PYTHON3}" == *"/anaconda"* ]]; then
    export PATH="/usr/bin:${PATH}"
    hash -r
    CURRENT_PYTHON3="$(command -v python3 2>/dev/null || true)"
  fi
fi
export JZ_PYTHON_BIN="${CURRENT_PYTHON3:-/usr/bin/python3}"

ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null 2>&1 || true

echo "=========================================="
echo "  WSL ROS2 environment configured"
echo "=========================================="
echo "  WSL_IP: ${WSL_IP}"
echo "  WIN_IP: ${WIN_IP}"
echo "  WSL_PEERS: ${WSL_PEER_IPS[*]}"
echo "  WIN_PEERS: ${WIN_PEER_IPS[*]}"
echo "  ROS_DOMAIN_ID: ${ROS_DOMAIN_ID}"
echo "  RMW_IMPLEMENTATION: ${RMW_IMPLEMENTATION}"
echo "  FASTRTPS_DEFAULT_PROFILES_FILE: ${FASTRTPS_DEFAULT_PROFILES_FILE}"
echo "  FASTDDS_DEFAULT_PROFILES_FILE: ${FASTDDS_DEFAULT_PROFILES_FILE}"
echo "  PYTHON_BIN: ${JZ_PYTHON_BIN}"
echo "  ROS_LOCALHOST_ONLY: ${ROS_LOCALHOST_ONLY}"
echo "  FASTDDS_USE_BUILTIN_TRANSPORTS: ${XML_USE_BUILTIN_TRANSPORTS}"
echo "  FASTDDS_MAX_INITIAL_PEERS: ${DEFAULT_MAX_INITIAL_PEERS}"
echo "  DISCOVERY_MCAST_PORT: ${DISCOVERY_MCAST_PORT}"
echo "  DISCOVERY_UCAST_PORT_RANGE: ${DISCOVERY_UCAST_PORT_FIRST}-${DISCOVERY_UCAST_PORT_LAST}"
if [[ "${MIRRORED_NET_HINT}" -eq 1 ]]; then
  echo "  WARN: WIN_IP equals WSL_IP (mirrored networking). Added 127.0.0.1 fallback peers."
fi
echo "=========================================="
echo "Windows side should use:"
echo "  set ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "  set RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "  set FASTRTPS_DEFAULT_PROFILES_FILE=E:\\jz_robot\\robot_simulator\\isaac_sim_implementations\\action_graph_method\\fastdds_profile_windows.xml"
echo "  set FASTDDS_DEFAULT_PROFILES_FILE=%%FASTRTPS_DEFAULT_PROFILES_FILE%%"
echo "Optional overrides:"
echo "  export JZ_WSL_IP=<wsl_ip>"
echo "  export JZ_WIN_IP=<windows_ip>"
echo "  export JZ_EXTRA_PEER_IPS=<ip1,ip2,...>"
if [[ "${IS_SOURCED}" -eq 0 ]]; then
  echo "NOTE: This script was executed, not sourced."
  echo "      Current shell env is unchanged. Use:"
  echo "      source /mnt/e/jz_robot/robot_simulator/isaac_sim_implementations/action_graph_method/setup_wsl_ros2_fixed.sh"
fi
