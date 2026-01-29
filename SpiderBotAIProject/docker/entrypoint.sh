#!/usr/bin/env bash
set -euo pipefail

LOCAL_UID="${LOCAL_UID:-1000}"
LOCAL_GID="${LOCAL_GID:-1000}"
LOCAL_USER="${LOCAL_USER:-isaac}"

create_group_if_needed() {
  local gid="$1"
  local name="$2"
  if getent group "${gid}" >/dev/null 2>&1; then
    return 0
  fi
  groupadd -g "${gid}" "${name}"
}

create_user_if_needed() {
  local uid="$1"
  local gid="$2"
  local name="$3"
  if getent passwd "${uid}" >/dev/null 2>&1; then
    return 0
  fi
  useradd -m -u "${uid}" -g "${gid}" -s /bin/bash "${name}"
}

create_group_if_needed "${LOCAL_GID}" "${LOCAL_USER}"
create_user_if_needed "${LOCAL_UID}" "${LOCAL_GID}" "${LOCAL_USER}"

USER_NAME="$(getent passwd "${LOCAL_UID}" | cut -d: -f1)"
HOME_DIR="${HOME:-}"
if [ -z "${HOME_DIR}" ]; then
  HOME_DIR="$(getent passwd "${LOCAL_UID}" | cut -d: -f6)"
fi
export HOME="${HOME_DIR}"

# Ensure common cache/data/log locations are writable for the runtime user.
# These may be mounted as docker volumes and default to root ownership.
ensure_writable_dir() {
  local path="$1"
  mkdir -p "${path}"
  chown -R "${LOCAL_UID}:${LOCAL_GID}" "${path}"
}

ensure_writable_dir "/isaac-sim/kit/cache"
ensure_writable_dir "${HOME}/.cache/ov"
ensure_writable_dir "${HOME}/.cache/pip"
ensure_writable_dir "${HOME}/.cache/nvidia/GLCache"
ensure_writable_dir "${HOME}/.nv/ComputeCache"
ensure_writable_dir "${HOME}/.nvidia-omniverse/logs"
ensure_writable_dir "${HOME}/.local/share/ov/data"
ensure_writable_dir "${HOME}/Documents"

if command -v runuser >/dev/null 2>&1; then
  exec runuser -u "${USER_NAME}" -- "$@"
fi

cmd="$(printf '%q ' "$@")"
if su -p -s /bin/bash "${USER_NAME}" -c "true" >/dev/null 2>&1; then
  exec su -p -s /bin/bash "${USER_NAME}" -c "exec ${cmd}"
fi
exec su -s /bin/bash "${USER_NAME}" -c "exec ${cmd}"
