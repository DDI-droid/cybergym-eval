#!/usr/bin/env bash
set -euo pipefail

runner_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="${CYBERGYM_ROOT:-${HOME}/cybergym}"
state_dir="${HOME}/.local/state/cybergym-eval"
apply=0
if [[ "${1:-}" == "--yes" ]]; then
  apply=1
fi

marked() {
  [[ -f "$state_dir/$1" ]]
}

step() {
  echo "- $1"
  shift
  if [[ $apply -eq 1 ]]; then
    "$@"
  fi
}

remove_images() {
  while read -r image; do
    if [[ -n "$image" ]]; then
      docker image rm "$image" >/dev/null || echo "  could not remove $image" >&2
    fi
  done < "$state_dir/images"
}

remove_data() {
  while read -r dir; do
    if [[ -n "$dir" ]]; then
      rm -rf "$dir"
    fi
  done < "$state_dir/data_dirs"
}

stop_servers() {
  while read -r pidfile; do
    if [[ -f "$pidfile" ]]; then
      pid="$(cat "$pidfile")"
      if [[ -O "/proc/$pid" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q -e "cybergym.server" -e "cybergym_eval.shim"; then
        kill -TERM -- "-$pid" 2>/dev/null || true
      fi
      rm -f "$pidfile"
    fi
  done < "$state_dir/server_pid_files"
}

restore_sysctl() {
  if marked mmap_rnd_bits.orig; then
    sudo sysctl -w vm.mmap_rnd_bits="$(cat "$state_dir/mmap_rnd_bits.orig")"
  fi
  if marked sysctl_file_added; then
    sudo rm -f /etc/sysctl.d/99-cybergym.conf
  fi
}

if [[ ! -d "$state_dir" ]]; then
  echo "Nothing recorded in ${state_dir}; nothing to revert."
  exit 0
fi

if [[ $apply -eq 1 ]]; then
  echo "Reverting:"
else
  echo "Dry run, nothing changes. Run with --yes to revert:"
fi

if marked server_pid_files; then
  step "stop the submission server" stop_servers
fi
if marked firewall_started; then
  step "stop the CyberGym firewall (proxy container and cybergym-internal network)" "${benchmark_root}/.venv/bin/python" -m cybergym.firewall stop-all
fi
if marked images; then
  step "remove task images: $(tr '\n' ' ' < "$state_dir/images")" remove_images
fi
if marked data_dirs; then
  step "delete downloaded task data: $(tr '\n' ' ' < "$state_dir/data_dirs")" remove_data
fi
if [[ -d "${runner_root}/.venv" ]]; then
  step "delete ${runner_root}/.venv" rm -rf "${runner_root}/.venv"
fi
if marked cybergym_cloned; then
  step "delete upstream checkout ${benchmark_root}" rm -rf "${benchmark_root}"
fi
if marked secrets_created; then
  step "delete ${HOME}/.secrets (Bedrock key and submission-server API key)" rm -rf "${HOME}/.secrets"
fi
if marked mmap_rnd_bits.orig || marked sysctl_file_added; then
  step "set vm.mmap_rnd_bits back to $(cat "$state_dir/mmap_rnd_bits.orig" 2>/dev/null || echo "its current value") and delete /etc/sysctl.d/99-cybergym.conf" restore_sysctl
fi
step "delete the change record ${state_dir}" rm -rf "$state_dir"
echo "Kept: ${runner_root} (the eval code and its git history)."
