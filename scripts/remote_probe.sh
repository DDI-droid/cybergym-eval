#!/usr/bin/env bash
set -u
printf 'user=%s\n' "$(id -un)"
printf 'os=%s\n' "$(. /etc/os-release && printf '%s' "${PRETTY_NAME}")"
printf 'kernel=%s\n' "$(uname -r)"
printf 'cpus=%s\n' "$(nproc)"
printf 'load=%s\n' "$(cut -d' ' -f1-3 /proc/loadavg)"
printf 'mmap_rnd_bits=%s\n' "$(sysctl -n vm.mmap_rnd_bits 2>/dev/null || sudo -n sysctl -n vm.mmap_rnd_bits 2>/dev/null || printf unreadable)"
printf 'python=%s\n' "$(python3 --version 2>&1)"
printf 'uv=%s\n' "$(command -v uv || command -v "${HOME}/.local/bin/uv" || printf absent)"
printf 'memory:\n'
free -h
printf 'root_disk:\n'
df -h /
printf 'docker=%s\n' "$(docker version --format '{{.Server.Version}}' 2>/dev/null || printf unavailable)"
if docker info >/dev/null 2>&1; then
  printf 'docker_access=yes\n'
  printf 'docker_disk:\n'
  docker system df
  printf 'cybergym_network_gateway=%s\n' "$(docker network inspect cybergym-internal --format '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || printf absent)"
  printf 'cybergym_proxy=%s\n' "$(docker inspect cybergym-proxy --format '{{.State.Status}}' 2>/dev/null || printf absent)"
  printf 'running_containers=%s\n' "$(docker ps -q | wc -l)"
else
  printf 'docker_access=no\n'
fi
