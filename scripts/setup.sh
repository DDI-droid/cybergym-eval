#!/usr/bin/env bash
set -euo pipefail

runner_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="${CYBERGYM_ROOT:-${HOME}/cybergym}"
state_dir="${HOME}/.local/state/cybergym-eval"
export PATH="${HOME}/.local/bin:${PATH}"
mkdir -p "$state_dir"

for executable in curl git; do
  command -v "$executable" >/dev/null || {
    echo "Missing $executable; install it before continuing." >&2
    exit 1
  }
done

if ! command -v docker >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y docker.io
  sudo systemctl enable --now docker
fi
if ! id -nG "$(id -un)" | grep -qw docker; then
  sudo usermod -aG docker "$(id -un)"
  echo "Added $(id -un) to the docker group. Log out and back in once." >&2
fi

current_bits="$(sudo -n sysctl -n vm.mmap_rnd_bits 2>/dev/null || true)"
if [[ -n "$current_bits" && ! -f "$state_dir/mmap_rnd_bits.orig" ]]; then
  echo "$current_bits" > "$state_dir/mmap_rnd_bits.orig"
fi
if [[ "$current_bits" != "28" ]]; then
  sudo sysctl -w vm.mmap_rnd_bits=28
fi
if [[ ! -f /etc/sysctl.d/99-cybergym.conf ]]; then
  touch "$state_dir/sysctl_file_added"
fi
printf 'vm.mmap_rnd_bits=28\n' | sudo tee /etc/sysctl.d/99-cybergym.conf >/dev/null
echo "vm.mmap_rnd_bits=$(sudo -n sysctl -n vm.mmap_rnd_bits)"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [[ ! -d "${benchmark_root}/.git" ]]; then
  git clone https://github.com/sunblaze-ucb/cybergym.git "${benchmark_root}"
  touch "$state_dir/cybergym_cloned"
fi
if [[ -n "$(git -C "${benchmark_root}" status --porcelain --untracked-files=no)" ]]; then
  echo "${benchmark_root} has local edits; upstream must stay unmodified." >&2
  exit 1
fi
if git -C "${benchmark_root}" symbolic-ref -q HEAD >/dev/null; then
  git -C "${benchmark_root}" pull -q --ff-only
fi
git -C "${benchmark_root}" submodule update --init --recursive examples/agents

uv python install 3.12

if [[ ! -x "${benchmark_root}/.venv/bin/python" ]]; then
  uv venv --python 3.12 "${benchmark_root}/.venv"
fi
uv pip install --python "${benchmark_root}/.venv/bin/python" -e "${benchmark_root}[dev,server]"

if [[ ! -x "${runner_root}/.venv/bin/python" ]]; then
  uv venv --python 3.12 "${runner_root}/.venv"
fi
uv pip install --python "${runner_root}/.venv/bin/python" -e "${runner_root}"

echo "upstream: $(git -C "${benchmark_root}" log -1 --format='%h %cs %s')"
echo "Bootstrap complete. Changes are recorded in ${state_dir}; scripts/desetup.sh reverts them."
