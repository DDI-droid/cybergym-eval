"""Stop only the selected work-root run and preserve its evidence before maintenance."""
import concurrent.futures
import json
import os
import signal
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import docker
from cybergym_eval.config import load_config, work_id
from cybergym_eval.shim import block_key, key_usage

config = load_config('config/train25_think_ec.json')
root = Path(config['work_root'])
stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
archive = root / 'archive' / ('oni-before-five-fixes-' + stamp)
archive.mkdir(parents=True)
marker = ('CYBERGYM_EVAL_WORK=' + work_id(config)).encode()
processes = []
for path in Path('/proc').iterdir():
    if not path.name.isdigit() or int(path.name) == os.getpid():
        continue
    try:
        if path.stat().st_uid != os.getuid():
            continue
        args = (path / 'cmdline').read_bytes()
        env = (path / 'environ').read_bytes().split(b'\0')
        is_driver = b'cybergym-eval\0run\0' in args and b'train25_think_ec.json' in args
        is_launcher = b'/home/anshul/tmp/ec_run.sh\0train25_think_ec' in args
        if marker in env or is_driver or is_launcher:
            processes.append(int(path.name))
    except OSError:
        pass
for pid in processes:
    try:
        os.kill(pid, signal.SIGSTOP)
    except ProcessLookupError:
        pass
client = docker.from_env()
containers = client.containers.list(all=True, filters={'label': 'cybergym-eval.work=' + work_id(config)})
usage = []
for container in containers:
    container.reload()
    env = dict(s.split('=', 1) for s in container.attrs['Config'].get('Env', []) if '=' in s)
    key = env.get('OPENAI_API_KEY')
    if key:
        try:
            block_key(config, key)
            usage.append({'container': container.name, 'usage': key_usage(config, key)})
        except Exception as error:
            usage.append({'container': container.name, 'error': type(error).__name__})
(archive / 'active-usage.json').write_text(json.dumps(usage, indent=2))

def stop(container):
    try:
        container.stop(timeout=10)
        (archive / (container.name + '.log')).write_bytes(container.logs())
    except Exception as error:
        return {'container': container.name, 'error': str(error)}
    return {'container': container.name, 'stopped': True}

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
    stopped = list(pool.map(stop, containers))
for pid in processes:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
# Stop the work-root's shim so its old in-flight retry handlers cannot spend after cancellation.
pid_path = root / 'shim/shim.pid'
if pid_path.exists():
    pid = int(pid_path.read_text())
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
        if b'cybergym_eval.shim' in command and Path(f'/proc/{pid}').stat().st_uid == os.getuid():
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except FileNotFoundError:
        pass
    time.sleep(1)
db_path = root / 'train25/state/runs.sqlite3'
con = sqlite3.connect(db_path)
backup = sqlite3.connect(archive / 'runs.before.sqlite3')
con.backup(backup)
backup.close()
con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute('SELECT * FROM executions WHERE label=?', ('codex_think_ec',))]
(archive / 'executions.before.json').write_text(json.dumps(rows, indent=2))
now = datetime.now(timezone.utc).isoformat()
con.execute('UPDATE executions SET status=?, finished_at=?, error=? WHERE label=? AND status IN (?, ?)', ('interrupted', now, 'Operator stopped for Oni readiness fixes; original state preserved in ' + str(archive), 'codex_think_ec', 'running', 'pending'))
con.commit()
con.close()
subprocess.run(['tar', '-czf', str(archive / 'run-evidence.tgz'), '-C', str(root), 'train25', 'shim'], check=True)
subprocess.run(['tar', '-czf', str(archive / 'eval-source.tgz'), 'src', 'config', 'scripts', 'pyproject.toml'], check=True)
subprocess.run(['tar', '-czf', str(archive / 'harness-source.tgz'), '-C', config['harness_repo'], 'harnesses', 'common', 'build.sh'], check=True)
(archive / 'stop.json').write_text(json.dumps({'processes': processes, 'containers': stopped, 'at': now}, indent=2))
print(json.dumps({'archive': str(archive), 'processes_stopped': len(processes), 'containers': stopped, 'running_containers_remaining': len(client.containers.list(filters={'label':'cybergym-eval.work=' + work_id(config)}))}, indent=2))
client.close()
