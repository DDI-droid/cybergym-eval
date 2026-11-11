# cybergym-eval

A runner for evaluating coding agents on [CyberGym](https://github.com/sunblaze-ucb/cybergym). Ray handles scheduling across a fixed number of concurrency slots, SQLite tracks every execution, and each run freezes the exact harness code it used — so a result is always traceable back to the code that produced it, not just the harness name. A metered LLM shim sits between the agent and the model and issues a scoped key per execution.

I built this to run a lot of agent/harness combinations against CyberGym without babysitting a shell script. Full writeup — results, the ASLR false-positive bug, what actually moved the numbers — here: [prescera interview report](https://ddi-droid.github.io/blog/2026/prescera-interview-report/).

## Setup

```bash
scripts/setup.sh
```

Clones upstream `cybergym`, sets up both venvs with `uv`, and locks `vm.mmap_rnd_bits` to 28. On a stock Ubuntu 26.04 box (default 32), old ASan binaries crash on startup often enough to register as false positives — worth doing before your first run, not after you notice weird numbers.

## Running an eval

```bash
cybergym-eval prepare --config config/train25.json   # freezes harness code, pulls images
cybergym-eval run     --config config/train25.json   # launches, one container per task
cybergym-eval status  --config config/train25.json   # live tally while it runs
cybergym-eval report  --config config/train25.json --output report.json
```

A config file names a taskset, a harness label, and a slot count. `cybergym-eval viz` turns a finished run into an HTML trajectory viewer; `sweep --slots 5,10,15,20` finds your actual concurrency ceiling instead of you guessing one.

One sharp edge worth knowing up front: `run` never re-touches a task that already has a database row, in any status. If you stop a run partway through, resuming it means resetting the stopped rows back to `pending` yourself before relaunching — `recover` only handles rows stuck in `running` from a crash, not ones you deliberately interrupted.

Needs Docker and AWS Bedrock credentials (`~/.secrets/bedrock.env`, `KEY=VALUE` lines) on the host.
