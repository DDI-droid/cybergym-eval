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
~/cybergym-harnesses/build.sh                              # build the harness images
cybergym-eval prepare      --config config/train25.json    # every check, then freeze settings, code and image
cybergym-eval server start --config config/train25.json
cybergym-eval shim start   --config config/train25.json
cybergym-eval run          --config config/train25.json    # launches, one container per task
cybergym-eval status       --config config/train25.json    # live tally while it runs
cybergym-eval report       --config config/train25.json --output report.json
```

A config file names a taskset, a harness label, and a slot count. `cybergym-eval viz` turns a finished run into an HTML trajectory viewer; `sweep --slots 5,10,15,20` finds your actual concurrency ceiling instead of you guessing one.

`prepare` does all the checking, before it writes anything. It refuses when a label's settings, harness code or harness image differ from what the label was frozen with (use a new label), and when a task set name is reused for different tasks (use a new task set name). Its last step writes `<label>/prepared.<split>.json`. `run` checks only that its config is the one that was prepared, plus the machine itself (ASLR, Docker, the two services, disk); containers start from the frozen code and the frozen image id, so rebuilding or editing a harness cannot reach a run in progress. Run `prepare` before every `run`: on an unchanged setup it just re-checks.

`run` resumes: it re-runs every execution that is not `passed` or `benchmark_failed`, including `infra_error`, `interrupted` and `stale_running` rows. After a crash, `recover` first turns rows stuck in `running` into their real result or into `stale_running`.

Tasks are a list of ids, or a seeded sample. A sample is one task set unless it names parts, as fractions that add up to 1; `split` then picks the part to run, or `all`:

```json
"tasks": {"sample": 100, "seed": 0, "splits": {"train": 0.75, "test": 0.25}},
"split": "train"
```

## Models

Each harness names the models it uses in its own `harness.toml` (`models = ["deepseek.v3.2"]`), so changing a harness's model changes its code hash and needs a new label, like any other harness change. The container gets the names as `CYBERGYM_MODELS` (a JSON list); it never sees a URL or a credential. Where each model is served, and what it costs, is a resource of the machine and lives in the run config:

```json
"models": {
  "deepseek.v3.2": {"base_url": "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
                    "api_key_env": ["OPENAI_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"],
                    "price_in_per_m": 0.62, "price_out_per_m": 1.85}
}
```

`prepare` refuses a harness whose models are not all in this table. The table is not frozen into labels: editing an entry or adding a model needs no new label. The shim serves the whole table, issues each execution a key for its harness's models only, refuses a request for any other model, and routes each request to its model's endpoint. Preflight fails if the running shim serves a different table (`shim stop`, then `shim start`).

`benchmark_slots`, `start_stagger_seconds`, ports, `env_file` and `experiment_note` can also change between runs of a label; everything else in the config is frozen with it.

## Or drive it from Python: the Gym

The CLI drives a `harnesses/` directory of Docker images through JSON configs. `cybergym_eval.gym.Gym` is the same machinery as a library, for when you want to try an agent interactively instead of writing a config file:

```python
from cybergym_eval.gym import Gym

gym = Gym(work_root="~/cybergym-gym", benchmark_slots=4)
train = gym.taskset("train25", tasks=TRAIN_IDS)   # TRAIN_IDS: a list of task ids you write
smoke = train.sample(5, seed=0)                    # 5 of those tasks: the same task set, fewer tasks

class MyAgent:
    def solve(self, task, box, lm):
        ...  # box is the sandbox, lm is the metered model client

agent = gym.agent(MyAgent, temperature=0.2)
for r in agent.rollouts(smoke):
    print(r.task, r.passed, r.feedback)
```

A task set works as it does in a config: a name stands for one fixed set of tasks, saved in `<work_root>/<name>/taskset.json`.

- `gym.taskset("train25", tasks=[...])` creates a set of the tasks you list.
- `gym.taskset("pool50", sample=50, seed=0)` creates a set drawn from CyberGym's task list, the same draw a config with `"tasks": {"sample": 50, "seed": 0}` makes, except that the Gym's `held_out` tasks are left out first (the file records which).
- `gym.taskset("train25")` opens a saved set by name, including one a CLI config made. Asking for other tasks under a saved name is refused; use a new name.
- `train.sample(5, seed=0)` picks 5 tasks of a set. It is not saved, because the same number and seed always give the same tasks; rollouts of it go to the set's own folder.

`gym.agent()` also takes a prebuilt Docker harness straight from `cybergym-harnesses` in place of a Python class — same `Rollout` objects either way, so a container-based harness and a one-off Python agent are interchangeable from the caller's side.

Needs Docker and AWS Bedrock credentials (`~/.secrets/bedrock.env`, `KEY=VALUE` lines) on the host.
