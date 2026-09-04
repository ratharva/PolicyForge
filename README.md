# PolicyForge

A Ray Data + Ray Train pipeline for finetuning robot-learning policies on
robot teleop data. Multi-policy and multi-dataset by design -- ACT,
MolmoAct2, and π0.5 today, on XDOF/ABC-130k, AgiBot World Alpha, or DROID,
built so adding another policy or dataset is a small, additive change (a
new file, not a rework of existing code).

- [`docs/`](docs/README.md) -- user-facing docs: one file per workflow
  step, plus deep dives on customizing datasets and policies.
- [`CLAUDE.md`](CLAUDE.md) -- agent-facing architecture notes, the
  add-a-new-policy/add-a-new-dataset playbooks, and hard-won implementation
  gotchas.

## Quick start

```bash
pip install -r training/requirements.txt
export HF_TOKEN=$(cat hf_tok.txt)   # after accepting the dataset's gated terms

python -m training.prepare_data --tasks <task-name> --max-episodes-per-task 20
python -m training.train --tasks <task-name> --max-train-steps 50
```

See [`docs/README.md`](docs/README.md) for the full step-by-step guide and
command reference.
