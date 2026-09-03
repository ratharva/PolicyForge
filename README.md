# PolicyForge

A Ray Data + Ray Train pipeline for finetuning robot-learning policies on
MCAP teleop data. Multi-policy by design -- ACT, MolmoAct2, and π0.5 today,
built so adding another policy is a small, additive change.

- [`CLAUDE.md`](CLAUDE.md) -- agent-facing architecture notes, the
  add-a-new-policy playbook, and hard-won implementation gotchas.
- [`training/README.md`](training/README.md) -- user-facing operational
  docs: setup, every CLI flag, data pipeline design, deployment.

## Quick start

```bash
pip install -r training/requirements.txt
export HF_TOKEN=$(cat hf_tok.txt)   # after accepting the dataset's gated terms
python -m training.train --tasks <task-name> --max-episodes-per-task 20 --max-train-steps 50
```

See `training/README.md` for the full command reference.
