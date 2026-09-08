# 5. View results

## `training/history.py` -- view past runs

| Flag | Default | Meaning |
|---|---|---|
| `--storage-root` | `./runs` | match whatever `--storage-root` your training runs used |
| `--limit` | `20` | most recent N runs to show |

```bash
python -m training.history --storage-root training/runs --limit 50
```

One line is appended to `<storage_root>/history.jsonl` per run (success or
failure) -- `history.py` is just a formatted view over that file; `cat`/`jq`
it directly if you want something custom.

## What a run produces, all under `<storage_root>/<run_name>/`

- `tensorboard/` -- live loss/l1/kl curves + learning rates:
  `tensorboard --logdir <path>` (the exact command is printed at the end of
  every run).
- `ray_data_stats.txt` -- per-operator Ray Data execution stats for what
  that run's shard actually consumed (throughput, block sizes, remote/UDF time).
- Checkpoints -- by default, written on every report and pruned to the
  `--checkpoint-max-to-keep` (default 3) lowest-loss ones, PLUS the single
  most recently written checkpoint even when it isn't among those N -- that
  extra one is what a resumed run (`ray.train.get_checkpoint()`, triggered
  automatically on a worker failure/restart within the same run) picks up
  from, so it's never more than one `--eval-every-steps` window stale
  relative to wherever training actually stopped. Confirmed by reading the
  installed `ray.train.v2` checkpoint manager's pruning logic (it excludes
  `self._latest_checkpoint_result` from its deletion set), not assumed.
  `--save-only-on-improvement` switches to the old behavior instead: a
  checkpoint only written when the loss improves (both window- and
  epoch-level reports share one running best) -- less write I/O, but no
  separate always-fresh checkpoint, so a resume falls back to whichever
  improving checkpoint happened most recently.

## Resuming after an interruption

A crashed/interrupted run resumes automatically within the same
`trainer.fit()` call, up to `FailureConfig(max_failures=1)` -- Ray Train
relaunches a fresh worker and it picks up from the latest checkpoint. This
was tested for real, deliberately: a crash injected mid-run (after a
checkpoint had already been saved), for ACT and MolmoAct2 (both DDP and
FSDP2), confirmed each resumed correctly to the original target step count
-- not just from the driver's own retry logic, but by making sure the
relaunched worker read back the right `epoch`/`step`/`epoch_complete` state.
See ["Resume-from-checkpoint" in Verification status](verification-status.md)
for the real bug this testing method found and fixed (a mid-epoch checkpoint
being silently treated as end-of-epoch, discarding all progress on resume).

## Next

That's the core workflow -- [Customizing datasets](customizing-datasets.md)
and [Customizing policies](customizing-policies.md) cover the parts you're
expected to adapt for your own use case.
