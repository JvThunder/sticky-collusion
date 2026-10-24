# Run records for the sticky-collusion report

`runs/` holds every run behind `reports/sticky-collusion/report.pdf`, in the same folder
layout as `results/`:

- `{luna6,ds,g31l}-branches/<pair>/<launch>_<datetime>/rep<N>/run.json`: saved histories
  (`build*` launches) and every arm branched from them; cross-model arms sit in
  `<alice model>__<bob model>/` folders.
- `schedule/<model>/no-reward_<datetime>/`: the never-rewarded reference pairs.
- `agreement.csv` next to each build run: the agreement judge's per-episode labels, used to
  classify how collusion began.

Each `run.json` keeps the run configuration (including the agents' system prompts) and, for
every episode, the arm flags, rewards and each agent's task, ground truth, verdict, stated
verdict reasons and reflection. The raw records also stored both agents' full conversation
again at every episode (several GB); those transcripts are not included here.

Every number and figure in the report can be rebuilt from this folder alone:

```bash
STICKY_RESULTS=data/sticky-collusion/runs python reports/sticky-collusion/make_figures.py
```

`reports/sticky-collusion/export_runs.py` writes this folder from a full `results/` tree.
