# Reproduction guide

Use Python 3.10 from the repository root. For CUDA execution, use a PyTorch 2.5.1 build for the installed runtime. The archived Forced-01 runs used PyTorch with CUDA 12.1 on Windows. Original receipts can contain Windows paths and device identities.

## Figures and tables

```sh
python -m pip install -r requirements/plotting.txt
python paper/figures/render.py --paper-inputs /path/to/paper-results --output reproduced/paper
```

Supply an external paper archive containing the paths listed in `INPUT_FILES` in [render.py](../paper/figures/render.py). Add `--check-inputs` to check those paths. For a data-free check, use `--framework-only --output reproduced/framework`.

| Manuscript output | Plot source | Data |
| --- | --- | --- |
| Fig. 1 | `paper/figures/framework_mechanism_cost.py` | Method diagram |
| Figs. 2–4, 6 and S5, S7 | `paper/figures/les_accuracy.py` | Decaying MSE, endpoint fields, IC105 and a priori scores |
| Fig. 5 and Forced-01 table | `paper/figures/forced_01.py` | Forced time series, terminal spectrum and 80-frame errors |
| Figs. 7–8, 11 and S3, S6 | `paper/figures/physical_statistics.py` | Resolved statistics, spectra, SGS transfers and solver timing |
| Figs. 9 and S4 | `paper/figures/framework_mechanism_cost.py` | Head error, interventions and complete-loop timings |
| Fig. 10 | `paper/figures/rotation_ablation.py` | Paired same-weight and unaugmented ablations |
| Figs. S1–S2 | `paper/figures/fdns_sensitivity.py`, `native_dns_spectra.py` | Filtered-reference checks and native DNS spectra |

To recompute the Forced-01 figure input from existing raw DNS/LES files, use `python paper/figures/forced_01.py /path/to/archive/research/JCP_DSF_TemporalSym_20261117`. This verifies the original run receipts and 81 saved fields and writes into ignored output folders under `paper/`.

For an external manuscript tree with the original TeX, bibliography and the generated figures, compile with XeLaTeX and BibTeX using:

```powershell
pwsh -File paper/latex/build.ps1 -PaperRoot "path/to/manuscript"
```

## Frozen numerical workspace

```sh
python -m pip install -r requirements/solver.txt
python tools/check_inputs.py
python reproduce.py prepare --workspace /path/to/new-run
python reproduce.py list
python reproduce.py test --workspace /path/to/new-run
```

The launcher reconstructs original filenames and imports from [frozen_workspace.json](frozen_workspace.json) in the runtime directory, which defaults to `.reproduction/workspace`. The manifest maps to one copy of each source in the repository. Source bytes and production model identifiers are preserved for checksum and receipt verification. The launcher creates source files only and checks for conflicts with pre-existing source files. If you edit source, use a separate runtime directory for the edited version.

To run the full frozen experiments, supply external assets in the runtime directory at the relative paths listed below. The archive may contain additional provenance files referenced by the run receipts. An asset-only check is available from `python tools/check_inputs.py --archive-root /path/to/archive --include-unaugmented`.

| Asset | Original path or locator relative to the runtime workspace |
| --- | --- |
| Network weights and run receipts | Listed by SHA-256 in [frozen_checkpoints.json](../research/closures/dsf_shared/frozen_checkpoints.json). Supply augmented seeds 2026 and 2028. Add the unaugmented weights for the corresponding ablation. |
| Frozen baseline configuration | `results/revision_v2/runs/full/seed_2026_recovery01/effective_config.json` |
| Physical decoder identity | `research/full_aug_orbit_rk4_one_step_v1_20261114/PHASE_ONE_MANIFEST.json` |
| Decaying FDNS | `research/ssc_long_horizon_v1_20261030/results/run01/` for Flow 101 and 102, and the matching `research/full_c4_multi_ic_v1_20261110/results/paired01/` archive for Flow 104. |
| IC105 and validated baselines | `research/full_aug_hold_audit_v1_20261117/independent_ic105/` and the matching `DNS/`, `LES/` and `CLARK/` archives under `research/JCP_DSF_TemporalSym_20261117/VALIDATION_20261118/` |
| Decaying rollouts and interventions | `research/full_aug_parameter_pc_full_v1_20261116/results/run01/` and `research/full_aug_hold_audit_v1_20261117/results/audit01/` for postprocessing. Reference reassessment additionally accepts an external `FDNS_REPLAY` directory through `--replay`. |
| Forced-01 raw fields | `FORCED_01_SHORT_16_20_20261122/runs/forced01_short/` and `FORCED_01_TAIL_PILOT_20261122/runs/forced01_tail_pilot/` in the original study tree, with primary reference receipt and original block hashes. |
| Training inputs | Original training caches, split, baseline registry and acceptance receipts consumed by [training_setup.py](../research/closures/dsf_aug/training/training_setup.py) |

## Numerical entry points

`run` accepts an optional `--workspace` before the task name. Options after `--` are forwarded to the selected program. For example, `python reproduce.py run validate-decaying -- --help` lists available checks.

| Task | Example after `python reproduce.py run` |
| --- | --- |
| Four DSF decaying rollouts | `decaying-rollouts -- --run new_run --execute-full-rollout` |
| NGM4 on decaying flows | `validate-decaying -- clark --execute` |
| IC105 independent LES | `validate-decaying -- les --group ic105 --execute` |
| Single-orientation and refresh ablations | `validate-decaying -- les --group single --execute` or the original groups `period4` and `period8` |
| Unaugmented single-orientation ablation | `orientation-ablation -- check`, then `orientation-ablation -- run` |
| Full-window LES time-step comparison | `time-step-comparison -- --execute-limited` |
| DNS reference reassessment | `reassess-dns -- --replay /path/to/FDNS_REPLAY --output /path/to/output` |
| A priori teacher-forced replay | `a-priori-replay -- --probe` for a two-timestamp check |
| NGM4 stress/flux equivalence | `verify-ngm4 -- --run new_check` |

### Forced-01

The archive uses the `DISSIPATION_TAIL_FAILED_EXPLORATORY` branch on the fixed `[16,20]` window. The original primary DNS receipt must match for the archived-reference LES. A new reference follows its own qualification path.

```powershell
python reproduce.py run forced-reference -- init
python reproduce.py run forced-reference -- dns --variant primary --until 20 --device cuda --execute
python reproduce.py run forced-reference -- qualify --primary-only
```

For the archived reference and a workspace with all inputs:

```powershell
python reproduce.py run forced-rollouts -- init
python reproduce.py run forced-rollouts -- les --model NGM4 --device cuda --execute --allow-dissipation-tail-fail
foreach ($seed in 2026, 2028) {
  foreach ($model in 'DSF-Aug', 'DSF-Sym', 'DSF-Trend', 'DSF-Hold') {
    python reproduce.py run forced-rollouts -- les --model $model --seed $seed --device cuda --execute --allow-dissipation-tail-fail
  }
}
python reproduce.py run score-forced -- --name reproduced
```

The launcher translates the five names to `Clark4`, `Aug`, `Fresh-C4`, `Parameter-PC` and `Parameter-Hold` when forwarding `--model` to the frozen Forced-01 drivers. Saved CSV identifiers and the numerical parameters use the original protocol.
