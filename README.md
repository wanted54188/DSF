# DSF-SGS paper reproduction

Code for the five SGS closures in the DSF-SGS study and for the associated DNS reference, LES experiments and figures.

| Paper topic | Source |
| --- | --- |
| NGM4 analytic closure | [research/closures/ngm4/](research/closures/ngm4/) |
| DSF-Aug training and rotation augmentation | [research/closures/dsf_aug/](research/closures/dsf_aug/) |
| DSF-Sym rotation and flux averaging | [research/closures/dsf_sym/](research/closures/dsf_sym/) |
| DSF-Trend and DSF-Hold stage solver and head history | [research/closures/dsf_trend_hold/](research/closures/dsf_trend_hold/) |
| Shared DSF network and physical decoder | [research/closures/dsf_shared/](research/closures/dsf_shared/) |
| DNS generation, FDNS and reference reassessment | [research/reference/](research/reference/) |
| Decaying flows, IC105, Forced-01 and ablations | [research/experiments/](research/experiments/) |
| A priori, autonomous and mechanism diagnostics | [research/evaluation/](research/evaluation/) |
| Grid, Fourier operators and LES integration | [numerics/](numerics/) |

## Render the paper figures

```sh
python -m pip install -r requirements/plotting.txt
python paper/figures/render.py --framework-only --output reproduced/framework
python paper/figures/render.py --paper-inputs /path/to/paper-results --output reproduced/paper
```

The method diagram uses code only. The remaining figures read the external paper archive containing `data/`, `tables/` and `audit/`. The renderer exports 11 main figures and seven supplementary figures as PDF and PNG.

## Check and run the numerical code

```sh
python -m pip install -r requirements/solver.txt
python tools/check_inputs.py
python reproduce.py test
python reproduce.py list
```

`reproduce.py` starts the numerical scripts from a generated runtime directory. This preserves the original file identities recorded by the archived experiments. See the [reproduction guide](docs/reproduction.md) for commands and external inputs, and the [file crosswalk](docs/source_layout.json) for the original and current paths.

Quantitative figures require the external results archive. Numerical reruns additionally require the relevant frozen configurations, checkpoints and input receipts.

## Release

The source project provided a licence placeholder. Code licence selection remains with the authors.
