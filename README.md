# LQ-HCB: Lyapunov-based cross-layer clustering and buffer control for hybrid RF–solar WSNs

Simulation code and raw results for the paper:

> Shiva Kumar V and Choupiri Shivakeshi, "Lyapunov-Based Cross-Layer Clustering and Buffer Control for Hybrid RF–Solar Wireless Sensor Networks," submitted to *Computer Networks* (Elsevier), 2026.

LQ-HCB treats each node's circular data buffer and energy store as two queues. Minimising one Lyapunov drift-plus-penalty bound yields three local rules: sensing admission (application layer), energy-adaptive flush (MAC) and deficit-weighted cluster-head election (data link).

## Contents
| File | Purpose |
|---|---|
| `lqhcb.py` | Unified simulator: environment (paired across protocols), LQ-HCB, its ablations, and the baselines LEACH-CB, DEEC-CB, EECHS-CB, HCB-XL and their reserve-augmented (+G) versions; scenarios S1–S3 |
| `hcbxl.py` | Physical, radio and harvesting parameters (unchanged copy of the HCB-XL code; imported read-only) |
| `campaign.sh`, `extra.sh` | All runs used in the paper |
| `closed_forms.py` | Table 2 (closed forms vs Monte Carlo) and the analytical numbers in Secs. 3 and 5 |
| `analyze.py` | All results tables, number macros (`numbers.tex`) and Figs. 3–5 |
| `res/*.json` | Raw results of every run reported in the paper |

## Reproducing
```bash
pip install numpy scipy matplotlib
bash campaign.sh && bash extra.sh          # about 1.5 h on 2 cores
PAPER_DIR=./paper python closed_forms.py
PAPER_DIR=./paper python analyze.py          # writes paper/tabs/*.tex, paper/numbers.tex, paper/figs/*.pdf
```
Seeds are fixed, so results are exactly reproducible. Each seed draws the deployment, shading, fading and weather from its own generator before any protocol runs, so all protocols see the same environment.

## License
MIT
