"""Table 2 (closed forms vs Monte Carlo) and the analytical numbers quoted in Secs. 3 and 5."""
import math, json, sys, os
import numpy as np
import lqhcb as m

q = dict(m.P)
lam = 3e8 / q['f']; k = q['Pt'] * q['Gt'] * q['Gr'] * lam**2 / (4 * math.pi)**2
ps, eta = q['rf_sens'], q['eta_rf']
Pidle = q['Eidle'] / q['Tr']; Tn = (24 - q['day_h']) * 3600


def erf_mean(d):
    Pb = k / d**2
    return eta * (Pb + ps) * math.exp(-ps / Pb)


g = np.random.default_rng(1)
rows = []
for d in (15.0, 60.0):
    Pb = k / d**2
    x = Pb * g.exponential(1.0, 2_000_000)
    mc = np.where(x >= ps, eta * x, 0).mean()
    rows.append((f"$\\mathbb{{E}}[P_{{\\mathrm{{rf,DC}}}}]$ at \\SI{{{d:.0f}}}{{m}} (Prop.~\\ref{{prop:rf}})",
                 f"\\SI{{{erf_mean(d)*1e6:.3f}}}{{\\micro W}}", f"\\SI{{{mc*1e6:.3f}}}{{\\micro W}}"))
    out_p = 1 - math.exp(-ps / Pb); out_mc = (x < ps).mean()
    if d == 60.0:
        rows.append((f"Rectifier outage at \\SI{{{d:.0f}}}{{m}} (Prop.~\\ref{{prop:rf}})", f"{out_p:.3f}", f"{out_mc:.3f}"))
# Prop 2: daily PV energy, unshaded node, i.i.d. clouds, from the simulator environment
pv_an = q['eta_pv'] * q['A'] * (1 - q['cloud_max'] / 2) * 2 / math.pi * q['Gmax'] * q['day_h'] * 3600
env = m.environment(dict(q, **m.SCEN['S1']), 0)
days = q['Rmax'] * q['Tr'] / 86400
pv_sim = (env['epv'] / env['shade'][None]).sum(0).mean() / days
rows.append(("Daily PV energy, $s_i=1$ (Prop.~\\ref{prop:pv})", f"\\SI{{{pv_an:.3f}}}{{J}}",
             f"\\SI{{{pv_sim:.3f}}}{{J}} (20-day mean)"))
# reserve radius: E[P_rf,DC](d) = P_idle
lo, hi = 1.0, 200.0
for _ in range(100):
    mid = (lo + hi) / 2
    if erf_mean(mid) > Pidle: lo = mid
    else: hi = mid
rres = lo
dmax = math.hypot(50, 50)
res_max = max(q['Emin_res'], (Pidle - erf_mean(dmax)) * Tn)
res_nobeacon = Pidle * Tn
rows.append((f"Distance at which $\\mathbb{{E}}[P_{{\\mathrm{{rf,DC}}}}]=P_{{\\mathrm{{idle}}}}$", f"\\SI{{{rres:.1f}}}{{m}}", "--"))
rows.append((f"Largest reserve $\\Eres_{{\\max}}$ (Eq.~\\ref{{eq:reserve}}, $d=\\SI{{{dmax:.1f}}}{{m}}$)", f"\\SI{{{res_max:.3f}}}{{J}}",
             f"\\SI{{{max(env['Erf_mean'].min()*0, 0) + max(q['Emin_res'], (Pidle - env['Erf_mean'].min())*Tn):.3f}}}{{J}} (farthest node, seed 0)"))

with open(os.path.join(os.environ.get('PAPER_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper')), 'tabs', 'closed.tex'), 'w') as f:
    f.write("\\begin{table}[!htbp]\n\\centering\n\\caption{Closed-form results versus Monte Carlo simulation (parameters of \\cref{tab:params}; "
            "$2\\times10^6$ fading samples).}\n\\label{tab:closed}\n\\small\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lll@{}}\n\\toprule\n"
            "Quantity & Closed form & Simulation \\\\\n\\midrule\n")
    for a, b, c in rows: f.write(f"{a} & {b} & {c} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")

# Prop. 4 thresholds at d = 20 m with the default kappa*
Bcap = q['Bcap']; F = q['F']; L = q['L']; H = q['H']
d0 = math.sqrt(q['efs'] / q['emp'])
amp = lambda d: q['efs'] * d**2 if d < d0 else q['emp'] * d**4
def epkt(n, d): return math.ceil(n / q['npkt']) * q['Est'] + (math.ceil(n / q['npkt']) * H + n * L) * (q['Eelec'] + amp(d))
cmax = epkt(Bcap, dmax)
kap = Bcap / (L * q['Eelec'] * (q['Emax'] - res_max - cmax))
def nstar(D, d):
    c0 = q['Est'] + H * (q['Eelec'] + amp(d)); c1 = L * (q['Eelec'] + amp(d))
    x = kap * D
    return 0.5 * (x * c1 + math.sqrt((x * c1)**2 + 4 * x * c0))
# deficit above which a node at 20 m never flushes (n* > Qmax)
lo, hi = 0.0, 5.0
for _ in range(100):
    mid = (lo + hi) / 2
    if nstar(mid, 20) > Bcap: hi = mid
    else: lo = mid
lo2, hi2 = 0.0, 5.0
for _ in range(100):
    mid = (lo2 + hi2) / 2
    if nstar(mid, 20) > Bcap // 2 + 1: hi2 = mid
    else: lo2 = mid
nums = dict(silentEeff=f"{q['Emax']-lo2:.1f}", resRadius=f"{rres:.0f}", resMax=f"{res_max:.2f}", resMaxPct=f"{100*res_max/q['Emax']:.1f}",
            nstarOne=f"{math.ceil(nstar(1.0, 20))}", nstarThree=f"{math.ceil(nstar(3.0, 20))}",
            kappaStar=f"${kap/1e4:.2f}\\times10^{{4}}$", silentE=f"{q['Emax']-lo:.2f}", pvDay=f"{pv_an:.2f}")
json.dump(nums, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res', 'closed_nums.json'), 'w'), indent=1)
print(nums); print(rows)
