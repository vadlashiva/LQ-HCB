"""Builds every results table (paper/tabs/*.tex), the number macros (paper/numbers.tex) and
the figures (paper/figs/*.pdf) from the raw results in res/. Run after campaign.sh."""
import json, os, math
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'res'); PAP = os.environ.get('PAPER_DIR', os.path.join(HERE, 'paper'))
RMAX = 28800
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "mathtext.fontset": "dejavuserif",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
# Okabe-Ito
COL = {'LQ-HCB': '#D55E00', 'LQ-HCB-T': '#CC79A7', 'EECHS-CB+G': '#0072B2', 'HCB-XL+G': '#009E73',
       'LEACH-CB+G': '#E69F00', 'EECHS-CB': '#0072B2', 'HCB-XL': '#009E73', 'LEACH-CB': '#E69F00',
       'DEEC-CB': '#999999'}
LS = {p: ('--' if not p.endswith('+G') and not p.startswith('LQ') else '-') for p in COL}


def load(f):
    p = os.path.join(RES, f)
    return json.load(open(p)) if os.path.exists(p) else []


def grp(R, proto):
    return [x for x in R if x['proto'] == proto]


def ci(v):
    v = np.asarray(v, float); n = len(v)
    if n < 2: return float(v.mean()), 0.0
    return float(v.mean()), float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n))


def val(xs, k, scale=1.0):
    return np.array([x[k] for x in xs], float) * scale


def fmt(m, h, d=0, sep=True):
    f = f"{{:,.{d}f}}" if sep else f"{{:.{d}f}}"
    return f"{f.format(m)} $\\pm$ {f.format(h)}"


def fnd_str(xs):
    v = val(xs, 'FND'); m, h = ci(v)
    if (v >= RMAX).all(): return f"$>${RMAX:,}"
    s = fmt(m, h)
    return s + ("$^\\dagger$" if (v >= RMAX).any() else "")


NUM = {}
def sg(x, d): return ('$+$' if x >= 0 else '$-$') + f'{abs(x):.{d}f}'
def pstr(p): return '$p<0.001$' if p < 0.001 else f'$p={p:.2f}$' if p >= 0.01 else f'$p={p:.3f}$'
def num(key, value): NUM[key] = value


# ---------------------------------------------------------------- S1 main table
S1 = load('S1.json')
MAINP = ['LEACH-CB', 'DEEC-CB', 'EECHS-CB', 'HCB-XL', 'LEACH-CB+G', 'EECHS-CB+G', 'HCB-XL+G', 'LQ-HCB', 'LQ-HCB-T']
NAME = {'LEACH-CB': 'LEACH-CB', 'DEEC-CB': 'DEEC-CB', 'EECHS-CB': 'EECHS-CB~\\cite{ren2020}', 'HCB-XL': 'HCB-XL',
        'LEACH-CB+G': 'LEACH-CB+G', 'EECHS-CB+G': 'EECHS-CB+G', 'HCB-XL+G': 'HCB-XL+G',
        'LQ-HCB': '\\textbf{LQ-HCB}', 'LQ-HCB-T': '\\textbf{LQ-HCB-T}',
        'LQ-HCB -R1': '\\quad without Rule 1 (sense $F$ every round)', 'LQ-HCB -R2': '\\quad without Rule 2 (flush every round)',
        'LQ-HCB -R3': '\\quad without Rule 3 (random spaced CHs)', 'LQ-HCB -FC': '\\quad without harvest forecast in bid',
        'LQ-HCB -R1R2R3': '\\quad without Rules 1--3 (reserve only)', 'LQ-HCB RF-only': '\\quad RF harvesting only',
        'LQ-HCB PV-only': '\\quad solar harvesting only'}


def row_main(xs):
    dead = ci(val(xs, 'deaths')); dl = ci(val(xs, 'deliv', 1e-6)); ov = ci(val(xs, 'overflow', 1e-6))
    dy = ci(val(xs, 'delay')); ee = ci(val(xs, 'EEc')); jn = ci(val(xs, 'jain')); ms = ci(val(xs, 'msgs'))
    return [fnd_str(xs), fmt(*dead, 1), fmt(*dl, 2), f"{ov[0]:.2f}", fmt(*dy, 0), f"{ee[0]:,.0f}", f"{jn[0]:.3f}", f"{ms[0]:.1f}"]


if S1:
    with open(f'{PAP}/tabs/main.tex', 'w') as f:
        f.write("\\begin{table}[!htbp]\n\\centering\n\\caption{Results in the shaded scenario S1 (mean $\\pm$ 95\\% confidence interval over "
                f"{len(grp(S1,'LQ-HCB'))} seeds; 20 days, $B=20$). FND: round of first node death ($^\\dagger$: some runs reach the "
                "horizon, so the mean is a lower bound). Dead: nodes dead after 20 days. Delivered and discarded: millions of readings. "
                "Delay: mean time from sensing to delivery. EE$_c$: delivered bits per joule of communication energy. "
                "Jain: fairness index of per-node delivered readings. Ctrl: control messages per round.}\n\\label{tab:main}\n"
                "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n"
                "Protocol & FND & Dead & Delivered & Discarded & Delay (s) & EE$_c$ (kbit/J) & Jain & Ctrl \\\\\n\\midrule\n")
        for i, p in enumerate(MAINP):
            xs = grp(S1, p)
            if not xs: continue
            if p == 'LEACH-CB+G': f.write("\\midrule\n\\multicolumn{9}{@{}l}{\\emph{Baselines with the LQ-HCB energy reserve}}\\\\\n")
            if p == 'LQ-HCB': f.write("\\midrule\n")
            if p == 'LEACH-CB': f.write("\\multicolumn{9}{@{}l}{\\emph{Baselines as published}}\\\\\n")
            f.write(NAME[p] + " & " + " & ".join(row_main(xs)) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")

    # ablation table
    ABL = ['LQ-HCB', 'LQ-HCB -R1', 'LQ-HCB -R2', 'LQ-HCB -R3', 'LQ-HCB -FC', 'LQ-HCB -R1R2R3', 'LQ-HCB RF-only', 'LQ-HCB PV-only']
    with open(f'{PAP}/tabs/ablation.tex', 'w') as f:
        f.write("\\begin{table}[!htbp]\n\\centering\n\\caption{Ablation in scenario S1 (mean over "
                f"{len(grp(S1,'LQ-HCB'))} seeds). Reserve overrides: transmissions that the drift-plus-penalty rule wanted but the "
                "energy reserve blocked, in thousands per run.}\n\\label{tab:ablation}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
                "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
                "Variant & Dead & Delivered (M) & Discarded (M) & Delay (s) & Reserve overrides (k) \\\\\n\\midrule\n")
        for p in ABL:
            xs = grp(S1, p)
            if not xs: continue
            if p == 'LQ-HCB RF-only': f.write("\\midrule\n")
            nm = NAME[p] if p != 'LQ-HCB' else 'LQ-HCB (all rules, RF + solar)'
            f.write(f"{nm} & {fmt(*ci(val(xs,'deaths')),1)} & {fmt(*ci(val(xs,'deliv',1e-6)),2)} & "
                    f"{ci(val(xs,'overflow',1e-6))[0]:.2f} & {ci(val(xs,'delay'))[0]:.0f} & {ci(val(xs,'guard_bind',1e-3))[0]:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")

    # numbers for the text
    def m_(p, k, s=1.0): return ci(val(grp(S1, p), k, s))[0]
    lq = grp(S1, 'LQ-HCB')
    for p, tag in [('LQ-HCB', 'LQ'), ('LQ-HCB-T', 'LQT'), ('EECHS-CB+G', 'EG'), ('HCB-XL+G', 'XG'), ('LEACH-CB+G', 'LG'),
                   ('EECHS-CB', 'E'), ('HCB-XL', 'X'), ('LEACH-CB', 'L'), ('DEEC-CB', 'D'),
                   ('LQ-HCB -R1', 'LQnoA'), ('LQ-HCB -R2', 'LQnoB'), ('LQ-HCB -R3', 'LQnoC'), ('LQ-HCB -FC', 'LQnoF'),
                   ('LQ-HCB -R1R2R3', 'LQnone'), ('LQ-HCB RF-only', 'LQrf'), ('LQ-HCB PV-only', 'LQpv')]:
        if not grp(S1, p): continue
        num(f'sA{tag}Del', f"{m_(p,'deliv',1e-6):.2f}"); num(f'sA{tag}Dead', f"{m_(p,'deaths'):.1f}")
        num(f'sA{tag}Delay', f"{m_(p,'delay'):.0f}"); num(f'sA{tag}Disc', f"{m_(p,'overflow',1e-6):.2f}")
        num(f'sA{tag}EEc', f"{m_(p,'EEc'):,.0f}"); num(f'sA{tag}EE', f"{m_(p,'EE'):,.0f}"); num(f'sA{tag}Jain', f"{m_(p,'jain'):.3f}")
        num(f'sA{tag}Msgs', f"{m_(p,'msgs'):.1f}"); num(f'sA{tag}FND', f"{m_(p,'FND'):,.0f}")
        num(f'sA{tag}Sensed', f"{m_(p,'sensed',1e-6):.2f}"); num(f'sA{tag}Lost', f"{m_(p,'lost'):,.0f}")
        num(f'sA{tag}Qmax', f"{max(val(grp(S1,p),'qmax')):.0f}"); num(f'sA{tag}Nch', f"{m_(p,'nch'):.1f}")
        num(f'sA{tag}Bind', f"{m_(p,'guard_bind',1e-3):.0f}")
        num(f'sA{tag}Util', f"{m_(p,'util'):.3f}")
    num('seedsMain', str(len(lq)))
    num('hybridGainPV', f"{(m_('LQ-HCB','deliv')/m_('LQ-HCB PV-only','deliv')-1)*100:.0f}")
    num('hybridGainRF', f"{(m_('LQ-HCB','deliv')/m_('LQ-HCB RF-only','deliv')-1)*100:.0f}")
    num('sALQmaxDeadRun', f"{max(val(lq,'deaths')):.0f}")

    # paired comparisons of LQ-HCB with each baseline
    def paired(a, b, k, s=1.0):
        A = val(grp(S1, a), k, s); B = val(grp(S1, b), k, s)
        sa = [x['seed'] for x in grp(S1, a)]; sb = [x['seed'] for x in grp(S1, b)]
        assert sa == sb
        md, hd = ci(A - B); mb = B.mean()
        t = stats.ttest_rel(A, B)
        return (100 * md / mb, 100 * hd / mb), t.pvalue, (md, hd)
    cmp = {}
    for b in ['EECHS-CB+G', 'HCB-XL+G', 'LEACH-CB+G', 'HCB-XL', 'EECHS-CB']:
        for a, ta in [('LQ-HCB', 'LQ'), ('LQ-HCB-T', 'LQT')]:
            if not grp(S1, b) or not grp(S1, a): continue
            (dm, dh), p, _ = paired(a, b, 'deliv')
            tg = {'EECHS-CB+G': 'EG', 'HCB-XL+G': 'XG', 'LEACH-CB+G': 'LG', 'HCB-XL': 'X', 'EECHS-CB': 'E'}[b]
            num(f'cmp{ta}{tg}Del', f"{abs(dm):.1f}"); num(f'cmp{ta}{tg}DelS', sg(dm, 1)); num(f'cmp{ta}{tg}DelCI', f"{dh:.1f}"); num(f'cmp{ta}{tg}DelP', pstr(p))
            (_, _), p2, (ddm, ddh) = paired(a, b, 'delay')
            num(f'cmp{ta}{tg}Delay', f"{abs(ddm):.0f}"); num(f'cmp{ta}{tg}DelayS', sg(ddm, 0)); num(f'cmp{ta}{tg}DelayP', pstr(p2))
            cmp[(a, b)] = (dm, dh, p, ddm, p2)
    json.dump({f"{k[0]} vs {k[1]}": v for k, v in cmp.items()}, open(f'{RES}/paired_S1.json', 'w'), indent=1)

# ---------------------------------------------------------------- scenarios S2, S3
SC = {'S2': load('S2.json'), 'S3': load('S3.json')}
SCP = ['LEACH-CB', 'EECHS-CB', 'HCB-XL', 'LEACH-CB+G', 'EECHS-CB+G', 'HCB-XL+G', 'LQ-HCB', 'LQ-HCB-T']
if SC['S2'] and SC['S3']:
    with open(f'{PAP}/tabs/scen.tex', 'w') as f:
        n2 = len(grp(SC['S2'], 'LQ-HCB')); n3 = len(grp(SC['S3'], 'LQ-HCB'))
        f.write("\\begin{table}[!htbp]\n\\centering\n\\caption{Results in the overcast-spell scenario S2 and the 200-node, four-beacon "
                f"scenario S3 (mean over {n2} and {n3} seeds; 20 days). Dead: nodes dead after 20 days (out of 100 in S2 and 200 in S3). "
                "Delivered: millions of readings.}\n\\label{tab:scen}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
                "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lrrrcrrr@{}}\n\\toprule\n"
                " & \\multicolumn{3}{c}{S2: overcast spells} && \\multicolumn{3}{c}{S3: 200 nodes, 4 beacons} \\\\\n"
                "\\cmidrule{2-4}\\cmidrule{6-8}\nProtocol & Dead & Delivered & Delay (s) && Dead & Delivered & Delay (s) \\\\\n\\midrule\n")
        for p in SCP:
            a = grp(SC['S2'], p); b = grp(SC['S3'], p)
            if p == 'LEACH-CB+G' or p == 'LQ-HCB': f.write("\\midrule\n")
            f.write(f"{NAME[p]} & {ci(val(a,'deaths'))[0]:.1f} & {ci(val(a,'deliv',1e-6))[0]:.2f} & {ci(val(a,'delay'))[0]:.0f} && "
                    f"{ci(val(b,'deaths'))[0]:.1f} & {ci(val(b,'deliv',1e-6))[0]:.2f} & {ci(val(b,'delay'))[0]:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    for s, R in SC.items():
        sk = {'S2': 'sB', 'S3': 'sC'}[s]
        for p, tag in [('LQ-HCB', 'LQ'), ('LQ-HCB-T', 'LQT'), ('EECHS-CB+G', 'EG'), ('HCB-XL+G', 'XG'), ('LEACH-CB+G', 'LG'),
                       ('EECHS-CB', 'E'), ('HCB-XL', 'X'), ('LEACH-CB', 'L'), ('LQ-HCB -FC', 'LQnoF')]:
            xs = grp(R, p)
            if not xs: continue
            num(f'{sk}{tag}Del', f"{ci(val(xs,'deliv',1e-6))[0]:.2f}"); num(f'{sk}{tag}Dead', f"{ci(val(xs,'deaths'))[0]:.1f}")
            num(f'{sk}{tag}Delay', f"{ci(val(xs,'delay'))[0]:.0f}"); num(f'{sk}{tag}Disc', f"{ci(val(xs,'overflow',1e-6))[0]:.2f}")
            num(f'{sk}{tag}Bind', f"{ci(val(xs,'guard_bind',1e-3))[0]:.0f}"); num(f'{sk}{tag}Nch', f"{ci(val(xs,'nch'))[0]:.1f}")
            num(f'{sk}{tag}MaxDead', f"{max(val(xs,'deaths')):.0f}")
        num(f'{sk}seeds', str(len(grp(R, 'LQ-HCB'))))
        for a, ta in [('LQ-HCB', 'LQ'), ('LQ-HCB-T', 'LQT')]:
            for b, tb in [('EECHS-CB+G', 'EG'), ('HCB-XL+G', 'XG'), ('LEACH-CB+G', 'LG'), ('HCB-XL', 'X'), ('EECHS-CB', 'E')]:
                A = val(grp(R, a), 'deliv'); Bv = val(grp(R, b), 'deliv')
                if len(A) and len(A) == len(Bv):
                    mm, hh = ci(A - Bv); mm, hh = 100 * mm / Bv.mean(), 100 * hh / Bv.mean()
                    num(f'cmp{sk.upper()}{ta}{tb}', f"{abs(mm):.0f}"); num(f'cmp{sk.upper()}{ta}{tb}CI', f"{hh:.0f}")
                    num(f'cmp{sk.upper()}{ta}{tb}P', pstr(stats.ttest_rel(A, Bv).pvalue))

# ---------------------------------------------------------------- buffer sweep / trade-off
BS = {}
for B in (10, 20, 40, 80):
    R = load(f'B{B}.json') if B != 20 else [x for x in S1 if x['seed'] < 10]
    if R: BS[B] = R
SWP = ['LEACH-CB+G', 'HCB-XL+G', 'EECHS-CB+G', 'LQ-HCB-T', 'LQ-HCB']
if len(BS) == 4:
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for p in SWP:
        xs = [ci(val(grp(BS[B], p), 'delay'))[0] for B in BS]
        ys = [ci(val(grp(BS[B], p), 'deliv', 1e-6))[0] for B in BS]
        ds = [ci(val(grp(BS[B], p), 'deaths'))[0] for B in BS]
        lw = 1.8 if p.startswith('LQ') else 1.1
        ax[0].plot(xs, ys, marker='o', ms=3.5, lw=lw, color=COL[p], label=p)
        ax[1].plot(list(BS), ds, marker='o', ms=3.5, lw=lw, color=COL[p], label=p)
        for B, x, y in zip(BS, xs, ys):
            if p == 'LQ-HCB-T' and B > 10:
                ax[0].annotate(f"$B$={B}", (x, y), textcoords='offset points', xytext=(-6, 5), fontsize=6.5, color=COL[p])
            if p == 'LQ-HCB' and B == 10:
                ax[0].annotate("$B$=10 (both LQ-HCB variants)", (x, y), textcoords='offset points', xytext=(6, 2), fontsize=6.5, color=COL[p])
    ax[0].set_xlabel('Mean delay (s)'); ax[0].set_ylabel('Delivered readings (millions)'); ax[0].set_title('(a) Throughput versus delay', fontsize=8.5)
    ax[1].set_xlabel('Buffer capacity $B$ (readings)'); ax[1].set_ylabel('Nodes dead after 20 days'); ax[1].set_xscale('log', base=2)
    ax[1].set_xticks(list(BS)); ax[1].set_xticklabels([str(b) for b in BS]); ax[1].set_title('(b) Node deaths', fontsize=8.5)
    ax[0].grid(alpha=.25); ax[1].grid(alpha=.25)
    ax[1].legend(fontsize=6.8, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout(); plt.savefig(f'{PAP}/figs/tradeoff.pdf'); plt.close()
    for B in BS:
        for p, tag in [('LQ-HCB', 'LQ'), ('LQ-HCB-T', 'LQT'), ('EECHS-CB+G', 'EG')]:
            xs = grp(BS[B], p)
            num(f'bw{tag}{"ABCD"[list(BS).index(B)]}Del', f"{ci(val(xs,'deliv',1e-6))[0]:.2f}")
            num(f'bw{tag}{"ABCD"[list(BS).index(B)]}Delay', f"{ci(val(xs,'delay'))[0]:.0f}")
            num(f'bw{tag}{"ABCD"[list(BS).index(B)]}Util', f"{ci(val(xs,'util'))[0]:.3f}")
            num(f'bw{tag}{"ABCD"[list(BS).index(B)]}Dead', f"{ci(val(xs,'deaths'))[0]:.1f}")
            num(f'bw{tag}{"ABCD"[list(BS).index(B)]}Qmax', f"{max(val(xs,'qmax')):.0f}")

# ---------------------------------------------------------------- sensitivity
SENS = [('kmult', 'kap{}', [0.25, 0.5, 1, 2, 4], '$\\kappa/\\kappa^\\star$'), ('K', 'K{}', [1, 5, 10, 20, 50], '$K$ (rounds)'),
        ('Rc', 'Rc{}', [20, 27, 35], '$R_c$ (m)')]
rows = []
base10 = [x for x in S1 if x['proto'] == 'LQ-HCB' and x['seed'] < 10]
for key, pat, vals, lab in SENS:
    for v in vals:
        is_def = (key == 'kmult' and v == 1) or (key == 'K' and v == 10) or (key == 'Rc' and v == 27)
        xs = base10 if is_def else load(pat.format(v if not isinstance(v, float) or v not in (1.0,) else int(v)) + '.json')
        if key == 'kmult' and not is_def: xs = load(f"kap{v}.json")
        if not xs: continue
        rows.append((lab, v, is_def, xs))
H4 = load('H400.json')
if rows:
    with open(f'{PAP}/tabs/sens.tex', 'w') as f:
        f.write("\\begin{table}[!htbp]\n\\centering\n\\caption{Sensitivity of LQ-HCB in scenario S1 (10 seeds; one parameter varied at a time, "
                "defaults in bold).}\n\\label{tab:sens}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
                "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n"
                "Parameter & Value & Dead & Delivered (M) & Delay (s) & Reserve overrides (k) & Ctrl msgs/round \\\\\n\\midrule\n")
        last = None
        for lab, v, d, xs in rows:
            if last is not None and lab != last: f.write("\\midrule\n")
            vs = f"\\textbf{{{v:g}}}" if d else f"{v:g}"
            f.write(f"{lab if lab != last else ''} & {vs} & {ci(val(xs,'deaths'))[0]:.1f} & {ci(val(xs,'deliv',1e-6))[0]:.2f} & "
                    f"{ci(val(xs,'delay'))[0]:.0f} & {ci(val(xs,'guard_bind',1e-3))[0]:.1f} & {ci(val(xs,'msgs'))[0]:.1f} \\\\\n")
            last = lab
        f.write("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    for lab, v, d, xs in rows:
        k = {'$\\kappa/\\kappa^\\star$': 'kap', '$K$ (rounds)': 'K', '$R_c$ (m)': 'Rc'}[lab]
        tag = f"{k}{str(v).replace('.', 'p')}"
        tag = ''.join(ch if ch.isalpha() else 'ABCDEFGHIJ'[int(ch)] if ch.isdigit() else '' for ch in tag)
        num(f'sens{tag}Del', f"{ci(val(xs,'deliv',1e-6))[0]:.2f}"); num(f'sens{tag}Dead', f"{ci(val(xs,'deaths'))[0]:.1f}")
        num(f'sens{tag}Delay', f"{ci(val(xs,'delay'))[0]:.0f}"); num(f'sens{tag}Bind', f"{ci(val(xs,'guard_bind',1e-3))[0]:.1f}")
        num(f'sens{tag}Msgs', f"{ci(val(xs,'msgs'))[0]:.1f}")
if H4:
    for p, tag in [('LQ-HCB', 'LQ'), ('LQ-HCB-T', 'LQT'), ('EECHS-CB+G', 'EG'), ('HCB-XL+G', 'XG'), ('LEACH-CB+G', 'LG'), ('HCB-XL', 'X')]:
        xs = grp(H4, p)
        if xs:
            num(f'hh{tag}Del', f"{ci(val(xs,'deliv',1e-6))[0]:.2f}"); num(f'hh{tag}Dead', f"{ci(val(xs,'deaths'))[0]:.1f}")
            num(f'hh{tag}Delay', f"{ci(val(xs,'delay'))[0]:.0f}")

# ---------------------------------------------------------------- S3 with larger packing radius
for rc, tg in [(35, 'A'), (40, 'B')]:
    xs = load(f'S3_Rc{rc}.json')
    if xs and SC.get('S3'):
        eg = grp(SC['S3'], 'EECHS-CB+G')
        A = val(xs, 'deliv'); Bv = val(eg, 'deliv')
        num(f'rc{tg}Del', f"{A.mean()/1e6:.2f}"); num(f'rc{tg}Dead', f"{val(xs,'deaths').mean():.1f}")
        num(f'rc{tg}DeadRuns', str(int((val(xs,'deaths') > 0).sum()))); num(f'rc{tg}Nch', f"{val(xs,'nch').mean():.1f}")
        num(f'rc{tg}Delay', f"{val(xs,'delay').mean():.0f}")
        d = 100 * (A - Bv).mean() / Bv.mean(); num(f'rc{tg}VsEG', f"{abs(d):.1f}"); num(f'rc{tg}VsEGS', sg(d, 1))
        num(f'rc{tg}VsEGP', pstr(stats.ttest_rel(A, Bv).pvalue))
if H4:
    for p, tag in [('LQ-HCB', 'LQ'), ('LQ-HCB-T', 'LQT')]:
        A = val(grp(H4, p), 'deliv'); Bv = val(grp(H4, 'EECHS-CB+G'), 'deliv')
        d = 100 * (A - Bv).mean() / Bv.mean(); num(f'hh{tag}VsEG', f"{abs(d):.1f}"); num(f'hh{tag}VsEGP', pstr(stats.ttest_rel(A, Bv).pvalue))
        num(f'hh{tag}DeadRuns', str(int((val(grp(H4, p), 'deaths') > 0).sum())))

# ---------------------------------------------------------------- alive curves
if S1 and SC.get('S2'):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for k, (R, ttl) in enumerate([(S1, '(a) S1: shaded field'), (SC['S2'], '(b) S2: overcast spells')]):
        for p in ['LEACH-CB', 'EECHS-CB', 'HCB-XL', 'LEACH-CB+G', 'EECHS-CB+G', 'HCB-XL+G', 'LQ-HCB', 'LQ-HCB-T']:
            xs = grp(R, p)
            if not xs: continue
            A = np.array([x['alive'] for x in xs], float).mean(0)
            t = np.arange(len(A)) * 60 * 60 / 86400  # every 60 rounds of 60 s
            ax[k].plot(t, A, color=COL[p], ls=LS[p], lw=1.8 if p.startswith('LQ') else 1.0, label=p)
        for d in range(20): ax[k].axvspan(d + 0.5, d + 1.0, color='0.92', lw=0, zorder=0)
        ax[k].set_xlim(0, 20); ax[k].set_xlabel('Time (days)'); ax[k].set_title(ttl, fontsize=8.5)
    ax[0].set_ylabel('Alive nodes'); ax[1].legend(fontsize=6.5, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout(); plt.savefig(f'{PAP}/figs/alive.pdf'); plt.close()

# ---------------------------------------------------------------- node trace + batching thresholds
TR = load('trace.json')
cn = json.load(open(f'{RES}/closed_nums.json'))
if TR:
    import lqhcb as m
    q = dict(m.P); kap = float(TR[0]['kappa'])
    d0 = math.sqrt(q['efs'] / q['emp'])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1, 1.7]))
    E = np.linspace(0, q['Emax'], 400)
    for d, c in [(10, '#56B4E9'), (20, '#0072B2'), (40, '#009E73'), (60, '#E69F00')]:
        amp = q['efs'] * d**2 if d < d0 else q['emp'] * d**4
        c0 = q['Est'] + q['H'] * (q['Eelec'] + amp); c1 = q['L'] * (q['Eelec'] + amp)
        x = kap * (q['Emax'] - E)
        ns = 0.5 * (x * c1 + np.sqrt((x * c1)**2 + 4 * x * c0))
        ax[0].plot(E, np.minimum(ns, 40), color=c, lw=1.2, label=f'$d$ = {d} m')
    ax[0].axhline(q['Bcap'], color='0.4', ls=':', lw=0.9); ax[0].text(0.05, q['Bcap'] + 0.6, '$B=Q_{\\max}=20$', fontsize=6.5, color='0.3')
    ax[0].set_ylim(0, 30); ax[0].set_xlim(0, 5)
    ax[0].set_xlabel('Stored energy $E_i$ (J)'); ax[0].set_ylabel('Flush threshold $n^\\star$ (readings)')
    ax[0].set_title('(a) Batch size needed to transmit', fontsize=8.5); ax[0].legend(fontsize=6.5, frameon=False)
    lq = [x for x in TR if x['proto'] == 'LQ-HCB'][0]; eg = [x for x in TR if x['proto'] == 'EECHS-CB+G'][0]
    nr = 4 * 1440
    t = np.arange(nr) * 60 / 86400
    a2 = ax[1]; a3 = a2.twinx()
    a2.plot(t, np.array(lq['trace']['E'][:nr]), color=COL['LQ-HCB'], lw=1.1, label='LQ-HCB: energy')
    a2.plot(t, np.array(eg['trace']['E'][:nr]), color=COL['EECHS-CB+G'], lw=0.9, alpha=.8, label='EECHS-CB+G: energy')
    a3.plot(t, np.array(lq['trace']['Q'][:nr]), color='0.35', lw=0.5, alpha=.7, label='LQ-HCB: backlog')
    for d in range(4): a2.axvspan(d + 0.5, d + 1.0, color='0.92', lw=0, zorder=0)
    a2.set_xlabel('Time (days)'); a2.set_ylabel('Stored energy (J)'); a3.set_ylabel('LQ-HCB backlog $Q_i$ (readings)', color='0.35')
    a3.set_ylim(0, 60); a3.spines['right'].set_visible(True); a2.set_xlim(0, 4)
    a2.set_title(f"(b) Node {lq['trace']['node']} (median distance to BS), seed 0", fontsize=8.5)
    h1, l1 = a2.get_legend_handles_labels(); h2, l2 = a3.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, fontsize=6.3, frameon=False, loc='upper right')
    plt.tight_layout(); plt.savefig(f'{PAP}/figs/node.pdf'); plt.close()

# ---------------------------------------------------------------- numbers.tex
for k, v in cn.items(): num(k, v)
import lqhcb as m
num('Rmax', f"{m.P['Rmax']:,}")
with open(f'{PAP}/numbers.tex', 'w') as f:
    for k, v in sorted(NUM.items()):
        assert k.isalpha(), k
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(NUM), 'numbers written')
