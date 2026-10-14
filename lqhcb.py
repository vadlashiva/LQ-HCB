"""LQ-HCB: Lyapunov queue-aware clustering with buffer control for hybrid RF-solar WSNs.

Unified simulator for the revised paper. All protocols run on the SAME environment
realisation per seed (deployment, shading, Rayleigh fading, cloud/weather), drawn from a
dedicated generator, so comparisons are fully paired. Protocol randomness uses a second
generator.

Physical, radio and buffer parameters are imported read-only from hcbxl.py (v1 code,
unchanged). Baseline logic (LEACH/DEEC/EECHS/HCB-XL) re-implements v1's run() with an
optional energy guard ("+G"); see paper Sec. 5.

LQ-HCB per round (paper Sec. 4):
  Rule 1  a_i = floor(min(F, max(0, V/Q_i - 1)))   (a_i = F if Q_i = 0)
  Rule 2  flush whole buffer iff Q_i^2 >= kappa |E~_i| e_pkt(Q_i, d_i)
  Rule 3  every K rounds: CHs = priority maximal R_c-packing in increasing bid
          b_j = |min(E_j + K*Ebar_h,j - theta, 0)| * e_CH,j  among feasible nodes
  with E~ = E - theta, theta = Emax, kappa = kappa* (Lemma 2), reserve E_res,i (Prop. 3).
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.environ.get("HCBXL_V1", os.path.dirname(os.path.abspath(__file__))))
import hcbxl as v1  # read-only

P = dict(v1.P)
P.update(Bcap=20, V=None, kmult=1.0, K=10, Rc=27.0, npkt=20, Emin_res=0.02, kappa=None,  # None -> kappa* (Lemma 2)
         night_res=1, beacons='bs', weather='iid', trace=0, util='log')

SCEN = {
    'S1': dict(),                                                    # v1 shaded scenario
    'S2': dict(Gmax=20.0, weather='markov'),                          # brighter sun, multi-day overcast spells
    'S3': dict(N=200, M=141.42, bs=(70.71, 70.71), beacons='quad'),   # larger network, 4 beacons
}

# name: dict(kind, guard, rf, pv, r1, r2, r3, fc)
def _b(kind, guard=0, **kw):
    d = dict(kind=kind, guard=guard, rf=1, pv=1, r1=1, r2=1, r3=1, fc=1); d.update(kw); return d
PROTO = {
    'LEACH-CB':        _b('leach'),
    'DEEC-CB':         _b('deec'),
    'EECHS-CB':        _b('eechs'),
    'HCB-XL':          _b('xl'),
    'LEACH-CB+G':      _b('leach', 1),
    'EECHS-CB+G':      _b('eechs', 1),
    'HCB-XL+G':        _b('xl', 1),
    'LQ-HCB':          _b('lq'),
    'LQ-HCB-T':        _b('lq', util='lin'),
    'LQ-HCB -R1':      _b('lq', r1=0),
    'LQ-HCB -R2':      _b('lq', r2=0),
    'LQ-HCB -R3':      _b('lq', r3=0),
    'LQ-HCB -FC':      _b('lq', fc=0),
    'LQ-HCB -R1R2R3':  _b('lq', r1=0, r2=0, r3=0, fc=0),
    'LQ-HCB RF-only':  _b('lq', pv=0),
    'LQ-HCB PV-only':  _b('lq', rf=0),
}


def environment(q, seed):
    """Deployment and per-round harvest matrices (J per round), identical for all protocols."""
    g = np.random.default_rng(seed)
    N, M, R, Tr = q['N'], q['M'], q['Rmax'], q['Tr']
    xy = g.uniform(0, M, (N, 2)); bs = np.array(q['bs'])
    dbs = np.linalg.norm(xy - bs, axis=1)
    lam = 3e8 / q['f']; k = q['Pt'] * q['Gt'] * q['Gr'] * lam**2 / (4*np.pi)**2
    if q['beacons'] == 'bs':
        bpos = bs[None]
    else:  # four beacons at the quarter points of the field
        bpos = np.array([[M/4, M/4], [3*M/4, M/4], [M/4, 3*M/4], [3*M/4, 3*M/4]])
    dB = np.linalg.norm(xy[:, None] - bpos[None], axis=2)              # N x nb
    pbar = k / np.maximum(dB, 1.0)**2                                   # mean power per beacon
    shade = g.uniform(q['shade_min'], 1.0, N)
    # RF: independent Rayleigh fading per beacon link, threshold rectifier on total incident power
    fad = g.exponential(1.0, (R, N, dB.shape[1])).astype(np.float32)
    Pin = (fad * pbar[None].astype(np.float32)).sum(2)
    erf = np.where(Pin >= q['rf_sens'], q['eta_rf'] * Pin, 0.0).astype(np.float32) * Tr
    # PV
    t_h = (np.arange(R) * Tr / 3600.0) % 24.0
    day = t_h < q['day_h']
    G = np.where(day, q['Gmax'] * np.sin(np.pi * np.minimum(t_h, q['day_h']) / q['day_h']), 0.0)
    if q['weather'] == 'iid':
        cloud = g.uniform(0, q['cloud_max'], R)
    else:  # day-level Markov chain: clear <-> overcast, persistent overcast spells
        nd = int(np.ceil(R * Tr / 86400.0)) + 1; st = np.zeros(nd, int); st[0] = 0
        u = g.random(nd)
        for d in range(1, nd):
            st[d] = (u[d] < 0.6) if st[d-1] == 1 else (u[d] < 0.25)
        dayidx = (np.arange(R) * Tr // 86400).astype(int)
        lo = np.where(st[dayidx] == 1, 0.7, 0.0); hi = np.where(st[dayidx] == 1, 0.95, 0.3)
        cloud = lo + (hi - lo) * g.random(R)
    epv = (q['eta_pv'] * q['A'] * G * (1 - cloud))[:, None] * shade[None] * Tr
    # expected RF DC power per node (Prop. 1 generalised: sum of exponentials -> use MC mean)
    Erf_mean = erf.mean(0) / Tr
    return dict(xy=xy, bs=bs, dbs=dbs, shade=shade, erf=erf, epv=epv.astype(np.float32),
                Erf_mean=Erf_mean, pbar=pbar, cloud=cloud)


def run(args):
    proto, seed, scen, over = args
    q = dict(P); q.update(SCEN[scen]); q.update(over)
    c = PROTO[proto]
    if 'util' in c: q['util'] = c['util']
    env = environment(q, seed)
    rng = np.random.default_rng(10_000 + seed)            # protocol randomness
    N, F, H, L, Tr = q['N'], q['F'], q['H'], q['L'], q['Tr']
    xy, dbs = env['xy'], env['dbs']
    d0 = np.sqrt(q['efs'] / q['emp'])
    amp = lambda d: np.where(d < d0, q['efs'] * d**2, q['emp'] * d**4)
    per = int(round(1 / q['p'])); npk = q['npkt']

    def e_pkt(n, d):                                     # Eq. (7) with n readings in ceil(n/npk) packets
        ntx = np.ceil(n / npk)
        return ntx * q['Est'] + (ntx * H + n * L) * (q['Eelec'] + amp(d))
    e_ch = ((per - 1) * F * (H + L) * q['Eelec'] + per * F * L * q['EDA']
            + q['Est'] + (H + F * L) * (q['Eelec'] + amp(dbs)))       # expected CH cost per round
    # night reserve from Props. 1 and 3 (idle consumption over the night minus expected RF)
    Tn = (24.0 - q['day_h']) * 3600.0
    Erf_mean = env['Erf_mean'] * (1 if c['rf'] else 0)
    Eres = np.maximum(q['Emin_res'], (q['Eidle'] / Tr - Erf_mean) * Tn) if q['night_res'] else np.full(N, q['Emin_res'])
    guard = c['guard'] or c['kind'] == 'lq'
    if not guard: Eres = np.zeros(N)

    Bcap = q['Bcap']                        # same buffer capacity for every protocol
    if q['V'] is not None: V = q['V']
    else: V = Bcap if q['util'] == 'log' else Bcap - F + 1   # largest V with Qmax <= Bcap
    Qmax = max(V, F) if q['util'] == 'log' else V + F - 1     # Theorem 1
    theta = q['Emax']
    cmax = float(e_pkt(max(Qmax, Bcap), dbs.max()))          # largest member flush cost
    kappa = q['kappa'] if q['kappa'] else q['kmult'] * Qmax / (L * q['Eelec'] * (theta - Eres.max() - cmax))   # Lemma 2
    Rc, K = q['Rc'], q['K']

    E = np.full(N, q['E0']); alive = np.ones(N, bool); Ebar = np.zeros(N)
    Qb = np.zeros(N)               # member buffers (readings)
    dnode = np.zeros(N)            # readings delivered per source node
    Gset = np.zeros(N, bool)       # LEACH epoch set
    ch = np.zeros(N, bool); lab = None; nclu = 0
    hist = []; tot_cons = 0.0; deliv = 0.0; sensed = 0.0; backlog = 0.0
    lost = 0.0; overflow = 0.0; util = 0.0; msgs = 0.0; nelect = 0; qmax_seen = 0.0
    nch_sum = 0.0; guard_bind = 0; idle_sum = 0.0
    tr = dict(E=[], Q=[], fl=[], ch=[], node=None) if q['trace'] else None

    for r in range(q['Rmax']):
        na = int(alive.sum()); hist.append(na)
        if na == 0: break
        eh = np.zeros(N)
        if c['rf']: eh += env['erf'][r]
        if c['pv']: eh += env['epv'][r]
        E = np.where(alive, np.minimum(q['Emax'], E + eh), E)
        Ebar = q['beta'] * eh + (1 - q['beta']) * Ebar

        # ------------------------------------------------ cluster-head election
        if c['kind'] == 'lq':
            need = (r % K == 0) or not (ch & alive).any() or (ch & alive & (E - e_ch < Eres)).any()
            if need:
                nelect += 1
                feas = alive & (E - K * e_ch >= Eres)
                if c['r3']:
                    Ep = E + (K * Ebar if c['fc'] else 0.0)
                    bid = np.abs(np.minimum(Ep - theta, 0.0)) * e_ch
                    order = np.lexsort((e_ch, bid))            # bid, ties -> cheaper CH
                else:
                    order = rng.permutation(N)
                ch = np.zeros(N, bool); chosen = []
                for j in order:
                    if not feas[j]: continue
                    if not chosen or np.min(np.hypot(*(xy[chosen] - xy[j]).T)) > Rc:
                        chosen.append(j)
                ch[chosen] = True
                msgs += feas.sum() + len(chosen) + (alive.sum() - len(chosen))   # bids + adverts + joins
        elif c['kind'] in ('leach', 'deec', 'xl', 'eechs'):
            if r % per == 0: Gset[:] = False
            T = q['p'] / (1 - q['p'] * (r % per))
            elig = alive & ~Gset
            if c['kind'] in ('leach', 'eechs'): wgt = np.ones(N)
            elif c['kind'] == 'deec': wgt = E / q['E0']
            else:
                hmax = max(Ebar[alive].max(), 1e-12); w1, w2, w3 = q['w']
                wgt = w1 * E / q['E0'] + w2 * Ebar / hmax + w3 * (1.0 - np.minimum(Qb / (2.0 * q['B']), 1))
                Emte = ((per-1)*F*(H+L)*q['Eelec'] + per*F*L*q['EDA'] + F*(H+L)*(q['Eelec'] + q['efs']*50**2))
                elig &= E >= Emte
            if guard: elig &= E - e_ch >= Eres
            mw = wgt[elig].mean() if elig.any() else 1.0
            thr = np.clip(T * wgt / max(mw, 1e-12), 0, 1)
            ch = elig & (rng.random(N) < thr); Gset |= ch
            if c['kind'] == 'eechs':
                if r % per == 0 or lab is None:
                    ch0 = np.where(ch)[0]; lab = np.full(N, -1)
                    if len(ch0):
                        dd = np.linalg.norm(xy[:, None] - xy[None, ch0], axis=2); kk = dd.argmin(1)
                        lab = np.where(dd[np.arange(N), kk] < dbs, kk, -1); lab[ch0] = np.arange(len(ch0))
                    nclu = len(ch0)
                ch = np.zeros(N, bool)
                for k in range(nclu):
                    m_ = np.where(alive & (lab == k) & ((E - e_ch >= Eres) if guard else True))[0]
                    if len(m_): ch[m_[E[m_].argmax()]] = True
                msgs += (alive.sum() if r % per == 0 else ch.sum())
            else:
                msgs += alive.sum() + ch.sum()
            nelect += 1
        cid = np.where(ch & alive)[0]; mem = np.where(alive & ~ch)[0]
        nch_sum += len(cid)
        cons = np.where(alive, q['Eidle'], 0.0)
        idle_sum += q['Eidle'] * na

        # ------------------------------------------------ cluster membership
        if c['kind'] == 'eechs' and len(cid) and lab is not None:
            pos = {lab[x]: i for i, x in enumerate(cid)}
            j = np.array([pos.get(lab[m], -1) for m in mem], dtype=int)
            dmin = np.where(j >= 0, np.linalg.norm(xy[mem] - xy[cid[np.maximum(j, 0)]], axis=1), np.inf)
        elif len(cid):
            dm = np.linalg.norm(xy[mem, None] - xy[None, cid], axis=2); j = dm.argmin(1)
            dmin = dm[np.arange(len(mem)), j]
        else:
            j = np.full(len(mem), -1); dmin = np.full(len(mem), np.inf)
        direct = dbs[mem] <= dmin; dist = np.where(direct, dbs[mem], dmin)

        # ------------------------------------------------ sensing (application layer)
        Qm = Qb[mem]
        if c['kind'] == 'lq' and c['r1']:
            if q['util'] == 'log':     # U(a) = log(1+a): maximise V log(1+a) - Q a
                a = np.where(Qm > 0, np.floor(np.minimum(F, np.maximum(0.0, V / np.maximum(Qm, 1e-12) - 1))), F)
            else:                      # U(a) = a: maximise (V - Q) a
                a = np.where(Qm < V, float(F), 0.0)
        else:
            a = np.full(len(mem), float(F))
        room = Bcap - Qm; ov = np.maximum(a - room, 0); overflow += ov.sum()
        a = a - ov; Qm = Qm + a; sensed += a.sum()
        util += np.log1p(a).sum() + np.log1p(F) * len(cid)

        # ------------------------------------------------ flush decision (MAC)
        cost = e_pkt(Qm, dist)
        if c['kind'] == 'lq':
            if c['r2']:
                Et = np.abs(np.minimum(E[mem] - theta, 0.0))
                fl = (Qm > 0) & (Qm**2 >= kappa * Et * cost)
            else:
                fl = Qm > 0
        elif c['kind'] == 'xl':
            low = (E[mem] < q['low_frac'] * q['Emax']) | (Ebar[mem] < np.median(Ebar[alive]))
            fl = (Qm > 0) & ~(low & (Qm <= F))       # hold one round's readings when energy-poor
        else:
            fl = Qm > 0
        if guard:
            ok = E[mem] - cost >= Eres[mem]
            guard_bind += int((fl & ~ok).sum())
            fl &= ok
        nsend = np.where(fl, Qm, 0.0)
        cons[mem] += np.where(fl, cost, 0.0)
        Qm = Qm - nsend; Qb[mem] = Qm
        qmax_seen = max(qmax_seen, float((Qm + nsend).max()) if len(mem) else 0.0)

        # ------------------------------------------------ CH: receive, fuse, forward
        if len(cid):
            sel = ~direct & fl
            jj = j[sel]
            rxbits = np.bincount(jj, weights=np.ceil(nsend[sel] / npk) * H + nsend[sel] * L, minlength=len(cid))
            rxread = np.bincount(jj, weights=nsend[sel], minlength=len(cid))
            cons[cid] += (rxbits * q['Eelec'] + (rxread + F) * L * q['EDA']
                          + q['Est'] + (H + F * L) * (q['Eelec'] + amp(dbs[cid])))
            deliv += rxread.sum() + F * len(cid); sensed += F * len(cid)
            dnode[cid] += F
        deliv += nsend[direct].sum()
        dnode[mem] += np.where(direct | (j >= 0), nsend, 0.0)
        backlog += Qb[alive].sum()

        if tr is not None:
            if tr['node'] is None:
                tr['node'] = int(np.argsort(dbs)[N // 2])         # median-distance node
            n0 = tr['node']; tr['E'].append(float(E[n0])); tr['Q'].append(float(Qb[n0]))
            k0 = np.where(mem == n0)[0]
            tr['fl'].append(int(fl[k0[0]]) if len(k0) else 0); tr['ch'].append(int(ch[n0]))

        E = E - cons; tot_cons += cons[alive].sum()
        died = alive & (E <= 0)
        lost += Qb[died].sum(); Qb[died] = 0; E[died] = 0; alive &= ~died

    h = np.array(hist + [0] * (q['Rmax'] - len(hist)))
    def first(cnd):
        k = np.where(cnd)[0]; return int(k[0]) if len(k) else q['Rmax']
    R = len(hist)
    out = dict(proto=proto, seed=seed, scen=scen, over=over,
               FND=first(h < N), HND=first(h <= N / 2), alive_end=int(h[q['Rmax'] - 1]),
               mean_alive=float(h.mean()), deaths=int(N - h[q['Rmax'] - 1]),
               deliv=float(deliv), sensed=float(sensed), nominal=float(N * F * q['Rmax']),
               cons=float(tot_cons), cons_comm=float(tot_cons - idle_sum),
               EEc=float(deliv * L / max(tot_cons - idle_sum, 1e-12) / 1e3), EE=float(deliv * L / max(tot_cons, 1e-12) / 1e3),
               delay=float(Tr / 2 + Tr * backlog / max(deliv, 1.0)),
               lost=float(lost), overflow=float(overflow), util=float(util / (N * q['Rmax'])),
               msgs=float(msgs / max(R, 1)), nch=float(nch_sum / max(R, 1)), qmax=qmax_seen,
               kappa=float(kappa), V=float(V), Bcap=float(Bcap), guard_bind=guard_bind,
               jain=float(dnode.sum()**2 / max(N * (dnode**2).sum(), 1e-12)), dmin_node=float(dnode.min()), dnode=dnode.tolist(), alive=h[::60].tolist())
    if tr is not None: out['trace'] = tr
    return out


if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument('--scen', default='S1'); ap.add_argument('--seeds', default='0:30')
    ap.add_argument('--protos', default=','.join(PROTO)); ap.add_argument('--over', default='{}')
    ap.add_argument('--out', required=True); ap.add_argument('--procs', type=int, default=2)
    a = ap.parse_args()
    s0, s1 = map(int, a.seeds.split(':')); over = json.loads(a.over)
    jobs = [(p, s, a.scen, over) for s in range(s0, s1) for p in a.protos.split(',')]
    with Pool(a.procs) as pl:
        res = pl.map(run, jobs, chunksize=1)
    json.dump(res, open(a.out, 'w'))
    print('done', a.out, len(res))
