"""HCB-XL simulation: hybrid RF+solar harvesting, cross-layer CH election, adaptive circular buffer. See README.md."""
# Ablation configurations C1..C7; all parameters in dict P (Table 3 of the paper).
import numpy as np, json, sys
from multiprocessing import Pool
P=dict(N=100,M=100.0,bs=(50.0,50.0),E0=5.0,Emax=5.0,p=0.1,Tr=60.0,F=10,
       Eelec=50e-9,efs=10e-12,emp=0.0013e-12,EDA=5e-9,Est=20e-6,H=2000,L=2000,Eidle=0.3e-3,
       # RF power beacon co-located with BS, 915 MHz
       Pt=3.0,Gt=10**0.6,Gr=10**0.2,f=915e6,eta_rf=0.3,rf_sens=1e-6,
       # PV: small panel under partial canopy
       A=1e-4,eta_pv=0.15,Gmax=8.0,day_h=12.0,cloud_max=0.6,shade_min=0.3,
       B=10,w=(0.4,0.4,0.2),beta=0.2,low_frac=0.3,Rmax=28800)
CFG={ # name: (rf, pv, buffer, election, adaptive)
 'C1 Battery-only LEACH':(0,0,0,'leach',0),
 'C2 RF-only LEACH':(1,0,0,'leach',0),
 'C3 Solar-only LEACH':(0,1,0,'leach',0),
 'C4 Hybrid LEACH':(1,1,0,'leach',0),
 'C5 Hybrid + CB (layered)':(1,1,1,'leach',0),
 'C6 Hybrid + CB + energy-aware election (DEEC-style)':(1,1,1,'deec',0),
 'C7 HCB-XL (hybrid + CB + cross-layer)':(1,1,1,'xl',1),
 # recent harvesting-aware baseline (Ren & Yao, Sensors 2020): clusters formed every 1/p rounds;
 # within an epoch the scheduling node appoints the member with the highest residual energy as next CH
 'C8 EECHS (hybrid + CB)':(1,1,1,'eechs',0)}
def run(args):
    name,seed,over=args; q=dict(P); q.update(over); rf,pv,cb,el,ad=CFG[name]
    rng=np.random.default_rng(seed); N=q['N']
    xy=rng.uniform(0,q['M'],(N,2)); bs=np.array(q['bs']); dbs=np.linalg.norm(xy-bs,axis=1)
    lam=3e8/q['f']; d0=np.sqrt(q['efs']/q['emp'])
    amp=lambda d: np.where(d<d0,q['efs']*d**2,q['emp']*d**4)
    prf_mean=q['Pt']*q['Gt']*q['Gr']*lam**2/((4*np.pi)**2*np.maximum(dbs,1.0)**2)
    shade=rng.uniform(q['shade_min'],1.0,N)
    E=np.full(N,q['E0']); alive=np.ones(N,bool); G=np.zeros(N,bool); Eh_bar=np.zeros(N)
    Tr,F,B,H,L=q['Tr'],q['F'],q['B'],q['H'],q['L']; per=int(round(1/q['p']))
    hist=[]; tot_cons=0.0; tot_h_rf=0.0; tot_h_pv=0.0; delivered=0.0; lost=0.0; delay_num=0.0
    lab=None; nclu=0
    pend=np.zeros(N)  # readings held across rounds (adaptive)
    Emte=(per-1)*F*(H+L)*q['Eelec']+per*F*L*q['EDA']+F*(H+L)*(q['Eelec']+q['efs']*50**2)
    for r in range(q['Rmax']):
        na=alive.sum(); hist.append(int(na))
        if na==0: break
        t_h=(r*Tr/3600.0)%24.0  # sim starts at sunrise
        # ---- PHY: harvesting
        eh=np.zeros(N)
        if rf:
            h2=rng.exponential(1.0,N); P_in=prf_mean*h2
            e=np.where(P_in>=q['rf_sens'],q['eta_rf']*P_in,0.0)*Tr; eh+=e; tot_h_rf+=e[alive].sum()
        if pv and t_h<q['day_h']:
            Gt_=q['Gmax']*np.sin(np.pi*t_h/q['day_h']); cloud=rng.uniform(0,q['cloud_max'])
            e=q['eta_pv']*q['A']*Gt_*shade*(1-cloud)*Tr; eh+=e; tot_h_pv+=(e*alive).sum()
        E=np.where(alive,np.minimum(q['Emax'],E+eh),E)
        Eh_bar=q['beta']*eh+(1-q['beta'])*Eh_bar
        # ---- DL: CH election
        if r%per==0: G[:]=False
        T=q['p']/(1-q['p']*(r%per))
        elig=alive&~G
        if el=='leach': wgt=np.ones(N)
        elif el=='deec': wgt=E/q['E0']
        else:
            hmax=max(Eh_bar[alive].max(),1e-12); w1,w2,w3=q['w']
            free=1.0-pend/(2.0*B)  # free buffer fraction
            wgt=w1*E/q['E0']+w2*Eh_bar/hmax+w3*free
            elig&=E>=Emte
        mw=wgt[elig].mean() if elig.any() else 1.0
        thr=np.clip(T*wgt/max(mw,1e-12),0,1)
        ch=elig&(rng.random(N)<thr); G|=ch
        if el=='eechs':
            if r%per==0 or lab is None:   # set-up: LEACH election, fixed membership for the epoch
                ch0=np.where(ch)[0]; lab=np.full(N,-1)
                if len(ch0):
                    dd=np.linalg.norm(xy[:,None]-xy[None,ch0],axis=2); k=dd.argmin(1)
                    lab=np.where(dd[np.arange(N),k]<dbs,k,-1); lab[ch0]=np.arange(len(ch0))
                nclu=len(ch0)
            ch=np.zeros(N,bool)
            for k in range(nclu):   # scheduling node appoints max-residual-energy member
                m_=np.where(alive&(lab==k))[0]
                if len(m_): ch[m_[E[m_].argmax()]]=True
        cid=np.where(ch)[0]; mem=np.where(alive&~ch)[0]
        cons=np.where(alive,q['Eidle'],0.0)
        # ---- APP/NET: member transmissions (readings per round = F)
        if el=='eechs' and len(cid):
            pos={lab[c]:i for i,c in enumerate(cid)}
            j=np.array([pos.get(lab[m],-1) for m in mem],dtype=int)
            dmin=np.where(j>=0,np.linalg.norm(xy[mem]-xy[cid[np.maximum(j,0)]],axis=1),np.inf)
        elif len(cid):
            dm=np.linalg.norm(xy[mem,None]-xy[None,cid],axis=2); j=dm.argmin(1); dmin=dm[np.arange(len(mem)),j]
        else:
            j=np.full(len(mem),-1); dmin=np.full(len(mem),np.inf)
        direct=dbs[mem]<=dmin; dist=np.where(direct,dbs[mem],dmin)
        if not cb:
            ntx=np.full(len(mem),float(F)); bits=np.full(len(mem),F*(H+L)*1.0); sent=np.full(len(mem),float(F)); dly=np.full(len(mem),Tr/F/2)
        else:
            low=np.zeros(len(mem),bool)
            if ad: low=(E[mem]<q['low_frac']*q['Emax'])|(Eh_bar[mem]<np.median(Eh_bar[alive]))
            # adaptive: low-energy members hold this round's readings and flush 2B every other round
            hold=low&(pend[mem]==0)
            flush=~hold
            nread=np.where(flush,F+pend[mem],0.0)
            ntx=np.where(flush,np.ceil(nread/(2*B if ad else B)),0.0)
            bits=ntx*H+nread*L; sent=nread
            dly=np.where(nread>0,np.where(pend[mem]>0,Tr,Tr/2),0)
            pend[mem]=np.where(hold,F,0.0)
        etx=ntx*q['Est']+bits*(q['Eelec']+amp(dist))
        cons[mem]+=etx
        # ---- CH side: receive, fuse, forward to BS
        if len(cid):
            jj=j[~direct]
            rxbits=np.bincount(jj,weights=bits[~direct],minlength=len(cid))
            rxread=np.bincount(jj,weights=sent[~direct],minlength=len(cid))
            fused_frames=F  # each frame fused into one reading
            if cb: ch_ntx=1.0; ch_bits=H+fused_frames*L
            else: ch_ntx=float(F); ch_bits=F*(H+L)
            cons[cid]+=rxbits*q['Eelec']+(rxread+F)*L*q['EDA']+ch_ntx*q['Est']+ch_bits*(q['Eelec']+amp(dbs[cid]))
            delivered+=rxread.sum()+F*len(cid)
        delivered+=sent[direct].sum()
        delay_num+=(dly*sent).sum()
        E=E-cons; tot_cons+=cons[alive].sum()
        died=alive&(E<=0); lost+=pend[died].sum(); pend[died]=0
        E[died]=0; alive&=~died
    h=np.array(hist)
    def first(c): 
        k=np.where(c)[0]; return int(k[0]) if len(k) else q['Rmax']
    fnd=first(h<N); hnd=first(h<=N/2); lnd=first(h==0)
    return dict(name=name,seed=seed,FND=fnd,HND=hnd,LND=lnd,cons=tot_cons,hrf=tot_h_rf,hpv=tot_h_pv,
                deliv=delivered,lost=lost,EE=delivered*L/max(tot_cons,1e-12)/1e3,
                EEb=delivered*L/(N*q['E0'])/1e6,delay=delay_num/max(delivered,1),alive=h[::60].tolist())
if __name__=="__main__":
    seeds=int(sys.argv[1]) if len(sys.argv)>1 else 30
    jobs=[(n,s,{}) for n in CFG for s in range(seeds)]
    with Pool() as pl: res=pl.map(run,jobs)
    json.dump(res,open('res_main.json','w'))
    print("done",len(res))
