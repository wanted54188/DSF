"""Draw publication plots from the reassessment tables and archived SGS spectra."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from reassess import HERE, PRODUCTION, VALID, METHODS, F, archive, geometry, read_json, shell_sum


MODELS = ("Clark4", *METHODS.values())
LABELS = {"Clark4": "NGM4", "FDNS": "FDNS", **{m:m for m in METHODS.values()}}
COLORS = dict(zip((*MODELS,"FDNS"),("#d92a60","#007eb8","#e87712","#934fd1","#00955e","#000000")))
LINES = dict(zip((*MODELS,"FDNS"),("--","--","-.",":","-.","-")))
REFS = {"n1024_main": ("#2475a8","--",r"$1024^2$, $2.5\times10^{-4}$"),
        "n1536_main": ("#dc682c","-.",r"$1536^2$, $2.5\times10^{-4}$"),
        "n1536_half": ("#161616","-",r"$1536^2$, $1.25\times10^{-4}$")}
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "pdf.fonttype":42,"axes.labelsize":9,"axes.titlesize":10})


def tables(target, name):
    return pd.read_csv(target/(name+".csv"))


def save(fig, target, filename, footer):
    fig.text(.5,.015,footer,ha="center",fontsize=7,color="#444444")
    fig.tight_layout(rect=(0,.05,1,.93))
    for ext in ("pdf","png"):
        fig.savefig(target/"figures"/(filename+"."+ext),dpi=360,facecolor="white")
    plt.close(fig)


def legend(fig, labels, mode):
    if mode=="reference":
        handles=[Line2D([0],[0],color=c,linestyle=s,label=l) for c,s,l in REFS.values()]
    else:
        handles=[Line2D([0],[0],color=COLORS[m],linestyle=LINES[m],label=LABELS[m]) for m in labels]
    fig.legend(handles=handles,ncol=len(handles),loc="upper center",frameon=False,bbox_to_anchor=(.5,.985))


def shade(ax):
    for k in (16.5,29.5):
        ax.axvline(k,color=".7",ls=":",lw=.75)


def plot_reference(target):
    d=tables(target,"FILTERED_DIFFERENCE")
    s=tables(target,"FILTERED_VALUES")
    fig, axs=plt.subplots(3,2,figsize=(7.6,7.0),sharex=True)
    for j,case in enumerate(sorted(d.case.unique())):
        p=d[d.case.eq(case)]
        for i,field in enumerate(("reference_omega","reference_source")):
            ax=axs[i,j]
            for axis,ls in (("space","-"),("time","--")):
                for b,col in zip(("all","low","mid","high"),("#121212","#2877a6","#dc682c","#9145b0")):
                    q=p[p.axis.eq(axis)&p.metric.eq(field)&p.band.eq(b)]
                    ax.semilogy(q.time,100*q.relative.clip(lower=1e-12),ls,color=col,lw=1.3,label=b if axis=="space" else None)
            ax.axhline(1,color=".65",lw=.7,ls=":")
            ax.set(ylabel=r"$\bar{\omega}$ relative RMS (%)" if i==0 else "Source relative RMS (%)")
            if i==0:
                ax.set_title(f"Flow {case%1000}")
                if j==0: ax.legend(frameon=False,ncol=4,fontsize=7,loc="lower right")
        ax=axs[2,j]
        for axis,ls in (("space","-"),("time","--")):
            a,b=(("n1024_main","n1536_main") if axis=="space" else ("n1536_main","n1536_half"))
            qa=s[s.case.eq(case)&s.reference.eq(a)]
            qb=s[s.case.eq(case)&s.reference.eq(b)]
            for quantity,col in (("K","#2877a6"),("Z","#d44247")):
                r=100*abs(qa[quantity].to_numpy()/qb[quantity].to_numpy()-1)
                ax.semilogy(qa.time,np.maximum(r,1e-12),ls,color=col,label=quantity if axis=="space" else None)
        ax.set(xlabel="Time",ylabel="Absolute K, Z difference (%)",xlim=(1,5))
        if j==0: ax.legend(frameon=False,ncol=2,fontsize=8)
    fig.suptitle(r"Space: $1024^2$ vs $1536^2$ at $2.5\times10^{-4}$ (solid)    Time: $1536^2$, step halved (dashed)",fontsize=9,y=.98)
    save(fig,target,"R01_Filtered_Reference_Sensitivity",r"Gaussian $\Delta=8\pi/100$, $P_{33}$ on $100^2$. Shells 1–16, 17–29, 30–47. Dotted line: legacy 1% field check.")


def plot_native(target):
    d=tables(target,"NATIVE_SPECTRA")
    fig,axs=plt.subplots(3,2,figsize=(7.6,7.1),sharex=True)
    for j,case in enumerate((110240101,110240102)):
        for index,(metric,yl) in enumerate((("E","Native E(k)"),("Z","Native Z(k)"),("viscous_Z_dissipation","Viscous Z dissipation"))):
            ax=axs[index,j]
            for ref,(col,ls,_) in REFS.items():
                for t,alpha in [(3,.35),(5,1)]:
                    q=d[d.case.eq(case)&d.reference.eq(ref)&d.time.eq(t)&d[metric].gt(0)]
                    ax.loglog(q.k,q[metric],ls,color=col,alpha=alpha,lw=1.35)
            for cut in (341,511): ax.axvline(cut,color=".6",lw=.65,ls=":")
            ax.set(ylabel=yl,xlim=(1,760))
            if index==0: ax.set_title(f"Flow {case%1000}")
        axs[-1,j].set_xlabel("Native radial shell k")
    legend(fig,[],"reference")
    save(fig,target,"R02_Native_DNS_Tails","Native DNS before filtering. Faint curves: t = 3. Solid opacity: t = 5. Dotted: axis cutoffs 341 and 511.")


def plot_rank(target):
    d=tables(target,"REFERENCE_RANKING_ROBUSTNESS")
    d=d[d.case.isin((110240101,110240102))]
    fig,axs=plt.subplots(2,2,figsize=(7.6,5.4),sharey=True)
    for ax,(case,seed) in zip(axs.flat,((c,s) for c in (110240101,110240102) for s in (2026,2028))):
        p=d[d.case.eq(case)&(d.network_seed.eq(seed)|d.network_seed.isna())]
        for j,ref in enumerate(REFS):
            part=p[p.reference.eq(ref)].set_index("model")
            for i,m in enumerate(MODELS):
                shift=(j-1)*.20
                ax.scatter(i+shift,part.at[m,"omega_MSE"],marker=("o","s","D")[j],s=23,
                           facecolors=(COLORS[m] if j==2 else "white"),edgecolors=COLORS[m],linewidth=.9)
        ax.set_title(f"Flow {case%1000}  |  network seed {seed}")
        ax.set_xticks(range(5),[LABELS[m].replace("DSF-","") for m in MODELS])
        ax.set_ylim(0,1.08*d.omega_MSE.max())
        ax.set_ylabel("Full-window vorticity MSE")
    fig.legend(handles=[Line2D([0],[0],color="k",ls="",marker=mark,mfc=("k" if j==2 else "white"),
                label=ref[2]) for j,(mark,ref) in enumerate(zip(("o","s","D"),REFS.values()))],
                loc="upper center",ncol=3,frameon=False)
    save(fig,target,"R03_Five_Model_Reference_Rescoring",r"Fixed LES states, 80 samples in [1.05, 5]. Gaussian $\Delta=8\pi/100$ and $P_{33}$ on $100^2$.")


def plot_physics(target):
    s=tables(target,"LES_STATISTICS")
    p=tables(target,"LES_PDF")
    fig,axs=plt.subplots(2,2,figsize=(7.6,5.1))
    for m in (*MODELS,"FDNS"):
        q=s[s.case.eq(110240102)&s.model.eq(m)&(s.seed.eq(2026)|s.seed.isna())]
        kw=dict(color=COLORS[m],ls=LINES[m],lw=2 if m=="FDNS" else 1.35)
        for ax,k in zip((axs[0,0],axs[0,1],axs[1,0]),("K","Z","CC")):
            ax.plot(q.time,q[k],**kw)
            ax.set(xlabel="Time",ylabel={"K":"Resolved K","Z":"Resolved Z","CC":"Vorticity CC with FDNS"}[k],xlim=(1,5))
        q=p[p.case.eq(110240102)&p.model.eq(m)&(p.seed.eq(2026)|p.seed.isna())]
        axs[1,1].semilogy((q.left+q.right)/2,q.density.where(q.density>0),**kw)
    axs[1,1].set(xlabel=r"$\omega / \sigma_{\mathrm{FDNS}}(t)$",ylabel="Vorticity PDF")
    legend(fig,(*MODELS,"FDNS"),"model")
    save(fig,target,"R04_Autonomous_Physical_Statistics",
         r"Flow 102, neural seed 2026. FDNS $1536^2$ at half step, Gaussian $\Delta=8\pi/100$, $P_{33}$. PDF: 80 archived times.")


def plot_labels(target):
    p=tables(target,"APRIORI_ARCHIVED_REPLAY")
    d=tables(target,"FIXED_REPLAY")
    fig,axs=plt.subplots(2,3,figsize=(9.5,5.1))
    for i,case in enumerate((110240101,110240102)):
        q=p[p.case.eq(case)].set_index("model")
        ax=axs[i,0]
        for j,m in enumerate(MODELS):
            ax.bar(j,q.at[m,"Source_MSE_all"],color=COLORS[m],width=.62)
        ax.set(title=f"Flow {case%1000}: archived a priori replay",ylabel="Absolute Source MSE")
        ax.set_xticks(range(5),[LABELS[m].replace("DSF-","") for m in MODELS])
        ax=axs[i,1]
        q=d[d.case.eq(case)&d.metric.eq("Pi_Z")]
        for ref,(col,ls,_) in REFS.items():
            b=q[q.label_reference.eq(ref)]
            c=b[b.model.eq("DSF-Hold")]
            ax.plot(c.time,c.backscatter_F1,ls,color=col,lw=1.35,marker="o",ms=3)
            comp=b.pivot(index="time",columns="model",values="MSE")
            axs[i,2].plot(comp.index,comp["DSF-Hold"]-comp["DSF-Sym"],ls,color=col,lw=1.35,marker="o",ms=3)
        ax.set(title=f"Flow {case%1000}: fixed Hold output",ylabel="Backscatter F1",xlabel="Time",
               xticks=(1.05,3,5),ylim=(0,1.04))
        ax=axs[i,2]
        ax.axhline(0,color=".5",lw=.7)
        ax.set(title=f"Flow {case%1000}: fixed outputs",ylabel=r"$\Pi_Z$ MSE (Hold minus Sym)",
               xlabel="Time",xticks=(1.05,3,5))
        ax.ticklabel_format(axis="y",style="sci",scilimits=(-2,2),useMathText=True)
    legend(fig,[],"reference")
    save(fig,target,"R05_Apriori_Label_Check",
         r"Seed 2026, Gaussian $\Delta=8\pi/100$, $P_{33}$. Left: 800 fine-reference states per flow. Middle and right: fixed predictions; labels vary.")


def archived_spectra(target):
    run=PRODUCTION/"results/run01"
    report=run/read_json(run/"LATEST_REPORT.json")["path"]
    data=pd.read_csv(report/"SPECTRA.csv",usecols=["case","seed","method","time_s","shell","E","Z","T_K_SGS","T_Z_SGS"])
    s=data[data.case.eq(110240102)&data.seed.eq(2026)&data.time_s.isin((3,5))&
           data.method.isin(METHODS)&data.shell.between(1,47)].copy()
    s["model"]=s.method.map(METHODS)
    s=s.rename(columns={"time_s":"time","shell":"k","T_K_SGS":"TK","T_Z_SGS":"TZ"})
    s["source"]=str((report/"SPECTRA.csv").relative_to(HERE.parents[2]))
    refs=tables(target,"FILTERED_SPECTRA")
    refs=refs[refs.case.eq(110240102)&refs.reference.eq("n1536_half")&refs.time.isin((3,5))].copy()
    refs["model"]="FDNS"
    refs["source"]="FILTERED_SPECTRA.csv"
    op=dict(archive(run/"OPERATORS.npz"))
    masks,shell=geometry(op)
    records=[]
    for time,step in ((3,400),(5,800)):
        path=VALID/f"CLARK/110240102/limited_completion/Clark4_0.005/steps/{step:04d}.npz"
        with archive(path) as z:
            np.testing.assert_allclose(z["omega"],z["observation__omega"],atol=2e-12,rtol=0)
            h=F(z["omega"])*masks["all"]
            p=.5*abs(h)**2
            tz=(h.conj()*F(z["observation__Source"])).real
        specs=[shell_sum(x,shell) for x in (p*op["inverse"],p,tz*op["inverse"],tz)]
        for k in range(1,48):
            records.append(dict(case=110240102,model="Clark4",time=time,k=k,
                source=str(path.relative_to(HERE.parents[2])), **dict(zip(("E","Z","TK","TZ"),(a[k] for a in specs)))))
    cols=["case","model","time","k","E","Z","TK","TZ","source"]
    d=pd.concat([s[cols],refs[cols],pd.DataFrame(records)[cols]],ignore_index=True)
    assert d.groupby(["model","time"]).size().eq(47).all()
    d.to_csv(target/"ARCHIVED_LES_SPECTRA.csv",index=False)
    fig,axs=plt.subplots(4,2,figsize=(7.6,9.0),sharex=True)
    for j,t in enumerate((3,5)):
        for i,metric in enumerate(("E","Z","TK","TZ")):
            ax=axs[i,j]
            for m in (*MODELS,"FDNS"):
                a=d[d.time.eq(t)&d.model.eq(m)]
                ax.plot(a.k,a[metric],color=COLORS[m],ls=LINES[m],lw=1.6 if m=="FDNS" else 1.2)
            ax.set_xscale("log")
            if i<2: ax.set_yscale("log")
            else: ax.axhline(0,lw=.6,color=".6")
            shade(ax)
            ax.set(ylabel={"E":"E(k)","Z":"Z(k)","TK":"SGS T$_K$(k)","TZ":"SGS T$_Z$(k)"}[metric])
            if i==0: ax.set_title(f"t = {t}")
        axs[-1,j].set_xlabel("Retained radial shell k")
    legend(fig,(*MODELS,"FDNS"),"model")
    save(fig,target,"R06_Archived_Autonomous_Spectra",
         r"Flow 102, seed 2026. FDNS $1536^2$ at half step, Gaussian $\Delta=8\pi/100$, $P_{33}$. Dotted: shells 16/17 and 29/30.")


def main(target):
    (target/"figures").mkdir(exist_ok=True)
    for plot in (plot_reference, plot_native, plot_rank, plot_physics, plot_labels, archived_spectra):
        plot(target)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE)
    main(parser.parse_args().output.resolve())
