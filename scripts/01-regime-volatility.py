"""مارکت رژیم چیست؟ نوسان بازار ثابت نیست: numbers and figures for the blog post /blog/market-regime-intro/.
Run from the repository root: python3 scripts/01-regime-volatility.py
Data: simulated BTC daily candles (data/btc_daily_5y.csv), not real prices.
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


C = {"bg": "#ffffff", "ink": "#1b1a17", "ref": "#8a8272", "s1": "#008a7a", "s2": "#c2410c",
     "low": "#d99566", "mid": "#c25a20", "high": "#8f2f0a"}
matplotlib.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                            "grid.color": "#e3dccd", "figure.facecolor": C["bg"], "axes.facecolor": C["bg"],
                            "axes.unicode_minus": False})
ROOT = pathlib.Path(__file__).resolve().parents[1]  # repository root
(ROOT / "figures").mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data/btc_daily_5y.csv", parse_dates=["date"], index_col="date")
close = df["close"]
r = np.log(close).diff()
ANN = np.sqrt(365)

# regime = tercile of 30-day volatility, known at the close of day t
vol = r.rolling(30).std() * ANN
lo, hi = vol.quantile([1 / 3, 2 / 3])
reg = pd.cut(vol, [-np.inf, lo, hi, np.inf], labels=["low", "mid", "high"])
nxt = r.shift(-1)  # the return we try to describe: the NEXT day
d = pd.DataFrame({"reg": reg, "nxt": nxt}).dropna()
g = d.groupby("reg", observed=True).nxt
tab = pd.DataFrame({"days": g.size(), "mean_pct": g.mean() * 100, "ann_vol_pct": g.std() * ANN * 100, "kurt": g.apply(lambda x: x.kurt())})
print("terciles", round(lo, 3), round(hi, 3))
print(tab.round(2))
t_high = d[d.reg == "high"].nxt
print("t-stat of mean in high regime", t_high.mean() / (t_high.std() / np.sqrt(len(t_high))))

rg = reg.dropna()
print("same regime next day", (rg == rg.shift(-1)).mean(), "30d later", (rg == rg.shift(-30)).mean())
runs = (rg != rg.shift()).cumsum()
print("run length median", rg.groupby(runs).size().median(), "n runs", rg.groupby(runs).size().shape[0])

# one fixed rule, yearly Sharpe
def sharpe(x):
    x = x.dropna()
    return x.mean() / x.std() * ANN
pos = (close.rolling(20).mean() > close.rolling(100).mean()).astype(float).shift(1)
sr = pos * r
ys = sr["2022":].groupby(sr["2022":].index.year).apply(sharpe)
print("MA 20/100 sharpe by year", ys.round(2).to_dict(), "full", round(sharpe(sr), 2), "buy&hold", round(sharpe(r), 2))
se = 1 / np.sqrt(1)  # daily data: SE of annualized Sharpe ~ 1/sqrt(years)
print("approx SE of one-year Sharpe", round(se, 2))

# naive trend/range label: efficiency ratio of past 30 days, known at t-1
p = np.log(close)
er = (p.diff(30).abs() / p.diff().abs().rolling(30).sum()).shift(1)
for name, m in [("trend ER>0.3", er > 0.3), ("mid", (er >= 0.1) & (er <= 0.3)), ("range ER<0.1", er < 0.1)]:
    print(name, int(m.sum()), "MA sharpe", round(sharpe(sr[m]), 2))

# ---- figures (English labels: matplotlib has no Persian shaping)
BAND = .38  # same opacity for the shading and its legend swatches
# Regimes are ordered (low < mid < high volatility): one hue, light to dark.
col = {"low": C["low"], "mid": C["mid"], "high": C["high"]}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 1.3]})
a1.plot(close.index, close, color=C["ink"], lw=1)
a1.set_ylabel("Price (simulated)")
a2.plot(vol.index, vol * 100, color=C["ink"], lw=1)
a2.axhline(lo * 100, color=C["ref"], ls=":", lw=.8); a2.axhline(hi * 100, color=C["ref"], ls=":", lw=.8)
a2.set_ylabel("30-day vol (%/yr)")
for ax in (a1, a2):
    s = None
    for t, v in rg.items():
        if v != s:
            if s is not None:
                ax.axvspan(t0, t, color=col[s], alpha=BAND, lw=0)
            t0, s = t, v
    ax.axvspan(t0, rg.index[-1], color=col[s], alpha=BAND, lw=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)  # the shaded regimes are the background here
# Shading alone would carry the regime by colour only: name it in a legend.
from matplotlib.patches import Patch
a1.legend(handles=[Patch(color=col[k], alpha=BAND, label=f"{k} vol") for k in ("low", "mid", "high")],
          loc="upper left", ncol=3, fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/regime-vol-timeline.png", dpi=150)

fig, ax = plt.subplots(figsize=(6.5, 3.6))
x = ["low", "mid", "high"]
ax.bar(x, [tab.loc[k, "ann_vol_pct"] for k in x], color=[col[k] for k in x])
for i, k in enumerate(x):
    ax.text(i, tab.loc[k, "ann_vol_pct"] + 1.5, f"{tab.loc[k, 'ann_vol_pct']:.0f}%", ha="center")
ax.set_ylabel("Next-day vol (annualized %)"); ax.set_xlabel("Regime today (30-day vol tercile)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(ROOT / "figures/regime-next-day-vol.png", dpi=150)
