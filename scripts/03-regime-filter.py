"""فیلتر مارکت رژیم: بهتر شدن یا Overfitting؟: numbers and figures for the blog post /blog/market-regime-filter/.
Run from the repository root: python3 scripts/03-regime-filter.py
Data: simulated BTC daily candles (data/btc_daily_5y.csv), not real prices.
"""
import pathlib
import numpy as np, pandas as pd
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
close = df["close"]; r = np.log(close).diff()
ANN = np.sqrt(365); COST = 0.001  # 0.1% per unit of position change
MIN = 365
IS, OOS = slice("2022", "2023"), slice("2024", "2025")

def sharpe(x):
    x = x.dropna(); return x.mean() / x.std() * ANN
def run(pos):
    pos = pos.fillna(0)
    return pos.shift(1) * r - COST * pos.diff().abs().shift(1)   # decide at close t, hold t+1

base_pos = (close.rolling(20).mean() > close.rolling(100).mean()).astype(float)
base = run(base_pos)
print("base Sharpe IS", round(sharpe(base[IS]), 2), "OOS", round(sharpe(base[OOS]), 2), "all", round(sharpe(base["2022":]), 2))

# grid of regime filters: stay flat when vol_w > past-only quantile q
grid = []
for w in (10, 20, 30, 60):
    vol = r.rolling(w).std()
    for q in (0.5, 0.6, 0.7, 0.8, 0.9):
        thr = vol.expanding(MIN).quantile(q)
        ok = (vol <= thr).astype(float).where(thr.notna(), 1.0)
        s = run(base_pos * ok)
        grid.append({"w": w, "q": q, "IS": sharpe(s[IS]), "OOS": sharpe(s[OOS]),
                     "on": (base_pos * ok)["2022":].mean(), "ok": ok})
g = pd.DataFrame(grid)
print(g[["w", "q", "IS", "OOS", "on"]].round(2).to_string())
best = g.loc[g.IS.idxmax()]
print("best IS config", best.w, best.q, "IS", round(best.IS, 2), "OOS", round(best.OOS, 2))
print("configs beating base IS", int((g.IS > sharpe(base[IS])).sum()), "/", len(g),
      "beating base OOS", int((g.OOS > sharpe(base[OOS])).sum()))
print("rank corr IS vs OOS", round(g.IS.corr(g.OOS, method="spearman"), 2))
print("median OOS of grid", round(g.OOS.median(), 2))

# random control: shift the best filter in time (keeps its run structure and exposure)
ok_b = best.ok
rng = np.random.default_rng(0)
vals = ok_b["2022":].values; idx = ok_b["2022":].index
rand = []
for _ in range(1000):
    k = rng.integers(30, len(vals) - 30)
    fake = pd.Series(np.roll(vals, k), index=idx).reindex(ok_b.index).fillna(1.0)
    rand.append(sharpe(run(base_pos * fake)[OOS]))
rand = np.array(rand)
print("random shifted filter OOS Sharpe: median", round(np.median(rand), 2), "p95", round(np.percentile(rand, 95), 2),
      "share >= best OOS", round((rand >= best.OOS).mean(), 3), "share >= grid median", round((rand >= g.OOS.median()).mean(), 3))
print("approx SE of 2-year Sharpe", round(1 / np.sqrt(2), 2))

# no-selection alternative: volatility targeting (size = target / vol, capped at 1)
vol20 = r.rolling(20).std() * ANN
tgt = vol20.expanding(MIN).median()
size = (tgt / vol20).clip(upper=1.0).where(tgt.notna(), 1.0)
vt = run(base_pos * size)
print("vol-target Sharpe IS", round(sharpe(vt[IS]), 2), "OOS", round(sharpe(vt[OOS]), 2), "avg size", round((base_pos*size)["2022":].mean(), 2))
for name, s in [("base", base), ("best filter", run(base_pos * ok_b)), ("vol target", vt)]:
    eq = s["2022":].cumsum(); dd = (eq - eq.cummax()).min()
    print(f"{name:12s} maxDD(log) {dd:.2f}  ann vol {s['2022':].std()*ANN:.2f}")

# figures
fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.scatter(g.IS, g.OOS, color=C["s1"], s=28, label="filter variants", edgecolors=C["bg"], linewidths=.6)
ax.scatter([best.IS], [best.OOS], color=C["s2"], s=60, label="best in-sample", edgecolors=C["bg"], linewidths=1)
ax.axhline(sharpe(base[OOS]), color=C["ref"], ls=":", lw=1); ax.axvline(sharpe(base[IS]), color=C["ref"], ls=":", lw=1)
ax.set_xlabel("Sharpe 2022-2023 (in-sample)"); ax.set_ylabel("Sharpe 2024-2025 (out-of-sample)")
ax.legend(frameon=False, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT / "figures/regime-filter-is-oos.png", dpi=150)

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.hist(rand, bins=40, color=C["ref"], label="time-shifted filters")
ax.axvline(best.OOS, color=C["s2"], lw=2, label="real filter")
ax.set_xlabel("Out-of-sample Sharpe with time-shifted filter"); ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(ROOT / "figures/regime-filter-random.png", dpi=150)
