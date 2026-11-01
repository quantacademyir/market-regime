"""تشخیص مارکت رژیم بدون نگاه به آینده: numbers and figures for the blog post /blog/market-regime-detection/.
Run from the repository root: python3 scripts/02-regime-detection.py
Data: simulated BTC daily candles (data/btc_daily_5y.csv), not real prices.
"""
import pathlib, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


C = {"bg": "#ffffff", "ink": "#1b1a17", "ref": "#8a8272", "s1": "#008a7a", "s2": "#c2410c",
     "low": "#d99566", "mid": "#c25a20", "high": "#8f2f0a"}
matplotlib.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                            "grid.color": "#e3dccd", "figure.facecolor": C["bg"], "axes.facecolor": C["bg"],
                            "axes.unicode_minus": False})

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]  # repository root
(ROOT / "figures").mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data/btc_daily_5y.csv", parse_dates=["date"], index_col="date")
close = df["close"]; r = np.log(close).diff()
ANN = np.sqrt(365)
vol = r.rolling(30).std() * ANN
MIN = 365  # first year only for warm-up

# --- A. one-sided threshold rule: quantiles from the past only
lo_e = vol.expanding(MIN).quantile(1 / 3).shift(0)   # known at close of day t (uses vol up to t)
hi_e = vol.expanding(MIN).quantile(2 / 3)
reg_a = pd.Series(np.where(vol > hi_e, 2, np.where(vol > lo_e, 1, 0)), index=vol.index).where(hi_e.notna())

# full-sample (look-ahead) version, for comparison
lo_f, hi_f = vol.quantile([1 / 3, 2 / 3])
reg_f = pd.Series(np.where(vol > hi_f, 2, np.where(vol > lo_f, 1, 0)), index=vol.index).where(vol.notna())
both = pd.concat([reg_a, reg_f], axis=1).dropna()
print("A: past-only vs full-sample label agreement", round((both.iloc[:, 0] == both.iloc[:, 1]).mean(), 3), "days", len(both))
print("A: high-vol share past-only", round((reg_a == 2).mean(), 3), "full-sample", round((reg_f.loc[reg_a.dropna().index] == 2).mean(), 3))

# --- B. GMM with 2 states, refit every 60 days on the PAST only
feat = pd.DataFrame({"lv": np.log(r.rolling(20).std()), "m": r.rolling(20).mean() / r.rolling(20).std()}).dropna()
def fit_gmm(X):
    g = GaussianMixture(2, covariance_type="full", n_init=3, random_state=0).fit(X)
    hi_comp = int(np.argmax(g.means_[:, 0]))      # state with higher mean log-vol = "high vol"
    return g, hi_comp
idx = feat.index
state = pd.Series(np.nan, index=idx)
refits = list(range(MIN, len(idx), 60))
for k, s in enumerate(refits):
    e = refits[k + 1] if k + 1 < len(refits) else len(idx)
    g, hc = fit_gmm(feat.iloc[:s].values)           # only rows before s
    p = g.predict_proba(feat.iloc[s:e].values)[:, hc]
    state.iloc[s:e] = (p > 0.5).astype(float)
state_wf = state.dropna()
# look-ahead GMM: fit once on everything
g, hc = fit_gmm(feat.values)
state_full = pd.Series((g.predict_proba(feat.values)[:, hc] > 0.5).astype(float), index=idx).loc[state_wf.index]
print("B: GMM walk-forward vs full-fit label agreement", round((state_wf == state_full).mean(), 3), "days", len(state_wf))
flips = lambda s: int((s != s.shift()).sum() - 1)
print("B: state changes walk-forward", flips(state_wf), "full-fit", flips(state_full), "A (high vs not)", flips((reg_a.loc[state_wf.index] == 2)))
print("B: high-vol share walk-forward", round(state_wf.mean(), 3))

# --- C. how well does each label predict NEXT-day variance?  (out-of-sample window only)
oos = state_wf.index
nxt2 = (r.shift(-1) ** 2).loc[oos]
nxt_abs = r.shift(-1).abs().loc[oos]
def summarize(name, flag):
    flag = flag.loc[oos].astype(bool)
    a, b = nxt_abs[flag], nxt_abs[~flag]
    print(f"C: {name:22s} high days {flag.sum():4d}  mean|r_next| high {a.mean()*100:.2f}% low {b.mean()*100:.2f}%  ratio {a.mean()/b.mean():.2f}")
    return a.mean() / b.mean()
ra = summarize("A threshold (past)", reg_a == 2)
rb = summarize("B GMM walk-forward", state_wf == 1)
rc = summarize("B GMM full fit (peek)", state_full == 1)
# plain baseline: yesterday's vol above its own expanding median
base = vol > vol.expanding(MIN).median()
rd = summarize("baseline vol>median", base)
# R^2 of continuous vol predictor
xy = pd.concat([vol.loc[oos], nxt2], axis=1).dropna()
c = np.corrcoef(xy.iloc[:, 0], xy.iloc[:, 1])[0, 1]
print("C: corr(30d vol, next-day r^2)", round(c, 3), " R2", round(c**2, 4))

# yearly stability of ratio for A vs B (simple check)
for yr in sorted(set(oos.year)):
    m = oos.year == yr
    f = (state_wf == 1).loc[oos][m]; fa = (reg_a == 2).loc[oos][m]
    def rat(fl):
        fl = fl.astype(bool)
        if fl.sum() < 20 or (~fl).sum() < 20: return np.nan
        return nxt_abs[m][fl].mean() / nxt_abs[m][~fl].mean()
    print("  year", yr, "days", int(m.sum()), "ratio A", round(rat(fa), 2), "ratio GMM", round(rat(f), 2))

# --- figure: GMM walk-forward vs full-fit state
fig, ax = plt.subplots(2, 1, figsize=(9, 3.8), sharex=True)
for a_, s_, t_ in [(ax[0], state_wf, "GMM, refit on past only"), (ax[1], state_full, "GMM, fit on all data (look-ahead)")]:
    a_.fill_between(s_.index, 0, s_.values, step="post", color=C["high"], alpha=.75, lw=0)
    a_.set_yticks([0, 1]); a_.set_yticklabels(["calm", "high vol"]); a_.set_title(t_, loc="left", fontsize=9)
    a_.spines[["top", "right"]].set_visible(False)
    a_.grid(False)
fig.tight_layout(); fig.savefig(ROOT / "figures/regime-gmm-wf-vs-full.png", dpi=150)
