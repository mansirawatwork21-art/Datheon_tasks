"""Customer Cohort & Churn Analytics - IBM Telco Churn dataset.

Run:  python3 churn_analysis.py <path/to/telco_customer_churn.csv> <output_dir>
Produces: results.json, tables (csv) and PNG charts in output_dir.
"""
import sys, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

src = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/telco_customer_churn.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/outputs"
os.makedirs(out, exist_ok=True)

# ---------------------------------------------------------------- 1. load/clean
df = pd.read_csv(src)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)  # 11 new customers, tenure 0
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

N = len(df)
CHURNED = int(df.ChurnFlag.sum())
BASE = CHURNED / N

# ---------------------------------------------------------------- 2. cohorts
tenure_bins = [-1, 12, 24, 36, 48, 60, 72]
tenure_lbl = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
df["TenureCohort"] = pd.cut(df.tenure, tenure_bins, labels=tenure_lbl)

charge_bins = [0, 35, 70, 90, 200]
charge_lbl = ["<$35", "$35-70", "$70-90", "$90+"]
df["ChargeBand"] = pd.cut(df.MonthlyCharges, charge_bins, labels=charge_lbl)

df["ContractS"] = df.Contract.replace({"Month-to-month": "Month-to-Month",
                                       "One year": "1-Year", "Two year": "2-Year"})
df["Pay"] = df.PaymentMethod.str.replace(" (automatic)", " (auto)", regex=False)

addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
for c in addons:
    df[c + "_f"] = df[c].replace({"No internet service": "No internet"})


def seg(col, order=None):
    g = df.groupby(col, observed=True).agg(Customers=("ChurnFlag", "size"), Churned=("ChurnFlag", "sum"),
                                           AvgMonthly=("MonthlyCharges", "mean"))
    g["ChurnRate"] = g.Churned / g.Customers
    g["ShareOfChurn"] = g.Churned / CHURNED
    g["Lift"] = g.ChurnRate / BASE
    if order is not None:
        g = g.reindex(order)
    return g.reset_index()


T = {}
T["tenure"] = seg("TenureCohort", tenure_lbl)
T["contract"] = seg("ContractS", ["Month-to-Month", "1-Year", "2-Year"])
T["payment"] = seg("Pay").sort_values("ChurnRate", ascending=False)
T["charge"] = seg("ChargeBand", charge_lbl)
T["internet"] = seg("InternetService", ["Fiber optic", "DSL", "No"])
for c in ["TechSupport", "OnlineSecurity", "OnlineBackup", "DeviceProtection"]:
    T[c] = seg(c + "_f")
T["senior"] = seg("SeniorCitizen")
T["partner"] = seg("Partner")
T["dependents"] = seg("Dependents")
T["paperless"] = seg("PaperlessBilling")

# ---------------------------------------------------------------- 3. retention matrices
# Snapshot data: 'tenure' = months customer has been (or was) subscribed; Churn=Yes means they left.
# Kaplan-Meier: event = churn at month `tenure`, still-active customers are right-censored at `tenure`.
def km_curve(sub, horizon=66):
    t = sub.tenure.clip(lower=0).values
    e = sub.ChurnFlag.values
    S, s = [1.0], 1.0
    for m in range(1, horizon + 1):
        at_risk = (t >= m).sum()
        events = ((t == m) & (e == 1)).sum()
        if at_risk > 0:
            s *= 1 - events / at_risk
        S.append(s)
    return np.array(S)  # index = month


milestones = [3, 6, 12, 24, 36, 48, 60]  # month 66+ excluded: very few customers at risk, KM tail unstable


def km_matrix(col, order):
    rows = {}
    curves = {}
    for k in order:
        sub = df[df[col] == k]
        S = km_curve(sub)
        curves[k] = S
        rows[k] = {f"M{m}": S[m] for m in milestones}
    return pd.DataFrame(rows).T.loc[order], curves


KM = {}
KM["contract"], CUR_contract = km_matrix("ContractS", ["Month-to-Month", "1-Year", "2-Year"])
KM["payment"], CUR_pay = km_matrix("Pay", list(T["payment"].Pay))
KM["charge"], CUR_charge = km_matrix("ChargeBand", charge_lbl)
KM["internet"], CUR_int = km_matrix("InternetService", ["Fiber optic", "DSL", "No"])
S_all = km_curve(df)
KM["overall"] = pd.DataFrame({"Overall": {f"M{m}": S_all[m] for m in milestones}}).T

# Vintage cohort x contract retention matrix (tenure-at-snapshot vintage; 1 - churn rate within cell)
vint = (df.pivot_table(index="TenureCohort", columns="ContractS", values="ChurnFlag", aggfunc="mean", observed=True)
        .reindex(tenure_lbl)[["Month-to-Month", "1-Year", "2-Year"]])
vint_n = (df.pivot_table(index="TenureCohort", columns="ContractS", values="ChurnFlag", aggfunc="size", observed=True)
          .reindex(tenure_lbl)[["Month-to-Month", "1-Year", "2-Year"]])
vint_ret = 1 - vint

# Hazard (monthly churn probability) by tenure for the Month-to-Month vs. contracted split
# ---------------------------------------------------------------- 4. heatmaps
heat_tier_cols = ["TechSupport", "OnlineSecurity", "OnlineBackup", "DeviceProtection"]
tier_rows = []
for inet in ["Fiber optic", "DSL"]:
    for c in heat_tier_cols:
        for v in ["Yes", "No"]:
            sub = df[(df.InternetService == inet) & (df[c] == v)]
            tier_rows.append({"Internet": inet, "Addon": c, "Has": v, "n": len(sub), "ChurnRate": sub.ChurnFlag.mean()})
tier = pd.DataFrame(tier_rows)
heat_tier = tier.pivot_table(index="Addon", columns=["Internet", "Has"], values="ChurnRate").loc[heat_tier_cols]
heat_tier = heat_tier[[("Fiber optic", "No"), ("Fiber optic", "Yes"), ("DSL", "No"), ("DSL", "Yes")]]

heat_cx = df.pivot_table(index="InternetService", columns="ContractS", values="ChurnFlag", aggfunc="mean", observed=True) \
    .reindex(["Fiber optic", "DSL", "No"])[["Month-to-Month", "1-Year", "2-Year"]]
heat_cx_n = df.pivot_table(index="InternetService", columns="ContractS", values="ChurnFlag", aggfunc="size", observed=True) \
    .reindex(["Fiber optic", "DSL", "No"])[["Month-to-Month", "1-Year", "2-Year"]]

heat_pc = df.pivot_table(index="Pay", columns="ContractS", values="ChurnFlag", aggfunc="mean", observed=True)[
    ["Month-to-Month", "1-Year", "2-Year"]].reindex(list(T["payment"].Pay))

# ---------------------------------------------------------------- 5. high risk segments
fiber = df.InternetService == "Fiber optic"
mtm = df.Contract == "Month-to-month"
echeck = df.PaymentMethod == "Electronic check"
notech = df.TechSupport == "No"
nosec = df.OnlineSecurity == "No"
new = df.tenure <= 12
seg_defs = {
    "Fiber optic, no Tech Support": fiber & notech,
    "Fiber optic, no Tech Support & no Online Security": fiber & notech & nosec,
    "Month-to-month + Electronic check": mtm & echeck,
    "Month-to-month, tenure 0-12 months": mtm & new,
    "Month-to-month + Fiber, no Tech Support": mtm & fiber & notech,
    "Month-to-month + Fiber + no Tech Support + Electronic check": mtm & fiber & notech & echeck,
    "Month-to-month + Fiber + no Tech Support + tenure 0-12": mtm & fiber & notech & new,
    "Month-to-month + Monthly charge > $70, no add-on protection": mtm & (df.MonthlyCharges > 70) & notech & nosec,
    "Senior citizen + Fiber + Month-to-month": (df.SeniorCitizen == 1) & fiber & mtm,
}
low_defs = {
    "2-Year contract": df.Contract == "Two year",
    "2-Year contract + Auto-pay": (df.Contract == "Two year") & df.PaymentMethod.str.contains("automatic"),
    "Fiber optic WITH Tech Support": fiber & (df.TechSupport == "Yes"),
    "DSL with Tech Support + Online Security": (df.InternetService == "DSL") & (df.TechSupport == "Yes") & (df.OnlineSecurity == "Yes"),
}
rows = []
for name, mask in {**seg_defs, **low_defs}.items():
    sub = df[mask]
    rows.append({"Segment": name, "Type": "High-risk" if name in seg_defs else "Low-risk benchmark",
                 "Customers": len(sub), "Churned": int(sub.ChurnFlag.sum()), "ChurnRate": sub.ChurnFlag.mean(),
                 "ShareOfAllChurn": sub.ChurnFlag.sum() / CHURNED, "ShareOfBase": len(sub) / N,
                 "Lift": sub.ChurnFlag.mean() / BASE,
                 "AvgMonthly": sub.MonthlyCharges.mean(),
                 "MonthlyRevenueAtRisk": sub.MonthlyCharges.sum() if name in seg_defs else np.nan})
RISK = pd.DataFrame(rows).sort_values(["Type", "ChurnRate"], ascending=[True, False]).reset_index(drop=True)

# ---------------------------------------------------------------- 6. churn probability model
feat = df.drop(columns=["TotalCharges", "customerID", "Churn", "ChurnFlag", "TenureCohort", "ChargeBand", "ContractS", "Pay"] +
               [c + "_f" for c in addons])
cats = feat.select_dtypes(include=["object", "string"]).columns.tolist()
X = pd.get_dummies(feat, columns=cats, drop_first=False)
# drop redundant dummies
drop = [c for c in X.columns if c.endswith("_No internet service") or c.endswith("_No phone service")]
X = X.drop(columns=drop)
for c in [c for c in X.columns if c.endswith("_No") and c.replace("_No", "_Yes") in X.columns]:
    X = X.drop(columns=c)
X = X.astype(float)
y = df.ChurnFlag
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Xtr), ytr)
auc = roc_auc_score(yte, lr.predict_proba(sc.transform(Xte))[:, 1])
coef = pd.Series(lr.coef_[0], index=X.columns).sort_values()
drivers = pd.DataFrame({"Feature": coef.index, "StdCoef": coef.values, "OddsRatioPerSD": np.exp(coef.values)})
drivers = drivers.reindex(drivers.StdCoef.abs().sort_values(ascending=False).index).reset_index(drop=True)

# full-population churn probability (for scoring table)
df["ChurnProbability"] = lr.predict_proba(sc.transform(X))[:, 1]
df["RiskTier"] = pd.cut(df.ChurnProbability, [0, 0.2, 0.4, 0.6, 1.0], labels=["Low (<20%)", "Medium (20-40%)", "High (40-60%)", "Critical (60%+)"])
RT = df.groupby("RiskTier", observed=True).agg(Customers=("ChurnFlag", "size"), ActualChurnRate=("ChurnFlag", "mean"),
                                                AvgProb=("ChurnProbability", "mean"),
                                                MonthlyRevenue=("MonthlyCharges", "sum")).reset_index()

# ---------------------------------------------------------------- 7. charts
BLUE, ORANGE, AQUA, YELLOW, MAG, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
seq = LinearSegmentedColormap.from_list("seq", ["#eaf2fc", "#9cc3f0", "#2a78d6", "#0f3f7a"])
seq_churn = LinearSegmentedColormap.from_list("seqc", ["#fbefe9", "#f5b595", "#eb6834", "#8a2f0a"])


def save(fig, name):
    fig.savefig(os.path.join(out, name), dpi=170, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


# 7a cohort retention matrix (KM milestones by contract) -------------
def matrix_plot(M, title, sub, fname, cmap, fmt="{:.0%}", vmin=None, vmax=None, note=None, nmat=None, figsize=(9.5, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    vals = M.values.astype(float)
    im = ax.imshow(vals, cmap=cmap, aspect="auto", vmin=vmin if vmin is not None else np.nanmin(vals),
                   vmax=vmax if vmax is not None else np.nanmax(vals))
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    lo, hi = im.get_clim()
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v): continue
            dark = (v - lo) / (hi - lo + 1e-9) > 0.55
            txt = fmt.format(v)
            if nmat is not None and not np.isnan(nmat[i, j]):
                txt += f"\nn={int(nmat[i, j]):,}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5, color="white" if dark else INK, fontweight="bold" if nmat is None else "normal")
    ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True); ax.set_yticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=2); ax.tick_params(which="minor", length=0)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=24)
    ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
    if note: fig.text(0.01, -0.02, note, fontsize=8, color=INK2, ha="left")
    save(fig, fname)


km_c = KM["contract"].copy(); km_c.columns = [f"Month {m}" for m in milestones]
matrix_plot(km_c, "Cohort retention matrix: % of customers still active", "Kaplan-Meier survival by contract type at tenure milestones",
            "01_retention_matrix_contract.png", seq, vmin=0.3, vmax=1.0,
            note=f"Source: IBM Telco Churn, n={N:,}. Active customers are right-censored at current tenure.")

km_p = KM["payment"].copy(); km_p.columns = [f"Month {m}" for m in milestones]
km_ch = KM["charge"].copy(); km_ch.columns = [f"Month {m}" for m in milestones]
km_all = pd.concat([km_c, km_p.rename(index=lambda s: "Pay: " + s), km_ch.rename(index=lambda s: "Charge: " + s)])
matrix_plot(km_all, "Retention by contract, payment method and monthly charge band", "Share of customers still active at each tenure milestone (Kaplan-Meier)",
            "02_retention_matrix_segments.png", seq, vmin=0.3, vmax=1.0, figsize=(10, 7))

matrix_plot(vint_ret, "Vintage cohort retention: tenure cohort x contract", "Share of each cohort's customers retained at snapshot (1 - churn rate); n shown in each cell",
            "03_vintage_cohort_contract.png", seq, vmin=0.3, vmax=1.0, nmat=vint_n.values.astype(float), figsize=(8.5, 5.2))

# 7b retention curves ------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
months = np.arange(0, 67)
for ax, (cur, ttl, cols) in zip(axes, [(CUR_contract, "By contract type", [ORANGE, BLUE, AQUA]),
                                        (CUR_pay, "By payment method", [RED, YELLOW, BLUE, VIOLET]),
                                        (CUR_charge, "By monthly charge band", [AQUA, BLUE, YELLOW, ORANGE])]):
    for (k, S), col in zip(cur.items(), cols):
        ax.plot(months, S, color=col, lw=2.2, label=k)
        ax.plot(66, S[66], "o", color=col, ms=7, mec=SURF, mew=2)
    ax.plot(months, S_all, color=INK2, lw=1.2, ls="--", label="All customers")
    ax.set_title(ttl, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("Tenure (months)"); ax.set_ylim(0.2, 1.02); ax.set_xlim(0, 68)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
axes[0].set_ylabel("Customers retained")
fig.suptitle("Retention curves (Kaplan-Meier): contract type is the dominant separator", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.03)
save(fig, "04_retention_curves.png")

# 7c churn by tenure cohort + baseline -------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.4))
tt = T["tenure"]
bars = ax.bar(tt.TenureCohort.astype(str), tt.ChurnRate, color=BLUE, width=0.62)
for b, r, n in zip(bars, tt.ChurnRate, tt.Customers):
    ax.text(b.get_x() + b.get_width() / 2, r + 0.008, f"{r:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax.text(b.get_x() + b.get_width() / 2, 0.01, f"n={n:,}", ha="center", fontsize=8, color="white")
ax.axhline(BASE, color=ORANGE, lw=1.6, ls="--"); ax.text(5.45, BASE + 0.008, f"Baseline {BASE:.1%}", color=ORANGE, ha="right", fontsize=9.5, fontweight="bold")
ax.set_xlabel("Tenure cohort (months)"); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("Churn rate by tenure cohort", loc="left", fontsize=13, fontweight="bold")
save(fig, "05_churn_by_tenure_cohort.png")

# 7d churn heatmap by service tier -----------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
ht = heat_tier.copy()
ax = axs[0]
im = ax.imshow(ht.values, cmap=seq_churn, vmin=0.05, vmax=0.5, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels(["Fiber\nNO add-on", "Fiber\nHAS add-on", "DSL\nNO add-on", "DSL\nHAS add-on"])
ax.set_yticks(range(4)); ax.set_yticklabels(["Tech Support", "Online Security", "Online Backup", "Device Protection"])
for i in range(4):
    for j in range(4):
        v = ht.values[i, j]
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=11, fontweight="bold", color="white" if v > 0.3 else INK)
ax.set_xticks(np.arange(-.5, 4, 1), minor=True); ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
ax.grid(which="minor", color=SURF, lw=2); ax.tick_params(which="both", length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Churn by internet tier and protection add-ons", loc="left", fontsize=12, fontweight="bold")
ax = axs[1]
hc = heat_cx
ax.imshow(hc.values, cmap=seq_churn, vmin=0.0, vmax=0.5, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(hc.columns); ax.set_yticks(range(3)); ax.set_yticklabels(["Fiber optic", "DSL", "No internet"])
for i in range(3):
    for j in range(3):
        v = hc.values[i, j]
        ax.text(j, i, f"{v:.0%}\nn={int(heat_cx_n.values[i, j]):,}", ha="center", va="center", fontsize=10, color="white" if v > 0.3 else INK)
ax.set_xticks(np.arange(-.5, 3, 1), minor=True); ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
ax.grid(which="minor", color=SURF, lw=2); ax.tick_params(which="both", length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Churn by service tier and contract", loc="left", fontsize=12, fontweight="bold")
fig.suptitle("Churn heatmap by service tier (cell = % of customers churned)", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.04)
save(fig, "06_churn_heatmap_service_tier.png")

# 7e payment x contract heatmap --------------------------------------
matrix_plot(heat_pc.rename(index=lambda s: s), "Churn rate: payment method x contract", "Electronic check on a month-to-month plan is the weakest cell",
            "07_churn_payment_x_contract.png", seq_churn, vmin=0.0, vmax=0.5, figsize=(8.5, 4.2))

# 7f high risk segments ----------------------------------------------
hr = RISK[RISK.Type == "High-risk"].sort_values("ChurnRate")
fig, ax = plt.subplots(figsize=(10.5, 5.2))
ax.barh(hr.Segment, hr.ChurnRate, color=ORANGE, height=0.6)
for y_, (r, n) in enumerate(zip(hr.ChurnRate, hr.Customers)):
    ax.text(r + 0.008, y_, f"{r:.0%}  (n={n:,})", va="center", fontsize=9)
ax.axvline(BASE, color=INK2, lw=1.2, ls="--"); ax.text(BASE + 0.004, -0.75, f"Baseline {BASE:.1%}", fontsize=8.5, color=INK2)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); ax.set_xlim(0, 0.8)
ax.set_title("High-risk segments", loc="left", fontsize=13, fontweight="bold")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
save(fig, "08_high_risk_segments.png")

# 7g drivers ----------------------------------------------------------
top = pd.concat([drivers.head(8), drivers.tail(0)])
topn = drivers.iloc[:14].sort_values("StdCoef")
fig, ax = plt.subplots(figsize=(9, 5.4))
ax.barh(topn.Feature, topn.StdCoef, color=[ORANGE if v > 0 else BLUE for v in topn.StdCoef], height=0.6)
ax.axvline(0, color=INK2, lw=1)
ax.set_xlabel("Standardized logistic coefficient (right = raises churn probability, left = lowers it)")
ax.set_title(f"Churn drivers (logistic regression, hold-out AUC {auc:.2f})", loc="left", fontsize=13, fontweight="bold")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
save(fig, "09_churn_drivers.png")

# ---------------------------------------------------------------- 8. dump
tables = {
    "Summary": pd.DataFrame({"Metric": ["Total customers", "Churned customers", "Baseline churn rate", "Avg tenure (months)",
                                        "Avg monthly charge", "Churned customers' avg monthly charge", "Monthly revenue lost to churned customers",
                                        "Model hold-out AUC"],
                             "Value": [N, CHURNED, BASE, df.tenure.mean(), df.MonthlyCharges.mean(),
                                       df[df.ChurnFlag == 1].MonthlyCharges.mean(), df[df.ChurnFlag == 1].MonthlyCharges.sum(), auc]}),
    "Tenure cohorts": T["tenure"], "Contract": T["contract"], "Payment": T["payment"], "Charge bands": T["charge"],
    "Internet": T["internet"], "TechSupport": T["TechSupport"], "OnlineSecurity": T["OnlineSecurity"],
    "OnlineBackup": T["OnlineBackup"], "DeviceProtection": T["DeviceProtection"],
    "Senior": T["senior"], "Partner": T["partner"], "Dependents": T["dependents"], "Paperless": T["paperless"],
    "KM contract": KM["contract"].reset_index(names="Segment"), "KM payment": KM["payment"].reset_index(names="Segment"),
    "KM charge": KM["charge"].reset_index(names="Segment"), "KM internet": KM["internet"].reset_index(names="Segment"),
    "Vintage retention": vint_ret.reset_index(), "Vintage n": vint_n.reset_index(),
    "Heat tier": heat_tier.copy().set_axis([f"{a} | {'has' if b == 'Yes' else 'no'}" for a, b in heat_tier.columns], axis=1).reset_index(),
    "Heat contract x internet": heat_cx.reset_index(), "Heat payment x contract": heat_pc.reset_index(),
    "Risk segments": RISK, "Drivers": drivers, "Risk tiers": RT,
}
os.makedirs(os.path.join(out, "tables"), exist_ok=True)
for k, v in tables.items():
    v.to_csv(os.path.join(out, "tables", k.replace(" ", "_").replace("|", "") + ".csv"), index=False)
df[["customerID", "tenure", "TenureCohort", "ContractS", "Pay", "InternetService", "MonthlyCharges", "ChargeBand",
    "Churn", "ChurnProbability", "RiskTier"]].to_csv(os.path.join(out, "tables", "customer_scores.csv"), index=False)
print("baseline", round(BASE, 4), "auc", round(auc, 3))
for k in ["tenure", "contract", "payment", "charge", "internet", "TechSupport", "OnlineSecurity"]:
    print("\n", k); print(T[k].round(4).to_string(index=False))
print("\nKM contract\n", KM["contract"].round(3)); print("\nKM payment\n", KM["payment"].round(3)); print("\nKM charge\n", KM["charge"].round(3))
print("\nvintage ret\n", vint_ret.round(3)); print(vint_n)
print("\nRISK\n", RISK.round(3).to_string(index=False))
print("\nDrivers\n", drivers.head(12).round(3).to_string(index=False))
print("\nRisk tiers\n", RT.round(3).to_string(index=False))
print("\nheat tier\n", heat_tier.round(3)); print(heat_cx.round(3)); print(heat_pc.round(3))
