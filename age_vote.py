import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

CSV_PATH = "dwts_with_week_normalized_shares.csv"

EARLY_WEEKS = 3      # 敏感性分析：只用前3周（你可改成 1/2/4/5 等）
N_BOOT = 10000       # bootstrap 次数
SEED = 1

# ===== 1) 读取 + 清洗 =====
df = pd.read_csv(CSV_PATH)

d = df[["season", "week", "contestant", "celebrity_age_during_season", "vote_share"]].copy()
d["week"] = pd.to_numeric(d["week"], errors="coerce")
d["celebrity_age_during_season"] = pd.to_numeric(d["celebrity_age_during_season"], errors="coerce")
d["vote_share"] = pd.to_numeric(d["vote_share"], errors="coerce")
d = d.dropna(subset=["season", "week", "contestant", "celebrity_age_during_season", "vote_share"])

# （保险）同一 season-week-contestant 若有重复行，先合并
week_level = (d.groupby(["season", "week", "contestant"], as_index=False)
                .agg(age=("celebrity_age_during_season", "first"),
                     vote_share=("vote_share", "mean"))
             )

# ===== 2) 工具函数：聚合到“名人-赛季” =====
def aggregate_celebrity_season(week_df: pd.DataFrame, early_weeks: int | None = None) -> pd.DataFrame:
    tmp = week_df.copy()
    if early_weeks is not None:
        tmp = tmp[tmp["week"] <= early_weeks]

    agg = (tmp.groupby(["season", "contestant"], as_index=False)
             .agg(age=("age", "first"),
                  vote_share_mean=("vote_share", "mean"),
                  weeks=("vote_share", "size"))
          )
    # 可选：过滤仅出现 1 周的名人-赛季点（如果你担心噪声）
    # agg = agg[agg["weeks"] >= 2]
    return agg

# ===== 3) bootstrap 95% CI =====
def bootstrap_ci(x, y, corr_func, n_boot=10000, seed=1, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = np.arange(n)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        stats[b] = corr_func(x[samp], y[samp])
    return np.quantile(stats, alpha/2), np.quantile(stats, 1-alpha/2)

def pearson_only(a, b):
    return np.corrcoef(a, b)[0, 1]

def spearman_only(a, b):
    return spearmanr(a, b).correlation

# ===== 4) 计算并输出（含 CI） =====
def analyze(agg: pd.DataFrame, label: str):
    x = agg["age"].to_numpy()
    y = agg["vote_share_mean"].to_numpy()

    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)

    ci_p = bootstrap_ci(x, y, pearson_only, n_boot=N_BOOT, seed=SEED)
    ci_s = bootstrap_ci(x, y, spearman_only, n_boot=N_BOOT, seed=SEED)

    out = {
        "setting": label,
        "N": len(agg),
        "Pearson_r": r_p, "Pearson_p": p_p, "Pearson_CI_low": ci_p[0], "Pearson_CI_high": ci_p[1],
        "Spearman_rho": r_s, "Spearman_p": p_s, "Spearman_CI_low": ci_s[0], "Spearman_CI_high": ci_s[1],
        "avg_weeks_per_point": agg["weeks"].mean()
    }
    return out, x, y

# 全季
agg_all = aggregate_celebrity_season(week_level, early_weeks=None)
res_all, x_all, y_all = analyze(agg_all, "All weeks (full season)")

# 前K周（敏感性分析）
agg_early = aggregate_celebrity_season(week_level, early_weeks=EARLY_WEEKS)
res_early, x_e, y_e = analyze(agg_early, f"Early weeks only (week <= {EARLY_WEEKS})")

# ===== 5) 打印汇总表 =====
summary = pd.DataFrame([res_all, res_early])
pd.set_option("display.max_columns", None)
print(summary[[
    "setting", "N", "avg_weeks_per_point",
    "Pearson_r", "Pearson_p", "Pearson_CI_low", "Pearson_CI_high",
    "Spearman_rho", "Spearman_p", "Spearman_CI_low", "Spearman_CI_high"
]])

# ===== 6) 画图：两张散点图对照（方向是否一致一眼看清）=====
def scatter_with_fit(x, y, title):
    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    plt.plot(xx, m*xx + b)
    plt.xlabel("celebrity_age_during_season")
    plt.ylabel("mean vote_share (per season)")
    plt.title(title)
    plt.show()

scatter_with_fit(
    x_all, y_all,
    f"Age vs Vote Share (aggregated) | Full season\nPearson r={res_all['Pearson_r']:.3f}, Spearman ρ={res_all['Spearman_rho']:.3f}"
)

scatter_with_fit(
    x_e, y_e,
    f"Age vs Vote Share (aggregated) | Week <= {EARLY_WEEKS}\nPearson r={res_early['Pearson_r']:.3f}, Spearman ρ={res_early['Spearman_rho']:.3f}"
)

# ===== 7) 导出聚合后的点（可选：写报告/复现用）=====
agg_all.to_csv("agg_age_votes_fullseason.csv", index=False)
agg_early.to_csv(f"agg_age_votes_week_le_{EARLY_WEEKS}.csv", index=False)
print("\n已导出：agg_age_votes_fullseason.csv 以及 early-week 版本")
