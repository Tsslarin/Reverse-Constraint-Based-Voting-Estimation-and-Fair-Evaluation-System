import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

CSV_PATH = "dwts_with_week_normalized_shares.csv"   # 改成你的路径
EARLY_WEEKS = 3                                     # 前几周敏感性分析：3周
N_BOOT = 5000                                       # bootstrap次数：5000足够稳
SEED = 0

# ========== 1) 读数据 + 清洗 ==========
df = pd.read_csv(CSV_PATH)

need_cols = ["season", "week", "contestant", "celebrity_age_during_season", "judge_share"]
d = df[need_cols].copy()

# 转数值（兜底处理）
d["week"] = pd.to_numeric(d["week"], errors="coerce")
d["celebrity_age_during_season"] = pd.to_numeric(d["celebrity_age_during_season"], errors="coerce")
d["judge_share"] = pd.to_numeric(d["judge_share"], errors="coerce")

# 去缺失
d = d.dropna(subset=["season", "week", "contestant", "celebrity_age_during_season", "judge_share"])

# 同一 (season, week, contestant) 若有重复行（例如不同来源/重复记录），先合并成一条
week_level = (d.groupby(["season", "week", "contestant"], as_index=False)
                .agg(age=("celebrity_age_during_season", "first"),
                     judge_share=("judge_share", "mean"))
             )

# ========== 2) 聚合函数：名人-赛季 -> 1个点 ==========
def aggregate_to_celebrity_season(week_df: pd.DataFrame, early_weeks: int | None = None) -> pd.DataFrame:
    tmp = week_df.copy()
    if early_weeks is not None:
        tmp = tmp[tmp["week"] <= early_weeks]

    agg = (tmp.groupby(["season", "contestant"], as_index=False)
             .agg(age=("age", "first"),
                  judge_share_mean=("judge_share", "mean"),
                  weeks=("judge_share", "size"))
          )
    # 若你担心只出现1周的人太噪声，可启用下一行（可选）
    # agg = agg[agg["weeks"] >= 2]
    return agg

# ========== 3) bootstrap 95% 置信区间 ==========
def bootstrap_ci(x, y, corr_func, n_boot=5000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = np.empty(n_boot, dtype=float)
    idx = np.arange(n)
    for b in range(n_boot):
        sample = rng.choice(idx, size=n, replace=True)
        stats[b] = corr_func(x[sample], y[sample])
    lo = np.quantile(stats, alpha/2)
    hi = np.quantile(stats, 1 - alpha/2)
    return lo, hi

def pearson_only(x, y):
    # 更快：直接用numpy算r
    return np.corrcoef(x, y)[0, 1]

def spearman_only(x, y):
    # 用scipy算spearman（返回ρ）
    return spearmanr(x, y).correlation

# ========== 4) 计算 + 画图 ==========
def analyze_and_plot(agg: pd.DataFrame, title_suffix: str):
    x = agg["age"].to_numpy()
    y = agg["judge_share_mean"].to_numpy()

    # 相关系数 + p值（用于报告）
    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)

    # bootstrap 95%CI（更直观地表达强度范围）
    ci_p = bootstrap_ci(x, y, pearson_only, n_boot=N_BOOT, seed=SEED)
    ci_s = bootstrap_ci(x, y, spearman_only, n_boot=N_BOOT, seed=SEED)

    print(f"\n=== {title_suffix} ===")
    print(f"N = {len(agg)} (名人-赛季点)")
    print(f"Pearson r = {r_p:.3f}, p = {p_p:.3g}, 95%CI = [{ci_p[0]:.3f}, {ci_p[1]:.3f}]")
    print(f"Spearman ρ = {r_s:.3f}, p = {p_s:.3g}, 95%CI = [{ci_s[0]:.3f}, {ci_s[1]:.3f}]")

    # 散点图 + 线性拟合线（看方向最直观）
    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    plt.plot(xx, m*xx + b)

    plt.xlabel("celebrity_age_during_season")
    plt.ylabel("mean judge_share (per season)")
    plt.title(f"Age vs Judge Share (aggregated) | {title_suffix}\n"
              f"Pearson r={r_p:.3f}, Spearman ρ={r_s:.3f}")
    plt.show()

    return {"N": len(agg),
            "pearson_r": r_p, "pearson_p": p_p, "pearson_ci": ci_p,
            "spearman_rho": r_s, "spearman_p": p_s, "spearman_ci": ci_s}

# 全季（你现在的主结果）
agg_all = aggregate_to_celebrity_season(week_level, early_weeks=None)
res_all = analyze_and_plot(agg_all, "All weeks (full season)")

# 敏感性分析：只用前3周
agg_early = aggregate_to_celebrity_season(week_level, early_weeks=EARLY_WEEKS)
res_early = analyze_and_plot(agg_early, f"Early weeks only (week <= {EARLY_WEEKS})")

# 导出聚合表，方便你写报告/复现（可选）
agg_all.to_csv("agg_age_judgeshare_fullseason.csv", index=False)
agg_early.to_csv(f"agg_age_judgeshare_week_le_{EARLY_WEEKS}.csv", index=False)
print("\n已导出聚合表：agg_age_judgeshare_fullseason.csv 和 early-week 版本")
