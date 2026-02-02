import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

CSV_PATH = "dwts_with_week_normalized_shares_已填normal.csv"
N_BOOT = 10000
SEED = 1

# 1) 读取 + 清洗（只取需要的列）
df = pd.read_csv(CSV_PATH)
d = df[["season", "contestant", "celebrity_age_during_season", "normal"]].copy()

d["celebrity_age_during_season"] = pd.to_numeric(d["celebrity_age_during_season"], errors="coerce")
d["normal"] = pd.to_numeric(d["normal"], errors="coerce")
d = d.dropna(subset=["season", "contestant", "celebrity_age_during_season", "normal"])

# 可选：处理异常 normal（比如 -0.1）
# 方案A：直接删除异常
# d = d[(d["normal"] >= 0) & (d["normal"] <= 1)]

# 方案B：裁剪到[0,1]（更温和）
d["normal"] = d["normal"].clip(0, 1)

# 2) 关键：聚合到“名人-赛季”唯一点（去重）
agg = (d.groupby(["season", "contestant"], as_index=False)
         .agg(age=("celebrity_age_during_season", "first"),
              normal=("normal", "first"))
      )

x = agg["age"].to_numpy()
y = agg["normal"].to_numpy()

# 3) 相关系数（Spearman为主，Pearson为辅；可再报Kendall）
r_p, p_p = pearsonr(x, y)
r_s, p_s = spearmanr(x, y)
t_k, p_k = kendalltau(x, y)

print(f"N = {len(agg)}（名人-赛季）")
print(f"Pearson r = {r_p:.3f}, p = {p_p:.3g}")
print(f"Spearman ρ = {r_s:.3f}, p = {p_s:.3g}")
print(f"Kendall τ = {t_k:.3f}, p = {p_k:.3g}")

# 4) bootstrap 95%CI（不需要正态假设）
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

ci_p = bootstrap_ci(x, y, pearson_only, n_boot=N_BOOT, seed=SEED)
ci_s = bootstrap_ci(x, y, spearman_only, n_boot=N_BOOT, seed=SEED)

print(f"Pearson 95% CI = [{ci_p[0]:.3f}, {ci_p[1]:.3f}]")
print(f"Spearman 95% CI = [{ci_s[0]:.3f}, {ci_s[1]:.3f}]")

# 5) 图：散点 + 拟合线（看方向最直观）
plt.figure()
plt.scatter(x, y, alpha=0.6)

m, b = np.polyfit(x, y, 1)
xx = np.linspace(x.min(), x.max(), 200)
plt.plot(xx, m*xx + b)

plt.xlabel("celebrity_age_during_season")
plt.ylabel("normal (normalized season rank)")
plt.title(f"Age vs Normalized Rank (celebrity-season)\n"
          f"Pearson r={r_p:.3f}, Spearman ρ={r_s:.3f}")
plt.show()

# 6) 导出聚合点（写报告/复现用）
agg.to_csv("agg_age_normal.csv", index=False)
print("已导出：agg_age_normal.csv")
