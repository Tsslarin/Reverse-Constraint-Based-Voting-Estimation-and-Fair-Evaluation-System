import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# 1) 读取数据
df = pd.read_csv("dwts_with_week_normalized_shares.csv")

# 2) 只保留需要的列
d = df[["season", "week", "contestant", "celebrity_age_during_season", "judge_share"]].copy()

# 3) 转数值 + 去缺失
d["celebrity_age_during_season"] = pd.to_numeric(d["celebrity_age_during_season"], errors="coerce")
d["judge_share"] = pd.to_numeric(d["judge_share"], errors="coerce")
d = d.dropna(subset=["season", "week", "contestant", "celebrity_age_during_season", "judge_share"])

# （可选保险）同一 season-week-contestant 若有重复行，就先合并
week_level = (d.groupby(["season", "week", "contestant"], as_index=False)
                .agg(age=("celebrity_age_during_season", "first"),
                     judge_share=("judge_share", "mean"))
             )

# 4) 关键：聚合到 “名人-赛季”
agg = (week_level.groupby(["season", "contestant"], as_index=False)
       .agg(age=("age", "first"),
            judge_share_mean=("judge_share", "mean"),
            weeks=("judge_share", "size"))
      )

# 5) 相关系数（回答“有没有关系、方向是什么”）
x = agg["age"].to_numpy()
y = agg["judge_share_mean"].to_numpy()

r_p, p_p = pearsonr(x, y)
r_s, p_s = spearmanr(x, y)

print(f"聚合后样本量 N = {len(agg)}（名人-赛季）")
print(f"Pearson r = {r_p:.3f}, p = {p_p:.3g}")
print(f"Spearman ρ = {r_s:.3f}, p = {p_s:.3g}")

# 6) 可视化：散点 + 线性拟合线（看方向）
plt.figure()
plt.scatter(x, y, alpha=0.6)
m, b = np.polyfit(x, y, 1)
xx = np.linspace(x.min(), x.max(), 200)
plt.plot(xx, m*xx + b)

plt.xlabel("celebrity_age_during_season")
plt.ylabel("mean judge_share (per season)")
plt.title("Age vs Judge Share (aggregated to celebrity-season)")
plt.show()

# 7) 导出聚合表
agg.to_csv("agg_age_judgeshare.csv", index=False)
print("已导出：agg_age_judgeshare.csv")
