import pandas as pd
import numpy as np

# ========== 1) 读取数据 ==========
DATA_PATH = r"dwts_with_week_normalized_shares_已填normal.csv"
OUT_XLSX  = r"dwts_vote_method_comparison.xlsx"

df = pd.read_csv(DATA_PATH)

# 你的文件里有一列 method（rank/percent）是重复行，我们只保留唯一的周数据
if "method" in df.columns:
    base = df.drop(columns=["method"]).drop_duplicates()
else:
    base = df.copy()

# 清理：去掉空 contestant 行
base = base.dropna(subset=["contestant"]).copy()
base["contestant"] = base["contestant"].astype(str)

# 强制数值化
base["judge_total"] = pd.to_numeric(base["judge_total"], errors="coerce").fillna(0)
base["estimated_votes_total"] = pd.to_numeric(base["estimated_votes_total"], errors="coerce").fillna(0)

# ========== 2) 推断每周淘汰人数 elim_k ==========
# 思路：看这一周人数 n 和下一次出现的周人数 n_next，差多少就淘汰多少
wk_n = (
    base.groupby(["season", "week"])["contestant"]
    .nunique()
    .reset_index(name="n")
    .sort_values(["season", "week"])
)
wk_n["n_next"] = wk_n.groupby("season")["n"].shift(-1)
wk_n["elim_k"] = (wk_n["n"] - wk_n["n_next"]).fillna(0).astype(int)

# ========== 3) 两种方法的“本周综合排序” ==========
def rank_desc(s: pd.Series) -> pd.Series:
    """1=最好，数字越大越差（并列用 average）"""
    return s.rank(method="average", ascending=False)

def compute_week_scores(dfw: pd.DataFrame, method: str) -> pd.DataFrame:
    out = dfw[["season", "week", "contestant", "judge_total", "estimated_votes_total"]].copy()

    if method == "rank":
        out["judge_rank"] = rank_desc(out["judge_total"])
        out["vote_rank"]  = rank_desc(out["estimated_votes_total"])
        out["rank_sum"]   = out["judge_rank"] + out["vote_rank"]  # 越小越好
        out["combined"]   = out["rank_sum"]

        # 排序：从“最差”到“最好”
        # tie-break：评委更低更差 -> 更容易被淘汰
        out = out.sort_values(["rank_sum", "judge_total", "estimated_votes_total"],
                              ascending=[False, True, True])

    elif method == "percent":
        jsum = out["judge_total"].sum()
        vsum = out["estimated_votes_total"].sum()

        out["judge_share"] = out["judge_total"] / jsum if jsum != 0 else 0
        out["vote_share"]  = out["estimated_votes_total"] / vsum if vsum != 0 else 0
        out["combined"]    = out["judge_share"] + out["vote_share"]  # 越大越好

        out["judge_rank"] = rank_desc(out["judge_total"])
        out["vote_rank"]  = rank_desc(out["estimated_votes_total"])

        # 排序：从“最差”到“最好”
        out = out.sort_values(["combined", "judge_total", "estimated_votes_total"],
                              ascending=[True, True, True])
    else:
        raise ValueError("method must be 'rank' or 'percent'")

    return out

def decide_eliminated(scored: pd.DataFrame, k: int) -> list:
    if k <= 0:
        return []
    return scored.head(k)["contestant"].tolist()

def decide_eliminated_with_judges_save(scored: pd.DataFrame, k: int) -> list:
    """
    judges save 模拟：
    - k==1：取 bottom2（最差两名），淘汰评委分更低的那个
    - k>1：先直接淘汰最差 (k-1) 名，再对剩下的 bottom2 用 judges save 决最后 1 个
    """
    if k <= 0:
        return []

    scored = scored.copy()
    eliminated = []

    if k == 1:
        bottom2 = scored.head(2)
        if len(bottom2) < 2:
            return bottom2["contestant"].tolist()
        bottom2 = bottom2.sort_values(["judge_total", "combined"], ascending=[True, True])
        return [bottom2.iloc[0]["contestant"]]

    # k > 1
    eliminated += scored.head(k - 1)["contestant"].tolist()
    remaining = scored.iloc[k - 1:].copy()

    bottom2 = remaining.head(2)
    if len(bottom2) < 2:
        eliminated += bottom2["contestant"].tolist()
        return eliminated

    bottom2 = bottom2.sort_values(["judge_total", "combined"], ascending=[True, True])
    eliminated.append(bottom2.iloc[0]["contestant"])
    return eliminated

def week_delta(scored: pd.DataFrame, eliminated: list) -> float:
    """
    衡量“更偏粉丝”的一个直观指标：
    delta = 被淘汰者(粉丝名次 - 评委名次) 的平均
    delta 越大 -> 淘汰的人越像“粉丝更不喜欢的人” -> 越偏粉丝
    """
    if not eliminated:
        return np.nan
    tmp = scored[scored["contestant"].isin(eliminated)]
    if len(tmp) == 0:
        return np.nan
    return float((tmp["vote_rank"] - tmp["judge_rank"]).mean())

# ========== 4) 逐周计算：谁会被淘汰 ==========
rows = []
for (s, w), dfw in base.groupby(["season", "week"]):
    k = int(wk_n.loc[(wk_n["season"] == s) & (wk_n["week"] == w), "elim_k"].iloc[0])

    scored_rank = compute_week_scores(dfw, "rank")
    scored_pct  = compute_week_scores(dfw, "percent")

    elim_rank = decide_eliminated(scored_rank, k)
    elim_pct  = decide_eliminated(scored_pct,  k)

    elim_rank_js = decide_eliminated_with_judges_save(scored_rank, k)
    elim_pct_js  = decide_eliminated_with_judges_save(scored_pct,  k)

    # “粉丝最差者”是谁（粉丝名次最大）
    fan_worst_rank = scored_rank.loc[scored_rank["vote_rank"].idxmax(), "contestant"] if len(scored_rank) else None
    fan_worst_pct  = scored_pct.loc[scored_pct["vote_rank"].idxmax(),  "contestant"] if len(scored_pct)  else None

    rows.append({
        "season": s,
        "week": w,
        "elim_k": k,
        "eliminated_rank": "; ".join(elim_rank),
        "eliminated_percent": "; ".join(elim_pct),
        "eliminated_rank_judgessave": "; ".join(elim_rank_js),
        "eliminated_percent_judgessave": "; ".join(elim_pct_js),
        "delta_rank": week_delta(scored_rank, elim_rank),
        "delta_percent": week_delta(scored_pct, elim_pct),
        "hit_fan_worst_rank": int(k > 0 and fan_worst_rank in elim_rank),
        "hit_fan_worst_percent": int(k > 0 and fan_worst_pct in elim_pct),
    })

week_level = pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)

# ========== 5) 赛季汇总：差异率 + 谁更偏粉丝 ==========
elim_weeks = week_level[week_level["elim_k"] > 0].copy()
diff_mask = elim_weeks["eliminated_rank"] != elim_weeks["eliminated_percent"]

diff_by_season = elim_weeks[diff_mask].groupby("season").size().rename("diff_weeks")

season_summary = (
    elim_weeks.groupby("season")
    .agg(
        elim_weeks=("week", "count"),
        mean_delta_rank=("delta_rank", "mean"),
        mean_delta_percent=("delta_percent", "mean"),
        hit_fanworst_rank=("hit_fan_worst_rank", "mean"),
        hit_fanworst_percent=("hit_fan_worst_percent", "mean"),
    )
    .join(diff_by_season, how="left")
    .fillna({"diff_weeks": 0})
)
season_summary["diff_rate"] = season_summary["diff_weeks"] / season_summary["elim_weeks"]
season_summary["fan_bias_gap_delta"] = season_summary["mean_delta_percent"] - season_summary["mean_delta_rank"]
season_summary["fan_bias_gap_hit"] = season_summary["hit_fanworst_percent"] - season_summary["hit_fanworst_rank"]

diff_weeks = elim_weeks[diff_mask].copy()

# ========== 6) 争议选手专题表 ==========
def controversy_report(season: int, contestant: str) -> pd.DataFrame:
    df_s = base[base["season"] == season].copy()
    weeks = sorted(df_s["week"].unique())
    out_rows = []
    for w in weeks:
        dfw = df_s[df_s["week"] == w]
        k = int(wk_n.loc[(wk_n["season"] == season) & (wk_n["week"] == w), "elim_k"].iloc[0])

        for method in ["rank", "percent"]:
            scored = compute_week_scores(dfw, method)

            if contestant not in scored["contestant"].values:
                continue

            me = scored[scored["contestant"] == contestant].iloc[0]
            elim = decide_eliminated(scored, k)
            bottom2 = scored.head(2)["contestant"].tolist() if k > 0 else []
            judges_out = decide_eliminated_with_judges_save(scored, k)

            out_rows.append({
                "season": season,
                "week": w,
                "method": method,
                "elim_k": k,
                "judge_rank": float(me["judge_rank"]),
                "vote_rank": float(me["vote_rank"]),
                "combined": float(me["combined"]),
                "in_eliminated": int(contestant in elim),
                "in_bottom2": int(contestant in bottom2),
                "judges_save_eliminated": int(contestant in judges_out),
            })
    return pd.DataFrame(out_rows).sort_values(["week", "method"]).reset_index(drop=True)

controversy_tables = {
    "controversy_s2_jerry_rice": controversy_report(2,  "Jerry Rice"),
    "controversy_s4_billy_ray_cyrus": controversy_report(4, "Billy Ray Cyrus"),
    "controversy_s11_bristol_palin": controversy_report(11, "Bristol Palin"),
    "controversy_s27_bobby_bones": controversy_report(27, "Bobby Bones"),
}

# ========== 7) 决赛周（真实决赛选手集合）下的名次对比 ==========
def finale_ranking(season: int, method: str) -> pd.DataFrame:
    df_s = base[base["season"] == season].copy()
    last_week = df_s["week"].max()
    dfw = df_s[df_s["week"] == last_week]
    scored = compute_week_scores(dfw, method)

    if method == "rank":
        ranked = scored.sort_values(["rank_sum", "judge_total", "estimated_votes_total"],
                                    ascending=[True, False, False])
    else:
        ranked = scored.sort_values(["combined", "judge_total", "estimated_votes_total"],
                                    ascending=[False, False, False])

    ranked = ranked.reset_index(drop=True)
    ranked["final_place"] = np.arange(1, len(ranked) + 1)
    ranked.insert(0, "final_week", last_week)
    return ranked[["final_week", "contestant", "judge_total", "estimated_votes_total",
                   "judge_rank", "vote_rank", "combined", "final_place"]]

finale_tables = {}
for s in [2, 4, 11, 27]:
    finale_tables[f"finale_rankings_season{s}_rank"] = finale_ranking(s, "rank")
    finale_tables[f"finale_rankings_season{s}_percent"] = finale_ranking(s, "percent")

# ========== 8) 写出 Excel ==========
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    week_level.to_excel(writer, sheet_name="week_level", index=False)
    season_summary.reset_index().to_excel(writer, sheet_name="season_summary", index=False)
    diff_weeks.to_excel(writer, sheet_name="diff_weeks", index=False)

    for name, t in controversy_tables.items():
        t.to_excel(writer, sheet_name=name[:31], index=False)  # sheet 名最长 31

    for name, t in finale_tables.items():
        t.to_excel(writer, sheet_name=name[:31], index=False)

print("Done! Output saved to:", OUT_XLSX)
