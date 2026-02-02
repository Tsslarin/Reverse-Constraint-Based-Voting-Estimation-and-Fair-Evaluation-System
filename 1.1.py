import re
import numpy as np
import pandas as pd

# =========================================================
# 1) 推断引擎（全局可复用版）：支持 rank / percent
#    - rank: 评委排名 + 粉丝排名 -> 最差淘汰
#    - percent: 评委占比 + 粉丝占比 -> 最差淘汰
# =========================================================
class DWTSInferenceEngine:
    def __init__(self, judges_scores, eliminated_name, contestant_names, method="rank", seed=0):
        self.judges_scores = np.array(judges_scores, dtype=float)
        self.eliminated_name = eliminated_name
        self.contestant_names = list(contestant_names)
        self.method = method
        self.rng = np.random.default_rng(seed)

        if eliminated_name not in self.contestant_names:
            raise ValueError(f"Eliminated contestant '{eliminated_name}' not found in contestant list.")
        self.elim_idx = self.contestant_names.index(eliminated_name)

        if len(self.judges_scores) != len(self.contestant_names):
            raise ValueError("judges_scores length must match contestant_names length.")
        if np.any(~np.isfinite(self.judges_scores)):
            raise ValueError("judges_scores contains NaN/inf; please filter week roster before inference.")

        self.n = len(self.contestant_names)

    @staticmethod
    def _rank_min_desc(values: np.ndarray) -> np.ndarray:
        """
        返回“分高排名小(1=best)”的 min-rank（并列取最小名次）。
        例如: [10, 10, 7] -> ranks [1, 1, 3]
        """
        v = np.asarray(values, dtype=float)
        order = np.argsort(-v, kind="mergesort")  # stable
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(v) + 1)

        # min-rank for ties:
        # 找到相同值的块，把块内 rank 统一成块起点 rank
        sorted_v = v[order]
        start = 0
        while start < len(v):
            end = start + 1
            while end < len(v) and sorted_v[end] == sorted_v[start]:
                end += 1
            min_rank = start + 1
            ranks[order[start:end]] = min_rank
            start = end
        return ranks

    def run_simulation(self, n_simulations=50000):
        """
        返回：
        - mean_votes: shape (n,)
        - std_votes: shape (n,)
        - feasible_rate: float
        - num_valid: int
        """
        # 评委指标
        if self.method == "rank":
            j_metric = self._rank_min_desc(self.judges_scores)  # 1=best
        elif self.method == "percent":
            total_j = float(np.sum(self.judges_scores))
            if total_j <= 0:
                raise ValueError("Total judges score must be positive for percent method.")
            j_metric = self.judges_scores / total_j
        else:
            raise ValueError("method must be 'rank' or 'percent'")

        # 采样：Dirichlet(1,...,1)
        samples = self.rng.dirichlet(np.ones(self.n), size=n_simulations)  # (S, n)

        # 批量计算是否兼容（向量化）
        if self.method == "rank":
            # 对每一行样本求 min-rank：这里为了简单与可读性，用循环；
            # 若你要更快可进一步向量化/numba。
            valid_mask = np.zeros(n_simulations, dtype=bool)
            for s in range(n_simulations):
                f_metric = self._rank_min_desc(samples[s])
                combined = j_metric + f_metric
                worst = np.max(combined)
                worst_indices = np.where(combined == worst)[0]
                valid_mask[s] = (self.elim_idx in worst_indices)
        else:
            combined = j_metric[None, :] + samples
            worst = np.min(combined, axis=1)  # 最小者淘汰
            # 若有并列最小：只要真实淘汰者属于最小集合就算兼容
            valid_mask = (combined[:, self.elim_idx] == worst)

        valid = samples[valid_mask]
        num_valid = int(valid.shape[0])
        feasible_rate = num_valid / float(n_simulations)

        if num_valid == 0:
            return None, None, feasible_rate, 0

        mean_votes = np.mean(valid, axis=0)
        std_votes = np.std(valid, axis=0)
        return mean_votes, std_votes, feasible_rate, num_valid


# =========================================================
# 2) 解析 results 字段：抽取淘汰周数
#    - "Eliminated Week 3" -> 3
#    - 决赛名次/冠军 -> NaN（不淘汰周）
# =========================================================
def parse_elim_week(results_str: str):
    if not isinstance(results_str, str):
        return np.nan
    m = re.search(r"Eliminated\s+Week\s+(\d+)", results_str, flags=re.IGNORECASE)
    if not m:
        return np.nan
    return int(m.group(1))


# =========================================================
# 3) 从宽表里取“某 season 某 week”的参赛名单 & 评委总分
#    规则：该 week 任意 judgeX_score 非空 -> 视为当周参赛
# =========================================================
def get_week_roster_and_scores(season_df: pd.DataFrame, week: int):
    judge_cols = [c for c in season_df.columns if c.startswith(f"week{week}_judge") and c.endswith("_score")]
    if not judge_cols:
        return [], [], judge_cols

    tmp = season_df[["celebrity_name"] + judge_cols].copy()

    # 当周参加：该周至少一个评委分不是 NaN
    participated_mask = tmp[judge_cols].notna().any(axis=1)
    tmp = tmp.loc[participated_mask].copy()

    if tmp.empty:
        return [], [], judge_cols

    # 评委总分 = 当周所有 judge 分数之和（跳过 NaN）
    tmp["judge_total"] = tmp[judge_cols].sum(axis=1, skipna=True)

    # 过滤 judge_total 为 0 或 NaN（通常不应发生，但保底）
    tmp = tmp[np.isfinite(tmp["judge_total"]) & (tmp["judge_total"] > 0)].copy()

    names = tmp["celebrity_name"].tolist()
    scores = tmp["judge_total"].astype(float).tolist()
    return names, scores, judge_cols


# =========================================================
# 4) 主流程：遍历 season-week，按淘汰者推断当周粉丝投票占比
# =========================================================
def run_global_inference(
    csv_path: str,
    default_method: str = "rank",
    n_simulations: int = 30000,
    seed: int = 0,
    out_csv_path: str = "dwts_inferred_fan_votes.csv",
):
    df = pd.read_csv(csv_path)

    # 预处理：淘汰周
    df["elim_week"] = df["results"].apply(parse_elim_week)

    seasons = sorted(df["season"].dropna().unique().astype(int).tolist())
    all_records = []

    for season in seasons:
        sdf = df[df["season"] == season].copy()

        # ========== 关键修改：动态判断赛季对应的方法 ==========
        # 1、2、28及以后赛季 → rank；中间赛季 → percent
        if season in {1, 2} or season >= 28:
            method = "rank"
        else:
            method = "percent"
        # =====================================================

        # 本季最大周：看 week*_judge*_score 的最大 week 序号
        week_nums = []
        for c in sdf.columns:
            m = re.match(r"week(\d+)_judge\d+_score", c)
            if m:
                week_nums.append(int(m.group(1)))
        if not week_nums:
            continue
        max_week = max(week_nums)

        # 当周淘汰者：elim_week == week 的所有人（可能存在双淘汰）
        for week in range(1, max_week + 1):
            elim_names = sdf.loc[sdf["elim_week"] == week, "celebrity_name"].dropna().tolist()
            if len(elim_names) == 0:
                continue  # 该周无淘汰（可能是决赛周/停赛周等）

            roster_names, roster_scores, used_cols = get_week_roster_and_scores(sdf, week)
            if len(roster_names) < 2:
                continue

            # 对每个淘汰者分别推断（双淘汰周：会生成两条记录）
            for eliminated_name in elim_names:
                if eliminated_name not in roster_names:
                    # 有时 results 记录的淘汰周与宽表缺分会不一致，直接跳过并记录原因
                    all_records.append({
                        "season": season,
                        "week": week,
                        "method": method,
                        "eliminated": eliminated_name,
                        "status": "skip_elim_not_in_roster",
                        "feasible_rate": np.nan,
                        "num_valid": 0,
                        "n_simulations": n_simulations,
                    })
                    continue

                engine = DWTSInferenceEngine(
                    judges_scores=roster_scores,
                    eliminated_name=eliminated_name,
                    contestant_names=roster_names,
                    method=method,
                    seed=seed + 100000 * season + 1000 * week,  # 稳定可复现
                )
                mean_votes, std_votes, feasible_rate, num_valid = engine.run_simulation(n_simulations=n_simulations)

                if mean_votes is None:
                    all_records.append({
                        "season": season,
                        "week": week,
                        "method": method,
                        "eliminated": eliminated_name,
                        "status": "no_feasible_solution",
                        "feasible_rate": feasible_rate,
                        "num_valid": 0,
                        "n_simulations": n_simulations,
                    })
                    continue

                # 展开到“每周-每人一行”，便于后续做回归/可视化
                for name, jscore, mv, sv in zip(roster_names, roster_scores, mean_votes, std_votes):
                    all_records.append({
                        "season": season,
                        "week": week,
                        "method": method,
                        "eliminated": eliminated_name,
                        "contestant": name,
                        "judge_total": float(jscore),
                        "fan_vote_mean": float(mv),
                        "fan_vote_std": float(sv),
                        "feasible_rate": float(feasible_rate),
                        "num_valid": int(num_valid),
                        "n_simulations": int(n_simulations),
                        "status": "ok",
                    })

    out = pd.DataFrame(all_records)
    out.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    return out


# =========================================================
# 5) 直接运行（按你的文件路径）
# =========================================================
if __name__ == "__main__":
    # 不再需要手动指定 method_by_season，逻辑已内置到主流程
    result_df = run_global_inference(
        csv_path="2026_MCM_Problem_C_Data.csv",
        default_method="rank",  # 仅作为兜底（实际已被动态逻辑覆盖）
        n_simulations=10000,    # 全局跑建议先 1~3 万；验证无误再加到 10 万
        seed=0,
        out_csv_path="dwts_inferred_fan_votes.csv",
    )

    print("Done. Output saved to dwts_inferred_fan_votes.csv")
    print(result_df.head(20))