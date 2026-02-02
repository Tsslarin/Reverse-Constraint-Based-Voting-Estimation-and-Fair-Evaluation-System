import pandas as pd
import numpy as np

# === 1. 投票池映射逻辑 (与之前一致) ===
def get_season_vote_volume(season, week):
    try:
        s = int(season)
        w = int(week)
    except ValueError:
        return 10_000_000 
    
    is_early = (w <= 4)
    
    if s == 1:
        if is_early: return 5_000_000
        if w == 5: return 6_500_000
        return 9_000_000 
    elif 2 <= s <= 4:
        is_late = (w >= 8 if s==2 else w >= 10)
        if is_early: return 6_000_000
        if is_late: return 12_000_000
        return 8_000_000
    elif 5 <= s <= 10:
        is_late = (w >= 10)
        if is_early: return 10_000_000
        if is_late: return 15_000_000
        return 11_000_000
    elif 11 <= s <= 15:
        is_late = (w >= 10)
        if is_early: return 11_000_000
        if is_late: return 20_000_000
        return 13_000_000
    elif 16 <= s <= 20:
        is_late = (w >= 10)
        if is_early: return 9_000_000
        if is_late: return 16_000_000
        return 10_000_000
    elif 21 <= s <= 27:
        is_late = (w >= 10)
        if is_early: return 7_000_000
        if is_late: return 12_000_000
        return 8_000_000
    elif 28 <= s <= 32:
        is_late = (w >= 10)
        if is_early: return 8_000_000
        if is_late: return 14_000_000
        return 9_000_000
    elif s == 33:
        if is_early: return 14_000_000
        if w >= 10: return 32_000_000
        return 18_000_000
    elif s == 34:
        if is_early: return 21_000_000
        if w >= 10: return 72_000_000
        return 35_000_000
        
    return 10_000_000

# === 主程序 ===
if __name__ == "__main__":
    file_path = 'dwts_inferred_fan_votes (1).csv' # 请确保文件名正确
    print(f"--- 正在读取文件: {file_path} ---")
    df = pd.read_csv(file_path)

    # 1. 【诊断】检查 fan_vote_mean 是否真的全是数字
    print(f"数据总行数: {len(df)}")
    
    # 强制将 fan_vote_mean 转为数字，无法转换的变为 NaN
    df['fan_vote_mean'] = pd.to_numeric(df['fan_vote_mean'], errors='coerce')
    df['fan_vote_std'] = pd.to_numeric(df['fan_vote_std'], errors='coerce')

    # 检查是否有空值 (NaN)
    nan_rows = df[df['fan_vote_mean'].isna()]
    if not nan_rows.empty:
        print("\n⚠️ 发现异常数据！以下行的 fan_vote_mean 是空的或无法读取：")
        print(nan_rows[['season', 'week', 'contestant', 'fan_vote_mean']])
        print(">> 原因分析：这些行可能在之前的模拟中未找到可行解 (Consistency=0)。")
        
        # 【修复 A】将空值填充为 0
        print(">> 自动修复：将这些异常行的得票率设为 0。")
        df['fan_vote_mean'] = df['fan_vote_mean'].fillna(0.0)
        df['fan_vote_std'] = df['fan_vote_std'].fillna(0.0)
    else:
        print(">> fan_vote_mean 列检查通过，没有发现空值。")

    # 2. 【计算】得票总数 (Pre-calculation)
    # 先算出浮点数结果，暂不转 int
    df['estimated_votes_total'] = df.apply(
        lambda row: row['fan_vote_mean'] * get_season_vote_volume(row['season'], row['week']), 
        axis=1
    )

    # 3. 【修复 B】检查计算结果是否有无穷大或空值
    # 有时候 fan_vote_mean 是数字，但计算过程可能产生 Inf
    df.replace([np.inf, -np.inf], 0, inplace=True) # 将无穷大替换为0
    df['estimated_votes_total'] = df['estimated_votes_total'].fillna(0) # 再次确保无空值

    # 4. 【转换】现在可以安全转为整数了
    try:
        df['estimated_votes_total'] = df['estimated_votes_total'].astype(int)
        print("\n>> 成功计算并转换 estimated_votes_total。")
    except Exception as e:
        print(f"\n❌ 转换整数时依然报错: {e}")
        print("尝试保留为浮点数继续运行...")

    # 5. 【计算】确定性指标 CV
    # CV = sigma / mu (加微小值 1e-9 避免除以0)
    df['certainty_cv'] = df['fan_vote_std'] / (df['fan_vote_mean'] + 1e-9)
    df['certainty_cv'] = df['certainty_cv'].fillna(0).round(4) # 再次填充空值

    # 6. 预览与保存
    print("\n--- 结果预览 (前5行) ---")
    print(df[['season', 'week', 'contestant', 'fan_vote_mean', 'estimated_votes_total', 'certainty_cv']].head())

    output_path = 'dwts_final_with_votes_and_cv.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ 文件已成功保存至: {output_path}")