import pandas as pd
import re

def process_dwts_data(input_file):
    # 读取原始数据
    df = pd.read_csv(input_file)

    # ==========================================
    # 任务 1: 生成淘汰名单表 (eliminated_players_by_week.csv)
    # ==========================================
    
    # 定义函数从 'results' 列提取周数
    def get_eliminated_week(result_str):
        if pd.isna(result_str):
            return None
        # 匹配 "Eliminated Week X" 格式
        match = re.search(r'Eliminated Week (\d+)', str(result_str))
        if match:
            return int(match.group(1))
        return None

    # 应用函数创建新列
    df['eliminated_week'] = df['results'].apply(get_eliminated_week)

    # 提取被淘汰的选手信息
    # 过滤掉没有被淘汰周数的记录，并按赛季和周数排序
    elim_df = df.dropna(subset=['eliminated_week'])[['season', 'eliminated_week', 'celebrity_name']]
    
    # 重命名列以符合输出要求
    elim_df.rename(columns={'eliminated_week': 'week', 'celebrity_name': 'eliminated_celebrity'}, inplace=True)
    
    # 转换为整数类型并排序
    elim_df['week'] = elim_df['week'].astype(int)
    elim_df = elim_df.sort_values(by=['season', 'week'])

    # 保存第一个文件
    elim_df.to_csv('eliminated_players_by_week.csv', index=False)
    print("生成完成: eliminated_players_by_week.csv")

    # ==========================================
    # 任务 2: 生成详细评委分表 (detailed_judge_scores.csv)
    # ==========================================
    
    all_weeks_data = []

    # 遍历第 1 周到第 11 周
    for w in range(1, 12):
        # 构造当前周的评委列名
        judge_cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
        
        # 检查这些列是否存在于数据中
        current_cols = [c for c in judge_cols if c in df.columns]
        
        if not current_cols:
            continue

        # 提取当前周的数据子集
        # 包括赛季、选手名和当前周的评委分
        temp_df = df[['season', 'celebrity_name'] + current_cols].copy()
        temp_df['week'] = w  # 添加周数列

        # 重命名评委列为统一格式 (judge1, judge2...)
        rename_map = {f'week{w}_judge{j}_score': f'judge{j}' for j in range(1, 5)}
        temp_df.rename(columns=rename_map, inplace=True)

        # 数据清洗：确保分数为数值型
        judge_renamed_cols = [rename_map[c] for c in current_cols]
        for col in judge_renamed_cols:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

        # 计算该周总分，用于过滤未参赛选手
        temp_df['total_score'] = temp_df[judge_renamed_cols].sum(axis=1)

        # 过滤掉总分为 0 的行（代表该选手该周未出场或数据缺失）
        active_df = temp_df[temp_df['total_score'] > 0].copy()

        # 确保 judge4 列存在（如果某季只有3个评委，补空值），保持格式一致
        for i in range(1, 5):
            if f'judge{i}' not in active_df.columns:
                active_df[f'judge{i}'] = None

        # 选择需要的列并添加到总列表
        cols_ordered = ['season', 'week', 'celebrity_name', 'judge1', 'judge2', 'judge3', 'judge4', 'total_score']
        all_weeks_data.append(active_df[cols_ordered])

    # 合并所有周的数据
    final_scores_df = pd.concat(all_weeks_data, ignore_index=True)

    # 排序：按赛季、周数升序，总分降序
    final_scores_df = final_scores_df.sort_values(by=['season', 'week', 'total_score'], ascending=[True, True, False])

    # 保存第二个文件
    final_scores_df.to_csv('detailed_judge_scores.csv', index=False)
    print("生成完成: detailed_judge_scores.csv")

# 运行处理函数
if __name__ == "__main__":
    # 确保文件名与你上传的文件名一致
    process_dwts_data('2026_MCM_Problem_C_Data.csv')