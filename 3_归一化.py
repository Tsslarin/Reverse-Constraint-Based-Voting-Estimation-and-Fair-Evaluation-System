import pandas as pd
import numpy as np

df = pd.read_csv("dwts_inferred_fan_votes2_with_info.csv")

group_keys = ['season', 'week', 'method']
g = df.groupby(group_keys, dropna=False)

# ----- vote share -----
vote_sum = g['estimated_votes_total'].transform('sum')
df['vote_share'] = np.where(vote_sum > 0, df['estimated_votes_total'] / vote_sum, np.nan)

# ----- judge share -----
judge_sum = g['judge_total'].transform('sum')
df['judge_share'] = np.where(judge_sum > 0, df['judge_total'] / judge_sum, np.nan)

# （可选）检查每个组内 share 之和是否为 1
check = df.groupby(group_keys)[['vote_share', 'judge_share']].sum()
print(check.describe())

df.to_csv("dwts_with_week_normalized_shares.csv", index=False)
