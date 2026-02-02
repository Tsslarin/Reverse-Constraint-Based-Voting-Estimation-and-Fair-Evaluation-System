import numpy as np
import scipy.stats as stats

class DWTS_Inference_Engine:
    def __init__(self, judges_scores, eliminated_name, contestant_names, method='rank'):
        """
        :param judges_scores: list of floats, 评委分数
        :param eliminated_name: str, 实际被淘汰者的名字
        :param contestant_names: list of str, 选手名字列表
        :param method: 'rank' (排名制) 或 'percent' (百分比制)
        """
        self.judges_scores = np.array(judges_scores)
        self.eliminated_name = eliminated_name
        self.contestant_names = contestant_names
        self.method = method
        self.n = len(judges_scores)
        
        # 找到淘汰者的索引
        try:
            self.elim_idx = contestant_names.index(eliminated_name)
        except ValueError:
            raise ValueError(f"Eliminated contestant '{eliminated_name}' not found in names list.")

    def _get_ranks(self, scores, descending=True):
        """
        计算排名。
        descending=True: 分数高者排名靠前（数值小，如1）。
        使用 'min' 方法处理并列：并列第一则都为1，下一名是3。
        注意：DWTS规则中，排名和计算时，通常分数越高排名数值越小(1st)。
        """
        # argsort两次得到排名 (0-based)
        # 负号用于实现降序排序
        if descending:
            return stats.rankdata(-np.array(scores), method='min')
        else:
            return stats.rankdata(np.array(scores), method='min')

    def run_simulation(self, n_simulations=100000):
        print(f"--- 开始模拟 (模式: {self.method}, 样本数: {n_simulations}) ---")
        
        # 1. 预计算评委部分的指标
        if self.method == 'rank':
            # 评委分越高，排名数字越小 (1=Best)
            j_metric = self._get_ranks(self.judges_scores, descending=True)
        else:
            # 百分比制：分数占比
            total_j = np.sum(self.judges_scores)
            j_metric = self.judges_scores / total_j

        valid_samples = []

        # 2. 批量生成样本以提高速度 (Vectorized generation)
        # 生成形状为 (n_simulations, n) 的随机矩阵，每行和为1
        samples = np.random.dirichlet(np.ones(self.n), n_simulations)

        for f_votes in samples:
            # 3. 计算粉丝指标
            if self.method == 'rank':
                # 票数越多，排名越小 (1=Best)
                f_metric = self._get_ranks(f_votes, descending=True)
                
                # 排名制逻辑：通常是 Rank_J + Rank_F
                # 总和最大者（表现最差）被淘汰
                # 注意：这是DWTS早期逻辑，但也可能存在 Bottom 2 规则
                # 这里我们模拟最纯粹的 "Lowest Combined Score leaves" 
                # 但在排名制里，"Score"其实是Rank Sum，越大越差。
                combined = j_metric + f_metric
                
                # 找到最大Rank Sum的索引（们）
                # 可能有平局，这里简单处理：如果有多个最大，随机选一个淘汰
                # 或者更严格：只要目标在最大值集合中就算符合
                max_score = np.max(combined)
                sim_elim_indices = np.where(combined == max_score)[0]
                
                is_compatible = (self.elim_idx in sim_elim_indices)
                
            else: # percent
                # 百分比制逻辑：(J_score / Total_J) + (F_votes / Total_F)
                # 总和最小者（表现最差）被淘汰
                combined = j_metric + f_votes
                min_score = np.min(combined)
                sim_elim_indices = np.where(combined == min_score)[0]
                
                is_compatible = (self.elim_idx in sim_elim_indices)

            # 4. 筛选
            if is_compatible:
                valid_samples.append(f_votes)

        # 5. 结果分析
        valid_samples = np.array(valid_samples)
        num_valid = len(valid_samples)
        
        if num_valid == 0:
            print("警告：未找到可行解。可能约束条件过于严格或模拟次数不足。")
            return None

        print(f"找到可行解数量: {num_valid} (可行率: {num_valid/n_simulations:.2%})")
        
        # 计算统计量
        mean_votes = np.mean(valid_samples, axis=0)
        std_votes = np.std(valid_samples, axis=0)
        
        print("\n--- 推断结果：潜在粉丝得票率 ---")
        print(f"{'选手':<15} {'评委分':<10} {'推断得票率(均值)':<15} {'不确定性(Std)':<10}")
        print("-" * 55)
        for i in range(self.n):
            print(f"{self.contestant_names[i]:<15} {self.judges_scores[i]:<10.1f} {mean_votes[i]:<15.2%} +/- {std_votes[i]:.2%}")
            
        return mean_votes

# ==========================================
# 实际案例测试：第27季 Bobby Bones 夺冠之谜
# ==========================================
# 假设决赛四强数据
# 评委分数 (Total out of 60 for 2 dances)
# Milo: 60, Evanna: 60, Alexis: 57, Bobby: 54
names = ["Milo", "Evanna", "Alexis", "Bobby Bones"]
scores = [60, 60, 57, 54]

# 在百分比制下，Bobby Bones 赢了 (Winner)
# 注意：上面的类是用来推断“淘汰”的。
# 为了推断“冠军”，我们需要稍微修改逻辑：combined score 最高者 = Bobby
# 我们直接在这里用脚本修改一下逻辑来演示

print("正在模拟第27季决赛（百分比制），假设 Bobby Bones 获胜...")

engine = DWTS_Inference_Engine(scores, "Bobby Bones", names, method='percent')

# 手动重写一下核心循环逻辑用于“获胜”推断
n_sim = 100000
total_j = np.sum(scores)
j_percent = np.array(scores) / total_j
valid_winner_samples = []

raw_samples = np.random.dirichlet(np.ones(4), n_sim)

for f_votes in raw_samples:
    combined = j_percent + f_votes
    # 获胜者是总分最高的
    winner_idx = np.argmax(combined)
    if names[winner_idx] == "Bobby Bones":
        valid_winner_samples.append(f_votes)

valid_winner_samples = np.array(valid_winner_samples)
mean_winner_votes = np.mean(valid_winner_samples, axis=0)

print(f"成功模拟次数: {len(valid_winner_samples)}")
print("\n--- Bobby Bones 夺冠所需的潜在粉丝投票分布 ---")
for i in range(4):
    print(f"{names[i]}: {mean_winner_votes[i]:.2%} (评委分占比: {j_percent[i]:.2%})")

print("\n分析：")
print("可以看到，尽管Bobby的评委分最低，但如果他在粉丝投票中占据压倒性优势")
print("（模型通常显示需要 40%-50% 甚至更高的独占票仓），他就能在百分比制下通过数学机制获胜。")