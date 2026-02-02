import pandas as pd
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from scipy.stats import ttest_ind

def describe():
    # 从 Excel 文件中读取数据
    df = pd.read_excel('data.xlsx', header=None) # 有可能依赖openpyxl

    # 显示 DataFrame 的前几行数据
    print(df.head())

    # 将数据集转换为 DataFrame
    df1 = pd.DataFrame(df)
    # 取表格的4到20行的二,三列
    df1 = df1.iloc[4:19, 1:3]
    df1.columns = ['凝血酶浓度(单位/毫升)', '凝血时间(秒)']  # 重命名列

    # 转换数据类型为数值（非数值转为NaN）
    df1 = df1.apply(pd.to_numeric, errors='coerce')
    # 清理缺失值
    df1 = df1.dropna()

    # 使用 describe() 函数计算描述性统计信息
    statistics = df1.describe(include='all')

    # 打印结果
    print(statistics)

    a = df1.max()  # 最大值
    b = df1.min()  # 最小值
    d = df1.var()  # 方差
    e = df1.std()  # 标准差
    f = df1.mean() / df1.std()  # 变异系数
    print('最大值：', a, '\n', '最小值：', b, '\n', '方差：', round(d, 2), '标准差：', round(e, 2), '\n','变异系数：', round(f, 2))

    # df2 = df1.transpose() # 转置后才符合pairplot的格式要求
    # print(df2)
    plt.rc('font', family='SimHei')  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负
    # 使用seaborn的pairplot函数绘制散点图矩阵
    sns.pairplot(df1) # https://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn-pairplot
    plt.show()

def JB_QQ():
    """
    JB检验用于检验数据是否符合正态分布
    """
    df = pd.read_excel('data.xlsx',header=None)  # 有可能依赖openxyl
    df1 = pd.DataFrame(df)
    # 假设前两组数据有相关性
    data1 = df1.iloc[4:19, 1]  # 第二列
    data2 = df1.iloc[4:19, 2]  # 第三列
    data1 = pd.to_numeric(data1, errors='coerce').dropna()
    data2 = pd.to_numeric(data2, errors='coerce').dropna()
    # 进行JB检验
    jb_stat1, p_value1 = stats.jarque_bera(data1)
    jb_stat2, p_value2 = stats.jarque_bera(data2)

    # 打印JB统计量和P值
    print(f"JB statistic for 1: {jb_stat1}")
    print(f"P-value for 1: {p_value1}")
    print(f"JB statistic for 2: {jb_stat2}")
    print(f"P-value for 2: {p_value2}")

    # 务必看清楚你的原假设是什么

    if p_value1 < 0.05:
        print("在置信水平95%的情况下，拒绝原假设：数据不符合正态分布")
    else:
        print("在置信水平95%的情况下，不拒绝原假设：数据可能符合正态分布")

    if p_value2 < 0.05:
        print("在置信水平95%的情况下，拒绝原假设：数据不符合正态分布")
    else:
        print("在置信水平95%的情况下，不拒绝原假设：数据可能符合正态分布")

    # 创建 Q-Q 图
    fig1 = sm.qqplot(data1) # import statsmodels.api as sm
    plt.title("Q-Q Plot for Data 1")
    plt.show()

    fig2 = sm.qqplot(data2) # import statsmodels.api as sm
    plt.title("Q-Q Plot for Data 2")
    plt.show()

def relation():
    df1 = pd.read_excel('data.xlsx',header=None)  # 有可能依赖openxyl
    df = pd.DataFrame(df1)
    # 假设前两组数据有相关性
    df = df.iloc[4:19, 1:3]
    # 对每一列分别转换为数值型
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    # 进行显著性检验
    p_values = pd.DataFrame(index=df.columns, columns=df.columns)

    for i in df.columns:
        for j in df.columns:
            _, p_value = stats.pearsonr(df[i], df[j])
            p_values.loc[i, j] = p_value

    # 显示显著性P值矩阵
    print(p_values)

    def convert_pvalue_to_asterisks(pvalue):
        if pvalue <= 0.0001:
            return "****"
        elif pvalue <= 0.001:
            return "***"
        elif pvalue <= 0.01:
            return "**"
        elif pvalue <= 0.05:
            return "*"
        else:
            return "ns"  # 不显著


    # 将P值转换为显著性星号
    significance_asterisks = p_values.applymap(convert_pvalue_to_asterisks)

    # 打印带有显著性星号的相关系数矩阵
    print(significance_asterisks)

    # 相关系数和协方差
    # 计算数据集中各列之间的协方差
    covariance_matrix = df.cov(ddof=1) ## 参数ddof=1，表示结果除以N-1
    # 打印协方差矩阵
    print(covariance_matrix)

    # 计算数据集中各列之间的相关系数
    correlation_matrix = df.corr()
    # 打印相关系数矩阵
    print(correlation_matrix)
    # correlation_matrix.to_excel("res.xlsx")
    # 绘制热力图
    plt.figure(figsize=(9, 6), dpi=100)
    sns.set_style(rc= {'font.sans-serif':"Microsoft Yahei"})
    sns.heatmap(df.corr().round(2), annot=True, cmap='YlOrRd')
    plt.show()

    # 针对df的两列做t检验和显著性标记
    col1 = df.iloc[:, 0]
    col2 = df.iloc[:, 1]
    labels = [df.columns[0], df.columns[1]]
    stat, p_value = ttest_ind(col1, col2, equal_var=False)

    def convert_pvalue_to_asterisks(pvalue):
        if pvalue <= 0.0001:
            return "****"
        elif pvalue <= 0.001:
            return "***"
        elif pvalue <= 0.01:
            return "**"
        elif pvalue <= 0.05:
            return "*"
        return "ns"

    # 归一化处理（0-1标准化）
    col1_norm = (col1 - col1.min()) / (col1.max() - col1.min())
    col2_norm = (col2 - col2.min()) / (col2.max() - col2.min())

    # 绘制箱线图
    plt.figure(figsize=(8, 6))
    # 组装数据为长格式
    plot_df = pd.DataFrame({labels[0]: col1_norm, labels[1]: col2_norm})
    plot_df_melt = plot_df.melt(var_name='group', value_name='value')
    ax = sns.boxplot(x='group', y='value', data=plot_df_melt, palette=['skyblue', 'lightgreen'])

    # 添加显著性标记
    x1, x2 = 0, 1
    y_max = plot_df_melt['value'].max()
    h = (plot_df_melt['value'].max() - plot_df_melt['value'].min()) * 0.05
    y = y_max + h
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
    plt.text((x1 + x2) * .5, y + h, convert_pvalue_to_asterisks(p_value), ha='center', va='bottom', color="black")

    plt.ylabel('归一化值')
    plt.title('两列数据归一化箱线图及t检验显著性')
    plt.show()

if __name__ == "__main__":
    describe()
    JB_QQ()
    relation()