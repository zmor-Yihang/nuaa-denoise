import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vmdpy import VMD
from mealpy import FloatVar, IntegerVar, SMA, Problem
import math
from scipy.stats import entropy

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 样本熵实现
def sample_entropy(time_series, m=2, r=0.2):
    """
    计算时间序列的样本熵
    
    参数:
    time_series: 输入的时间序列
    m: 嵌入维度
    r: 相似性阈值，通常为时间序列标准差的0.2倍
    
    返回:
    样本熵值
    """
    # 确保输入是numpy数组
    time_series = np.array(time_series)
    # 如果r是相对值，则转换为绝对值
    if r < 1:
        r = r * np.std(time_series)
    
    N = len(time_series)
    # 创建模板向量
    templates = np.zeros((N-m+1, m))
    for i in range(N-m+1):
        templates[i] = time_series[i:i+m]
    
    # 计算m维度下的匹配数
    A = np.zeros(N-m+1)
    B = np.zeros(N-m+1)
    
    for i in range(N-m+1):
        # 计算所有模板与当前模板的距离
        distances = np.max(np.abs(templates - templates[i]), axis=1)
        # 不计算自匹配
        distances[i] = float('inf')
        # 计算m维度下的匹配数
        B[i] = np.sum(distances <= r)
        
        # 如果可以计算m+1维度
        if i < N-m:
            # 计算m+1维度下的匹配数
            m_plus_1_distances = np.max(np.abs(templates[:-1] - templates[i]), axis=1)
            if i < N-m-1:
                m_plus_1_distances[i] = float('inf')
            A[i] = np.sum(m_plus_1_distances <= r)
    
    # 计算样本熵
    return -np.log(np.sum(A) / np.sum(B))

# 排列熵实现
def permutation_entropy(time_series, order=3, delay=1, normalize=False):
    """
    计算时间序列的排列熵
    
    参数:
    time_series: 输入的时间序列
    order: 嵌入维度
    delay: 时间延迟
    normalize: 是否归一化
    
    返回:
    排列熵值
    """
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    
    # 嵌入时间序列
    embedded = np.zeros((len(x)-(order-1)*delay, order))
    for i in range(order):
        embedded[:, i] = x[i*delay:i*delay + len(embedded)]
    
    # 对每个嵌入向量进行排序并获取索引
    sorted_idx = embedded.argsort(axis=1)
    
    # 计算每个排列模式的哈希值
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    
    # 计算每个排列模式的频率
    _, c = np.unique(hashval, return_counts=True)
    p = c / c.sum()
    
    # 计算排列熵
    pe = -np.sum(p * np.log2(p))
    
    # 归一化
    if normalize:
        pe /= np.log2(math.factorial(order))
    
    return pe

# VMD参数优化问题定义
class VMDOptimizationProblem(Problem):
    def __init__(self, bounds=None, minmax="max", data=None, entropy_type="sample", **kwargs):
        self.data = data
        self.entropy_type = entropy_type  # 'sample' 或 'permutation'
        super().__init__(bounds, minmax, **kwargs)
    
    def obj_func(self, x):
        """
        目标函数：计算给定VMD参数下的熵值
        对于样本熵，值越小表示信号越规则（去噪效果越好）
        对于排列熵，值越小表示信号越规则（去噪效果越好）
        因此我们使用负熵值作为优化目标（最大化问题）
        """
        x_decoded = self.decode_solution(x)
        K = int(x_decoded["K"])  # 模态数量
        alpha = x_decoded["alpha"]  # 惩罚因子
        
        # 固定的VMD参数
        tau = 0
        DC = 1
        init = 1
        tol = 1e-7
        
        # 执行VMD分解
        u, _, _ = VMD(self.data, alpha, tau, K, DC, init, tol)
        
        # 重构信号（所有模态的和）
        reconstructed = np.sum(u, axis=0)
        
        # 计算熵值
        if self.entropy_type == "sample":
            entropy_value = sample_entropy(reconstructed)
        else:  # permutation
            entropy_value = permutation_entropy(reconstructed)
        
        # 返回负熵值（因为我们要最大化目标函数）
        return -entropy_value

def optimize_vmd_parameters(entropy_type="sample"):
    """
    使用SMA算法优化VMD参数
    
    参数:
    entropy_type: 使用的熵类型，'sample'表示样本熵，'permutation'表示排列熵
    
    返回:
    最优参数和优化结果
    """
    # 读取数据
    data = pd.read_excel('../data/Thz_data.xlsx')
    amplitude = data.iloc[:, 1].values
    
    # 定义参数边界
    # K: 模态数量，整数，范围[2, 12]
    # alpha: 惩罚因子，浮点数，范围[100, 5000]
    my_bounds = [
        IntegerVar(lb=2, ub=12, name="K"),
        FloatVar(lb=100, ub=5000, name="alpha")
    ]
    
    # 创建优化问题
    problem = VMDOptimizationProblem(bounds=my_bounds, minmax="max", data=amplitude, entropy_type=entropy_type)
    
    # 创建SMA优化器
    model = SMA.OriginalSMA(epoch=2, pop_size=20)
    
    # 创建列表用于记录每次迭代的最佳适应度值
    fitness_history = []
    
    # 求解优化问题
    model.solve(problem)
    
    # 从模型的历史记录中获取每次迭代的最佳适应度值
    # 由于我们在优化过程中使用的是负熵值（最大化问题），需要取负数得到实际熵值
    fitness_history = [-fit for fit in model.history.list_current_best_fit]
    
    # 获取最优解
    best_position = model.g_best
    best_fitness = best_position.target.fitness
    
    # 解码最优解
    best_parameters = problem.decode_solution(best_position.solution)
    
    # 打印结果
    print(f"最优参数: {best_parameters}")
    print(f"最优适应度值: {-best_fitness}")
    
    # 使用最优参数进行VMD分解
    K_optimal = int(best_parameters["K"])
    alpha_optimal = best_parameters["alpha"]
    
    # 固定的VMD参数
    tau = 0
    DC = 1
    init = 1
    tol = 1e-7
    
    # 执行VMD分解
    u, _, _ = VMD(amplitude, alpha_optimal, tau, K_optimal, DC, init, tol)
    
    # 保存最优参数
    result_df = pd.DataFrame({
        '参数': ['K', 'alpha'],
        '最优值': [K_optimal, alpha_optimal]
    })
    result_df.to_excel('../save/optimal_vmd_parameters.xlsx', index=False)
    
    # 可视化优化结果
    time = data.iloc[:, 0].values
    time_aligned = time[:u.shape[1]]
    
    plt.figure(figsize=(15, 10))
    
    # 原始信号
    plt.subplot(K_optimal+1, 1, 1)
    plt.plot(time, amplitude, 'b')
    plt.title('原始信号')
    plt.xlabel('时间')
    plt.ylabel('幅值')
    
    # 各模态分量
    for i in range(K_optimal):
        plt.subplot(K_optimal+1, 1, i+2)
        plt.plot(time_aligned, u[i], 'g')
        plt.title(f'模态 {i+1}')
        plt.xlabel('时间')
        plt.ylabel('幅值')
    
    plt.tight_layout()
    plt.show()

    
    # 绘制适应度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history, 'b-', marker='o')
    plt.title(f'{entropy_type}熵优化过程适应度曲线')
    plt.xlabel('迭代次数')
    plt.ylabel(f'{entropy_type}熵值')
    plt.grid(True)
    plt.show()
    
    # 保存适应度历史数据
    fitness_df = pd.DataFrame({
        '迭代次数': range(1, len(fitness_history) + 1),
        f'{entropy_type}熵值': fitness_history
    })
    fitness_df.to_excel('../save/fitness_history.xlsx', index=False)
    
    return best_parameters, -best_fitness

if __name__ == "__main__":
    # 用户选择使用的熵类型
    print("请选择使用的熵类型:")
    print("1. 样本熵 (Sample Entropy)")
    print("2. 排列熵 (Permutation Entropy)")
    choice = input("请输入选择 (1 或 2): ")
    
    entropy_type = "sample" if choice == "1" else "permutation"
    print(f"使用{entropy_type}熵进行VMD参数优化...")
    
    # 执行优化
    best_params, best_fitness = optimize_vmd_parameters(entropy_type)
    
    print("\n优化完成!")
    print(f"最优参数: K = {int(best_params['K'])}, alpha = {best_params['alpha']:.2f}")
    print(f"最优{entropy_type}熵值: {best_fitness:.6f}")
    print("\n最优分解结果已保存至 '../save/optimal_IMFs.xlsx'")
    print("最优参数已保存至 '../save/optimal_vmd_parameters.xlsx'")
    print("适应度历史数据已保存至 '../save/fitness_history.xlsx'")