import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vmdpy import VMD
from mealpy import FloatVar, IntegerVar, SMA, SSA, Problem
import math
from scipy.stats import entropy, kurtosis, skew
from scipy.fft import fft
from scipy import signal
from scipy.linalg import svd

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

# 功率谱熵计算函数
def spectral_entropy(signal):
    """
    计算信号的功率谱熵
    
    参数:
    signal (array_like): 输入的一维信号数据
    
    返回值:
    float: 功率谱熵值，反映信号能量分布的复杂度
    """
    # FFT变换及功率谱计算
    fft_vals = np.abs(fft(signal))  # 计算信号的傅里叶变换幅值
    psd = fft_vals ** 2  # 计算功率谱密度
    psd_norm = psd / psd.sum()  # 归一化处理
    
    # 避免log(0)的情况
    psd_norm = psd_norm[psd_norm > 0]
    
    return -np.sum(psd_norm * np.log2(psd_norm))  # 香农熵公式计算

# 奇异谱熵计算函数
def singular_spectrum_entropy(signal, embedding_dimension=20):
    """
    计算信号的奇异谱熵
    
    参数:
    signal (array_like): 输入的一维信号数据
    embedding_dimension: 嵌入维度，用于构建轨迹矩阵
    
    返回值:
    float: 奇异谱熵值，反映信号的复杂度和非线性特性
    """
    # 确保输入是numpy数组
    signal = np.array(signal)
    N = len(signal)
    
    # 构建轨迹矩阵（Hankel矩阵）
    K = N - embedding_dimension + 1
    trajectory_matrix = np.zeros((embedding_dimension, K))
    for i in range(K):
        trajectory_matrix[:, i] = signal[i:i+embedding_dimension]
    
    # 奇异值分解
    U, sigma, Vt = svd(trajectory_matrix, full_matrices=False)
    
    # 计算归一化的奇异值
    sigma_norm = sigma / np.sum(sigma)
    
    # 计算奇异谱熵
    sse = -np.sum(sigma_norm * np.log2(sigma_norm + np.finfo(float).eps))
    
    return sse

# 能量熵计算函数
def energy_entropy(signal, num_bands=8):
    """
    计算信号的能量熵
    
    参数:
    signal (array_like): 输入的一维信号数据
    num_bands: 频带数量
    
    返回值:
    float: 能量熵值，反映信号能量在不同频带上分布的均匀程度
    """
    # 确保输入是numpy数组
    signal = np.array(signal)
    
    # 计算信号的功率谱
    fft_vals = np.abs(fft(signal))
    power_spectrum = fft_vals ** 2
    
    # 将功率谱分成num_bands个频带
    bands = np.array_split(power_spectrum[:len(power_spectrum)//2], num_bands)
    
    # 计算每个频带的能量
    band_energies = np.array([np.sum(band) for band in bands])
    
    # 计算总能量
    total_energy = np.sum(band_energies)
    
    # 避免除零错误
    if total_energy == 0:
        return 0
    
    # 计算每个频带的能量比例
    energy_ratios = band_energies / total_energy
    
    # 避免log(0)的情况
    energy_ratios = energy_ratios[energy_ratios > 0]
    
    # 计算能量熵
    return -np.sum(energy_ratios * np.log2(energy_ratios))

# 近似熵计算函数
def approximate_entropy(signal, m=2, r=0.2):
    """
    计算信号的近似熵
    
    参数:
    signal (array_like): 输入的一维信号数据
    m: 嵌入维度
    r: 相似性阈值，通常为信号标准差的0.2倍
    
    返回值:
    float: 近似熵值，反映信号的复杂度和不规则性
    """
    # 确保输入是numpy数组
    signal = np.array(signal)
    N = len(signal)
    
    # 如果r是相对值，则转换为绝对值
    if r < 1:
        r = r * np.std(signal)
    
    # 定义内部函数计算Phi
    def _phi(m_value):
        # 创建m_value维度的模板向量
        templates = np.zeros((N - m_value + 1, m_value))
        for i in range(N - m_value + 1):
            templates[i] = signal[i:i + m_value]
        
        # 计算每个模板与其他所有模板的距离
        count = np.zeros(N - m_value + 1)
        for i in range(N - m_value + 1):
            # 计算所有模板与当前模板的距离
            distances = np.max(np.abs(templates - templates[i]), axis=1)
            # 计算小于阈值r的距离数量
            count[i] = np.sum(distances <= r) / (N - m_value + 1)
        
        # 计算对数平均值
        return np.sum(np.log(count)) / (N - m_value + 1)
    
    # 计算m和m+1维度下的Phi值
    phi_m = _phi(m)
    phi_m_plus_1 = _phi(m + 1)
    
    # 计算近似熵
    return phi_m - phi_m_plus_1

# 模糊熵计算函数
def fuzzy_entropy(signal, m=2, r=0.2, n=2):
    """
    计算信号的模糊熵
    
    参数:
    signal (array_like): 输入的一维信号数据
    m: 嵌入维度
    r: 相似性阈值，通常为信号标准差的0.2倍
    n: 模糊函数的指数，控制模糊隶属度函数的陡峭程度
    
    返回值:
    float: 模糊熵值，反映信号的复杂度和不规则性
    """
    # 确保输入是numpy数组
    signal = np.array(signal)
    N = len(signal)
    
    # 如果r是相对值，则转换为绝对值
    if r < 1:
        r = r * np.std(signal)
    
    # 定义模糊隶属度函数
    def _fuzzy_membership(d, r_value):
        return np.exp(-np.power(d, n) / r_value)
    
    # 定义内部函数计算Phi
    def _phi(m_value):
        # 创建m_value维度的模板向量
        templates = np.zeros((N - m_value + 1, m_value))
        for i in range(N - m_value + 1):
            templates[i] = signal[i:i + m_value]
        
        # 去除每个模板的均值（基线）
        templates = templates - np.mean(templates, axis=1, keepdims=True)
        
        # 计算每个模板与其他所有模板的模糊相似度
        similarity_sum = 0
        for i in range(N - m_value + 1):
            # 计算所有模板与当前模板的最大距离
            distances = np.max(np.abs(templates - templates[i]), axis=1)
            # 计算模糊隶属度并求和（排除自身）
            similarity_sum += np.sum(_fuzzy_membership(distances, r)) - 1  # 减1排除自身
        
        # 计算平均模糊相似度
        return similarity_sum / ((N - m_value) * (N - m_value + 1))
    
    # 计算m和m+1维度下的Phi值
    phi_m = _phi(m)
    phi_m_plus_1 = _phi(m + 1)
    
    # 避免除零错误
    if phi_m == 0:
        return 0
    
    # 计算模糊熵
    return np.log(phi_m / phi_m_plus_1)

# 计算信号稀疏性指标
def sparsity_measure(signal):
    """
    计算信号的稀疏性指标（基于L1/L2范数比）
    
    参数:
    signal (array_like): 输入的一维信号数据
    
    返回值:
    float: 稀疏性指标，值越大表示信号越稀疏
    """
    # 避免除零错误
    if np.linalg.norm(signal, 2) == 0:
        return 0
    
    # 计算L1范数与L2范数的比值
    l1_norm = np.linalg.norm(signal, 1)
    l2_norm = np.linalg.norm(signal, 2)
    
    # 归一化稀疏性指标
    n = len(signal)
    sparsity = (np.sqrt(n) - l1_norm / l2_norm) / (np.sqrt(n) - 1)
    
    return sparsity

# 计算信号能量比
def energy_ratio(signal, original_signal):
    """
    计算重构信号与原始信号的能量比
    
    参数:
    signal (array_like): 重构的信号
    original_signal (array_like): 原始信号
    
    返回值:
    float: 能量比值，接近1表示重构信号保留了原始信号的大部分能量
    """
    # 计算信号能量
    signal_energy = np.sum(signal**2)
    original_energy = np.sum(original_signal**2)
    
    # 避免除零错误
    if original_energy == 0:
        return 0
    
    return signal_energy / original_energy

# 计算信号相关性
def signal_correlation(signal, original_signal):
    """
    计算重构信号与原始信号的相关系数
    
    参数:
    signal (array_like): 重构的信号
    original_signal (array_like): 原始信号
    
    返回值:
    float: 相关系数，值越接近1表示重构信号与原始信号越相似
    """
    return np.abs(np.corrcoef(signal, original_signal)[0, 1])

# IMF特征提取函数（从imf_feature_extraction.py移植）
def calculate_imf_features(imfs):
    """
    计算IMF模态信号的特征指标
    
    参数:
    imfs (numpy.ndarray): IMF模态数组，形状为(K, N)，K为模态数量，N为信号长度
    
    返回值:
    pandas.DataFrame: 包含各模态特征指标的结果表格，包含峭度、能量、偏度和功率谱熵
    """
    features = []
    
    # 遍历所有模态
    for i in range(imfs.shape[0]):
        signal = imfs[i]
        
        # 计算时域统计特征
        kurt = kurtosis(signal, fisher=False)  # 峰态系数（Fisher定义时为False）
        energy = np.sum(signal**2)  # 信号能量
        skewness = skew(signal)     # 偏度系数
        
        # 计算频域特征
        entropy_val = spectral_entropy(signal)  # 功率谱熵
        
        features.append({
            '模态': f'模态{i+1}',
            '峭度': kurt,
            '能量': energy,
            '偏度': skewness,
            '功率谱熵': entropy_val
        })
    
    return pd.DataFrame(features)

# IMF聚类函数（从imf_clustering.py移植）
def cluster_imfs(feature_df, n_clusters=2):
    """
    对IMF特征进行K-Means聚类分析
    
    参数：
    feature_df (pd.DataFrame): 包含IMF特征的DataFrame
    n_clusters (int): 要形成的簇数，默认为2
    
    返回值：
    tuple: (pd.DataFrame, np.ndarray)
        - feature_df: 包含原始特征和聚类标签的DataFrame
        - centers_original: 原始尺度下的聚类中心坐标数组
    """
    # 提取四维特征矩阵（峭度、能量、偏度、功率谱熵）
    X = feature_df[['峭度', '能量', '偏度', '功率谱熵']].values
    
    # 数据标准化处理（Z-score标准化）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 执行K-Means聚类算法
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 将聚类标签合并到特征数据集
    feature_df = feature_df.copy()
    feature_df['聚类标签'] = clusters
    
    # 计算原始尺度的聚类中心
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    
    return feature_df, centers_original

# 识别有效IMF函数
def identify_valid_imfs(feature_df):
    """
    根据聚类结果识别有效IMF
    
    参数：
    feature_df (pd.DataFrame): 包含特征数据和聚类标签的DataFrame
    
    返回值：
    list: 有效IMF的索引列表
    """
    # 计算每个簇的平均能量
    cluster_energy = feature_df.groupby('聚类标签')['能量'].mean()
    
    # 能量较高的簇通常对应有效IMF
    valid_cluster = cluster_energy.idxmax()
    
    # 获取有效IMF的索引
    valid_indices = []
    for i, row in feature_df.iterrows():
        if row['聚类标签'] == valid_cluster:
            # 从'模态X'中提取索引
            mode_str = row['模态']
            if isinstance(mode_str, str) and mode_str.startswith('模态'):
                try:
                    idx = int(mode_str[2:]) - 1  # '模态1' -> 0
                    valid_indices.append(idx)
                except ValueError:
                    pass
    
    return valid_indices

# VMD参数优化问题定义
class VMDOptimizationProblem(Problem):
    def __init__(self, bounds=None, minmax="max", data=None, objective_type="sample_entropy", **kwargs):
        self.data = data
        self.objective_type = objective_type
        super().__init__(bounds, minmax, **kwargs)
    
    def obj_func(self, x):
        """
        目标函数：根据选择的优化目标计算适应度值
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
        
        # 计算IMF特征
        feature_df = calculate_imf_features(u)
        
        # 对IMF进行聚类
        feature_df, _ = cluster_imfs(feature_df)
        
        # 识别有效IMF
        valid_indices = identify_valid_imfs(feature_df)
        
        # 如果没有有效IMF，则使用所有IMF
        if not valid_indices:
            valid_indices = list(range(K))
        
        # 使用有效IMF重构信号
        valid_imfs = np.zeros_like(u[0])
        for idx in valid_indices:
            valid_imfs += u[idx]
        
        # 根据选择的目标函数计算适应度值
        if self.objective_type == "sample_entropy":
            # 样本熵（值越小表示信号越规则，去噪效果越好）
            value = sample_entropy(valid_imfs)
            return -value  # 负值用于最大化问题
            
        elif self.objective_type == "permutation_entropy":
            # 排列熵（值越小表示信号越规则，去噪效果越好）
            value = permutation_entropy(valid_imfs)
            return -value  # 负值用于最大化问题
            
        elif self.objective_type == "kurtosis":
            # 峭度（值越大表示信号中的脉冲成分越明显，通常噪声的峭度较小）
            value = kurtosis(valid_imfs, fisher=False)
            return value  # 直接返回，因为我们要最大化峭度
            
        elif self.objective_type == "spectral_entropy":
            # 功率谱熵（值越小表示频谱分布越集中，通常去噪后的信号频谱更集中）
            value = spectral_entropy(valid_imfs)
            return -value  # 负值用于最大化问题
            
        elif self.objective_type == "singular_spectrum_entropy":
            # 奇异谱熵（值越小表示信号的复杂度越低，通常去噪后的信号复杂度降低）
            value = singular_spectrum_entropy(valid_imfs)
            return -value  # 负值用于最大化问题
            
        elif self.objective_type == "energy_entropy":
            # 能量熵（值越小表示信号能量分布越集中，通常去噪后的信号能量分布更集中）
            value = energy_entropy(valid_imfs)
            return -value  # 负值用于最大化问题
            
        elif self.objective_type == "approximate_entropy":
            # 近似熵（值越小表示信号的复杂度和不规则性越低）
            value = approximate_entropy(valid_imfs)
            return -value  # 负值用于最大化问题
            
        elif self.objective_type == "fuzzy_entropy":
            # 模糊熵（值越小表示信号的复杂度和不规则性越低）
            value = fuzzy_entropy(valid_imfs)
            return -value  # 负值用于最大化问题
            
        elif self.objective_type == "correlation":
            # 相关性（值越大表示重构信号与原始信号越相似）
            value = signal_correlation(valid_imfs, self.data)
            return value  # 直接返回，因为我们要最大化相关性
            
        elif self.objective_type == "energy_ratio":
            # 能量比（值越接近1表示重构信号保留了原始信号的大部分能量）
            value = energy_ratio(valid_imfs, self.data)
            # 我们希望能量比接近1，但不超过1
            if value > 1:
                return 2 - value  # 惩罚能量比过大的情况
            return value  # 直接返回，因为我们要最大化能量比
            
        elif self.objective_type == "sparsity":
            # 稀疏性（值越大表示信号越稀疏，通常去噪后的信号更稀疏）
            value = sparsity_measure(valid_imfs)
            return value  # 直接返回，因为我们要最大化稀疏性
            
        else:
            # 默认使用样本熵
            value = sample_entropy(valid_imfs)
            return -value  # 负值用于最大化问题

def optimize_vmd_parameters(objective_type="sample_entropy"):
    """
    使用SMA算法优化VMD参数
    
    参数:
    objective_type: 使用的目标函数类型，可选值包括：
        - 'sample_entropy': 样本熵（值越小表示信号越规则）
        - 'permutation_entropy': 排列熵（值越小表示信号越规则）
        - 'kurtosis': 峭度（值越大表示信号中的脉冲成分越明显）
        - 'spectral_entropy': 功率谱熵（值越小表示频谱分布越集中）
        - 'singular_spectrum_entropy': 奇异谱熵（值越小表示信号的复杂度越低）
        - 'energy_entropy': 能量熵（值越小表示信号能量分布越集中）
        - 'approximate_entropy': 近似熵（值越小表示信号的复杂度和不规则性越低）
        - 'fuzzy_entropy': 模糊熵（值越小表示信号的复杂度和不规则性越低）
        - 'correlation': 相关性（值越大表示重构信号与原始信号越相似）
        - 'energy_ratio': 能量比（值越接近1表示重构信号保留了原始信号的大部分能量）
        - 'sparsity': 稀疏性（值越大表示信号越稀疏）
    
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
        FloatVar(lb=100, ub=10000, name="alpha")
    ]
    
    # 创建优化问题
    problem = VMDOptimizationProblem(bounds=my_bounds, minmax="max", data=amplitude, objective_type=objective_type)
    
    # 创建SMA优化器
    model = SSA.OriginalSSA(epoch=100, pop_size=10)
    
    # 创建列表用于记录每次迭代的最佳适应度值
    fitness_history = []
    
    # 求解优化问题
    model.solve(problem)
    
    # 从模型的历史记录中获取每次迭代的最佳适应度值
    # 对于熵类型的目标函数，我们使用的是负熵值（最大化问题），需要取负数得到实际熵值
    # 对于其他类型的目标函数，直接使用适应度值
    if objective_type in ["sample_entropy", "permutation_entropy", "spectral_entropy", "singular_spectrum_entropy", "energy_entropy", "approximate_entropy", "fuzzy_entropy"]:
        fitness_history = [-fit for fit in model.history.list_current_best_fit]
    else:
        fitness_history = [fit for fit in model.history.list_current_best_fit]
    
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
    
    # 计算IMF特征
    feature_df = calculate_imf_features(u)
    
    # 对IMF进行聚类
    feature_df, centers = cluster_imfs(feature_df)
    
    # 识别有效IMF
    valid_indices = identify_valid_imfs(feature_df)
    
    # 如果没有有效IMF，则使用所有IMF
    if not valid_indices:
        valid_indices = list(range(K_optimal))
        print("警告：未找到有效IMF，使用所有IMF")
    else:
        print(f"识别到的有效IMF索引: {valid_indices}")
    
    # 保存聚类结果
    feature_df.to_excel('../save/imf_features.xlsx', index=False)
    
    # 保存聚类标签
    labels_df = pd.DataFrame({
        '模态': feature_df['模态'],
        '聚类标签': feature_df['聚类标签']
    })
    labels_df.to_excel('../save/cluster_labels.xlsx', index=False)
    
    # 保存聚类中心
    centers_df = pd.DataFrame(centers,
                columns=['峭度', '能量', '偏度', '功率谱熵'],
                index=[f'中心{i}' for i in range(len(centers))])
    centers_df.to_excel('../save/cluster_centers.xlsx')
    
    # 保存最优参数
    result_df = pd.DataFrame({
        '参数': ['K', 'alpha'],
        '最优值': [K_optimal, alpha_optimal]
    })
    result_df.to_excel('../save/optimal_vmd_parameters.xlsx', index=False)
    
    # 保存有效IMF
    time = data.iloc[:, 0].values
    time_aligned = time[:u.shape[1]]
    
    # 创建保存有效IMF的DataFrame
    optimal_imfs_df = pd.DataFrame({'时间': time_aligned})
    for idx in valid_indices:
        optimal_imfs_df[f'模态{idx+1}'] = u[idx]
    optimal_imfs_df.to_excel('../save/optimal_IMFs.xlsx', index=False)
    
    # 可视化优化结果
    plt.figure(figsize=(15, 10))
    
    # 原始信号
    plt.subplot(len(valid_indices)+2, 1, 1)
    plt.plot(time, amplitude, 'b')
    plt.title('原始信号')
    plt.xlabel('时间')
    plt.ylabel('幅值')
    
    # 有效IMF重构信号
    valid_imfs = np.zeros_like(u[0])
    for idx in valid_indices:
        valid_imfs += u[idx]
    
    plt.subplot(len(valid_indices)+2, 1, 2)
    plt.plot(time_aligned, valid_imfs, 'r')
    plt.title('有效IMF重构信号')
    plt.xlabel('时间')
    plt.ylabel('幅值')
    
    # 各有效模态分量
    for i, idx in enumerate(valid_indices):
        plt.subplot(len(valid_indices)+2, 1, i+3)
        plt.plot(time_aligned, u[idx], 'g')
        plt.title(f'有效模态 {idx+1}')
        plt.xlabel('时间')
        plt.ylabel('幅值')
    
    plt.tight_layout()
    plt.show()
    
    # 可视化聚类结果
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111, projection='3d')
    scatter = ax.scatter(feature_df['峭度'],
                         feature_df['能量'],
                         feature_df['偏度'],
                         c=feature_df['聚类标签'],
                         cmap='viridis',
                         s=50)
    ax.set_xlabel('峭度')
    ax.set_ylabel('能量')
    ax.set_zlabel('偏度')
    plt.title('IMF特征三维聚类分布')
    plt.show()
    
    # 绘制适应度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history, 'b-', marker='o')
    plt.title(f'{objective_type}优化过程适应度曲线')
    plt.xlabel('迭代次数')
    plt.ylabel(f'适应度值')
    plt.grid(True)
    plt.show()
    
    # 保存适应度历史数据
    fitness_df = pd.DataFrame({
        '迭代次数': range(1, len(fitness_history) + 1),
        f'适应度值': fitness_history
    })
    fitness_df.to_excel('../save/fitness_history.xlsx', index=False)
    
    # 根据目标函数类型返回适当的适应度值
    if objective_type in ["sample_entropy", "permutation_entropy", "spectral_entropy", "singular_spectrum_entropy", "energy_entropy", "approximate_entropy", "fuzzy_entropy"]:
        return best_parameters, -best_fitness  # 对于熵类型，返回负值的负数（即原始熵值）
    else:
        return best_parameters, best_fitness  # 对于其他类型，直接返回适应度值

if __name__ == "__main__":
    # 确保导入sklearn库
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("请先安装sklearn库: pip install scikit-learn")
        exit(1)
        
    # 用户选择使用的目标函数类型
    print("请选择使用的目标函数类型:")
    print("1. 样本熵 (Sample Entropy) - 值越小表示信号越规则")
    print("2. 排列熵 (Permutation Entropy) - 值越小表示信号越规则")
    print("3. 峭度 (Kurtosis) - 值越大表示信号中的脉冲成分越明显")
    print("4. 功率谱熵 (Spectral Entropy) - 值越小表示频谱分布越集中")
    print("5. 奇异谱熵 (Singular Spectrum Entropy) - 值越小表示信号的复杂度越低")
    print("6. 能量熵 (Energy Entropy) - 值越小表示信号能量分布越集中")
    print("7. 近似熵 (Approximate Entropy) - 值越小表示信号的复杂度和不规则性越低")
    print("8. 模糊熵 (Fuzzy Entropy) - 值越小表示信号的复杂度和不规则性越低")
    print("9. 相关性 (Correlation) - 值越大表示重构信号与原始信号越相似")
    print("10. 能量比 (Energy Ratio) - 值越接近1表示重构信号保留了原始信号的大部分能量")
    print("11. 稀疏性 (Sparsity) - 值越大表示信号越稀疏")
    
    choice = input("请输入选择 (1-11): ")
    
    objective_types = {
        "1": "sample_entropy",
        "2": "permutation_entropy",
        "3": "kurtosis",
        "4": "spectral_entropy",
        "5": "singular_spectrum_entropy",
        "6": "energy_entropy",
        "7": "approximate_entropy",
        "8": "fuzzy_entropy",
        "9": "correlation",
        "10": "energy_ratio",
        "11": "sparsity"
    }
    
    if choice in objective_types:
        print(f"\n使用 {objective_types[choice]} 作为优化目标...")
        best_params, best_fitness = optimize_vmd_parameters(objective_types[choice])
        print("\n优化完成!")
        print(f"最优参数: {best_params}")
        print(f"最优适应度值: {best_fitness}")
        print("\n结果已保存到save目录下的相关文件中。")
    else:
        print("无效选择，使用默认的样本熵作为优化目标。")
        optimize_vmd_parameters()