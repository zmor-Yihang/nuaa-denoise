import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.stats import kurtosis, skew
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

    return -np.sum(psd_norm * np.log2(psd_norm))  # 香农熵公式计算

def calculate_features(df):
    """
    计算DataFrame中各模态信号的特征指标

    参数:
    df (DataFrame): 包含时间列和各IMF模态列的输入数据

    返回值:
    DataFrame: 包含各模态特征指标的结果表格，包含峭度、能量、偏度和功率谱熵
    """
    features = []
    # 遍历所有模态列（跳过首列时间列）
    for col in df.columns[1:]:
        signal = df[col].values

        # 计算时域统计特征
        kurt = kurtosis(signal, fisher=False)  # 峰态系数（Fisher定义时为False）
        energy = np.sum(signal**2)  # 信号能量
        skewness = skew(signal)     # 偏度系数

        # 计算频域特征
        entropy = spectral_entropy(signal)  # 功率谱熵

        features.append({
            '模态': col,
            '峭度': kurt,
            '能量': energy,
            '偏度': skewness,
            '功率谱熵': entropy
        })

    return pd.DataFrame(features)

if __name__ == '__main__':
    """主处理流程：特征提取、可视化与结果保存"""
    # 读取IMF分解结果
    imf_df = pd.read_excel('../save/IMFs.xlsx')

    # 特征矩阵计算
    feature_df = calculate_features(imf_df)

    # 多子图分布可视化
    feature_df.set_index('模态').plot(kind='bar', subplots=True, layout=(2,3))
    plt.suptitle('IMF特征分布')
    plt.tight_layout()  # 自动调整子图间距
    plt.show()

    # 结果持久化
    feature_df.to_excel('../save/imf_features.xlsx', index=False)
    print('特征分析完成，结果已保存至imf_features.xlsx')
