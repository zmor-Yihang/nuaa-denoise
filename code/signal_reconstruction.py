import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 评估指标计算函数
def calculate_metrics(original, denoised):
    # 信噪比（SNR）
    noise = original - denoised
    snr = 10 * np.log10(np.sum(denoised**2) / (np.sum(noise**2) + 1e-12))
    
    # 均方误差（MSE）
    mse = np.mean((original - denoised)**2)
    
    # 新增绝对平均误差（MAE）
    mae = np.mean(np.abs(original - denoised))  # 新增MAE计算
    
    # 峰值信噪比（PSNR）改进：使用信号的绝对最大值
    max_signal = np.max(np.abs(original))  # 修正为绝对最大值
    psnr = 10 * np.log10((max_signal**2) / (mse + 1e-12))
    
    # 均方根误差（MSRE）
    msre = np.sqrt(mse)

    
    # 新实现：均值中心化后计算样本协方差和样本标准差
    original_centered = original - np.mean(original)
    denoised_centered = denoised - np.mean(denoised)
    covariance = np.dot(original_centered, denoised_centered) / (len(original)-1)  # 样本协方差
    std_product = np.std(original, ddof=1) * np.std(denoised, ddof=1)  # 样本标准差
    ncc = covariance / (std_product + 1e-12)

    return {'SNR(dB)': snr,
            'PSNR(dB)': psnr,
            'MSE': mse,
            'MSRE': msre,
            'MAE': mae,  # 新增MAE指标
            'NCC': ncc}

# 主处理函数
def reconstruct_and_evaluate():
    # 读取聚类标签
    labels_df = pd.read_excel('../save/cluster_labels.xlsx')
    valid_imfs = labels_df[labels_df['聚类标签'] == 1]['模态'].tolist()
    
    # 读取IMF数据
    imf_df = pd.read_excel('../save/IMFs.xlsx')
    time = imf_df.iloc[:, 0].values
    
    # 重构有效信号
    if not valid_imfs:
        print("警告：未找到有效IMF")
        return None
    
    valid_columns = ['时间'] + valid_imfs
    denoised = imf_df[valid_imfs].sum(axis=1).values
    
    # 插值处理（复用vmd-kmeans逻辑）
    if len(denoised) != len(time):
        interpolator = interp1d(np.linspace(0, 1, len(denoised)), denoised, kind='linear')
        denoised = interpolator(np.linspace(0, 1, len(time)))
    
    # 计算评估指标
    original = imf_df.iloc[:,1:].sum(axis=1).values  # 原始合成信号
    metrics = calculate_metrics(original, denoised)
    
    # 保存评估结果
    pd.DataFrame([metrics]).to_excel('../save/evaluation_metrics.xlsx', index=False)
    
    # 可视化对比
    plt.figure(figsize=(12, 6))
    plt.plot(time, original, alpha=0.5, label='原始信号')
    plt.plot(time, denoised, 'r--', linewidth=2, label='降噪信号')
    plt.legend()
    plt.title('信号降噪效果对比')
    plt.xlabel('时间')
    plt.ylabel('幅值')
    plt.show()
    
    return metrics

if __name__ == '__main__':
    result = reconstruct_and_evaluate()
    if result:
        print("评估指标：")
        for k, v in result.items():
            print(f"{k}: {v:.4f}")
