import pandas as pd
import matplotlib.pyplot as plt
from vmdpy import VMD

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取Excel数据
data = pd.read_excel('../data/Thz_data.xlsx')
time = data.iloc[:, 0].values
amplitude = data.iloc[:, 1].values

# VMD参数设置
alpha = 2000
tau = 0
K = 8
DC = 1
init = 1
tol = 1e-7

# 执行VMD分解
u, _, _ = VMD(amplitude, alpha, tau, K, DC, init, tol)

# 保存分解结果到新Excel文件
# 调整时间序列长度与模态分量对齐
time_aligned = time[:u.shape[1]]
result_df = pd.DataFrame({'时间': time_aligned})
for i in range(K):
    result_df[f'模态{i+1}'] = u[i]
result_df.to_excel('../save/IMFs.xlsx', index=False)

# 可视化原始信号和所有模态
plt.figure(figsize=(15, 10))

# 原始信号
plt.subplot(K+1, 1, 1)
plt.plot(time, amplitude, 'b')
plt.title('原始信号')
plt.xlabel('时间')
plt.ylabel('幅值')

# 各模态分量
for i in range(K):
    plt.subplot(K+1, 1, i+2)
    plt.plot(time_aligned, u[i], 'g')
    plt.title(f'模态 {i+1}')
    plt.xlabel('时间')
    plt.ylabel('幅值')

plt.tight_layout()
plt.show()
