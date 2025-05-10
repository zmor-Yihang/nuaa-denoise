import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from imf_feature_extraction import calculate_features

# 聚类分析主函数
def cluster_imfs(n_clusters=2):
    """
    对IMF特征进行K-Means聚类分析

    参数：
    n_clusters (int): 要形成的簇数，默认为2

    返回值：
    tuple: (pd.DataFrame, np.ndarray)
        - feature_df: 包含原始特征和聚类标签的DataFrame
        - centers_original: 原始尺度下的聚类中心坐标数组
    """
    # 读取特征数据
    feature_df = pd.read_excel('../save/imf_features.xlsx')

    # 提取四维特征矩阵（峭度、能量、偏度、功率谱熵）
    X = feature_df[['峭度', '能量', '偏度', '功率谱熵']].values

    # 数据标准化处理（Z-score标准化）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 执行K-Means聚类算法
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # 保存带模态名称的聚类标签到Excel
    labels_df = pd.DataFrame({
        '模态': feature_df['模态'],
        '聚类标签': clusters
    })
    labels_df.to_excel('../save/cluster_labels.xlsx', index=False)

    # 将聚类标签合并到特征数据集
    feature_df = pd.concat([feature_df, labels_df['聚类标签']], axis=1)

    # 计算并保存原始尺度的聚类中心
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    pd.DataFrame(centers_original,
                columns=['峭度', '能量', '偏度', '功率谱熵'],
                index=[f'中心{i}' for i in range(n_clusters)])\
        .to_excel('../save/cluster_centers.xlsx')

    return feature_df, centers_original

# 可视化函数
def visualize_clusters(feature_df):
    """
    可视化展示聚类结果

    参数：
    feature_df (pd.DataFrame): 包含特征数据和聚类标签的DataFrame

    返回值：
    None
    """
    # 创建三维散点图展示前三个特征
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

    # 绘制二维特征组合的散点图矩阵
    features = ['峭度', '能量', '偏度', '功率谱熵']
    plt.figure(figsize=(15, 10))
    for i, combo in enumerate([(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)], 1):
        plt.subplot(2,3,i)
        plt.scatter(feature_df.iloc[:, combo[0]+1],
                    feature_df.iloc[:, combo[1]+1],
                    c=feature_df['聚类标签'],
                    cmap='viridis')
        plt.xlabel(features[combo[0]])
        plt.ylabel(features[combo[1]])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 执行特征计算流程
    imf_df = pd.read_excel('../save/IMFs.xlsx')
    feature_df = calculate_features(imf_df)
    feature_df.to_excel('../save/imf_features.xlsx', index=False)

    # 执行完整聚类分析流程
    clustered_df, centers = cluster_imfs()

    # 展示聚类可视化结果
    visualize_clusters(clustered_df)

    print('聚类分析完成，结果已保存至imf_features.xlsx和cluster_centers.xlsx')
