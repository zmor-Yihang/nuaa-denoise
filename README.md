# 信号去噪与特征分析系统

## 文件说明
1. `vmd_saveImf.py` - 原始信号vmd分解
2. `imf_feature_extraction.py` - IMF特征提取模块,计算峭度、能量、偏度、功率谱熵
3. `imf_clustering.py` - Imf 聚类分析,分成噪声模态和正常模态
4. `signal_reconstruction.py` - 重构信号
5. `run_pipeline.py` - 流程控制主文件

## 执行流程
### 方式一：分步执行
1 -> 2 -> 3 -> 4

### 方式二：一键执行
直接运行run_pipeline.py自动执行完整流程