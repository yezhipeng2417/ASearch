# Overview
qq匹配、相似文本打分、文本检索重排方案的探索与实现

# Details
## 模型文件：
1. simbert
locate：model_file
model file: https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_simbert_L-12_H-768_A-12.zip
出处：https://github.com/ZhuiyiTechnology/pretrained-models

## 数据文件：
1. 摘要数据集
locate：data/
目前在用dataset：https://dataset-bj.cdn.bcebos.com/qianyan/LCSTS_new.zip
出处：千言数据 https://www.luge.ai/#/

2. 相似文本数据集
locate：data/
短文本相似度dataset：https://dataset-bj.cdn.bcebos.com/qianyan/lcqmc.zip
出处：千言数据 https://www.luge.ai/#/


## 目前方案：
ctr+cqr+simbert+xgb融合

## TODO list：
* 并行抽取特征，模型预测 p0
* ffm替换xgb测试效果 p1
* 扩充数据集 （长文本数据集、标题信息等）p2
* 扩充特征（bert上下文相似度等）p2