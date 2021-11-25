# “未组队 组队私聊队”方案

本方案在百度 [Baseline](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_matching/question_matching) 的基础上进行改进，逐渐优化，得到最终 B 榜第 3 名的成绩，分数为 91.837。

使用的深度学习框架为 PaddlePaddle，预训练模型为 ERNIE-Gram。在比赛的最后方案中，我只使用了单个 ERNIE-Gram 模型，没有进行模型集成。

数据集使用的是主办方提供的全部训练集，并参考 [ACL 2020 最佳论文](https://aclanthology.org/2020.acl-main.442.pdf) 的方法，用模板构造了几十个样本。在最后也使用了 [伪标签](https://www.researchgate.net/publication/280581078_Pseudo-Label_The_Simple_and_Efficient_Semi-Supervised_Learning_Method_for_Deep_Neural_Networks) 的方式进行半监督迭代训练。

使用的开发环境是百度 AI Studio V100 16G。



## 文件介绍

`train.sh`：运行这个 bash 文件进行模型训练

`predict.sh`：运行这个 bash 文件进行模型预测

`train.py`：模型训练文件，主要参数如下。

- `--layerwise_decay`：学习率逐层衰减比率
- `--LSR_coef`：标签平滑系数
- `--rdrop_coef`：R-Drop 系数
- `--fgm_coef`：对抗训练时添加扰动的范数大小
- `--fgm_alpha`：对抗训练 Loss 的占比
- `--clip_norm`：梯度裁剪
- `--lookahead_k`：Lookahead 优化器 Fast 权重更新次数
- `--lookahead_alpha`：Lookahead 优化器 Slow 权重更新步长
- `--lazy_embedding`：是否对最底层的 Embedding 进行稀疏更新
- `--amp`：是否开启混合精度训练

`predict.py`：用模型预测测试集

`generate_sample.py`：参考 ACL 2020 最佳论文，用模板构造样本

`data_preprocess.py`：数据预处理，整合文件

`data.py`：加载数据文件，把输入文本变成模型可以处理的形式

`model.py`：模型文件，包括问题匹配分类模型、Embedding 稀疏更新模块、对抗训练模块、最终未使用的拼音层

`loss.py`：损失函数文件，包括标签平滑、R-Drop

`lookahead.py`：复现了 [Lookahead](https://arxiv.org/abs/1907.08610) 优化器

`postprocess.py`：后处理，包括 [CAN](https://arxiv.org/abs/2109.13449)，频率副词、时态、错字、语法、数字、填充词检测和结果纠正

`similar_character.npy`：形近字 DIct，用于错字检测。来源于 [GitHub 开源项目](https://github.com/contr4l/SimilarCharacter/blob/master/%E5%BD%A2%E8%BF%91%E5%AD%97%E8%AF%AD%E6%96%99%E5%BA%93%EF%BC%88CV2%EF%BC%89.txt)，我把它从 txt 格式转成了 npy 格式，内容一样

`syntactic_analysis.npy`：语法分析 Dict。用 HanLP 和 jieba.posseg 对测试集部分样本进行了语法解析，将解析结果存到了 npy 文件中



## 训练

```bash
bash train.sh
```

如果是单卡 V100，预计需要训练 12 个小时。代码未测试过单机多卡或多机多卡训练。训练结束后会自动生成 B 榜预测结果，无需再单独预测一遍。



## 预测

```bash
bash predict.sh
```

用训练好的 checkpoint 进行预测，预计 2 分钟就好。
