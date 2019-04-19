这个Repo的主要目的是从最新的论文集中构建一个自然语言处理的工具集，目前实现的算法有ESIM和Transformer。后续计划实现的算法是MTDNN和Semi-Supervised sequence tag

ESIM还存在一些Bug，再测试时发现分类器会倾向分类到不相似，可能是mask部分有问题，近期会修复

进度：

Transformer完成了基本框架，训练已经完成，需要测试mask的效果
后续实现算法参考以下论文：

完成了semi-supervised sequence tag的unlabel数据部分

https://arxiv.org/pdf/1705.00108.pdf

https://arxiv.org/pdf/1901.11504.pdf
