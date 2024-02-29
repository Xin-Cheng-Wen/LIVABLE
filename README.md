<div align="center">
    <p>
    <h1>
    LIVABLE - Implementation
    </h1>
    <a href="https://github.com/ddlBoJack/MT4SSL"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/ddlBoJack/MT4SSL"><img src="https://img.shields.io/badge/Python-3.8+-orange" alt="version"></a>
    <a href="https://github.com/ddlBoJack/MT4SSL"><img src="https://img.shields.io/badge/License-MIT-red.svg" alt="mit"></a>
</div>


## 📥 Guide

#### 1、Preprocessing

- (1) **Joern**: 
  We download Joern to generate the code structure graph and we provide a compiled version of joern [here](https://zenodo.org/record/7323504#.Y3OQL3ZByUk). 

- (2) **Parse**: 
  We use the `preprocessing\process.py` to use Joern.

#### 2、Word2Vec
For each code structure graph, we use the word2vec to initialize the node representation in the graph branch and the token representation
in the sequence branch.

-  (3) **Word2Vec Training**:
  We use the `preprocessing\word2vec_multi.py` to train the word2vec model.
  
-  (4) **Node and Token Representation**:
  We use the `preprocessing\ori_ourdevign+token.py` to generate the node representation and the token representation.
  

#### 3、Training the LIVABLE model


-  (5) **The vulnerability detection model's training configs**:
  batch_size = 64, lr = 0.0001, epoch = 100, patience = 20
  opt ='RAdam', weight_decay=1e-6, class_num =2
-  (6) **The vulnerability type classification model's training configs**:
  
  batch_size = 64, lr = 0.0001, epoch = 50, patience = 20, opt ='RAdam', weight_decay=1e-6, class_num = 31

-  (7) **Model Training**: 
The model implementation code is under the `code\` folder. The model can be runned from `code\main_sta.py`.

## 🚨 Abstract

In this paper, we propose a long-tailed software vulnerability type classification approach, called LIVABLE. LIVABLE mainly consists of two modules, including (1) vulnerability representation learning module, which improves the propagation steps in GNN to distinguish node representations by a differentiated propagation method. A sequence-to-sequence model is also involved to enhance the vulnerability representations. (2) adaptive re-weighting module, which adjusts the learning weights for different types according to the training epochs and numbers of associated samples by a novel training loss. We verify the effectiveness of LIVABLE in both type classification and vulnerability detection tasks. For vulnerability type classification, the experiments on the Fan et al. dataset show that LIVABLE outperforms the state-of-the-art methods by 24.18% in terms of the accuracy metric, and also improves the performance in predicting tails by 7.7%. To evaluate the efficacy of the vulnerability representation learning module in LIVABLE, we further compare with the recent vulnerability detection approaches on three benchmark datasets, which shows that the proposed representation learning module improves the best baselines by 4.03% on average in terms of accuracy.

## 🤯 Dataset

To investigate the effectiveness of LIVABLE in vulnerability detection, we adopt three vulnerability datasets from these paper:

- Fan et al. [1]: 
```bash
https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing
```

- Reveal [2]: 
```bash
https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOyF
```

- FFMPeg+Qemu [3]: 
```bash
https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
```

In vulnerability type classification, we extract a new dataset from Fan et al., which is in `data\` folder.

## 📅 Requirement

Our code is based on Python3 (>= 3.7). There are a few dependencies to run the code. The major libraries are listed as follows:

- torch (==1.9.0)
- dgl (==0.7.2)
- numpy (==1.22.3)
- sklearn (==0.0)
- pandas (==1.4.1)
- tqdm





**References**：

[1] Jiahao Fan, Yi Li, Shaohua Wang, and Tien Nguyen. 2020. A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries. In The 2020 International Conference on Mining Software Repositories (MSR). IEEE.

[2] Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, and Baishakhi Ray. 2020. Deep Learning based Vulnerability Detection: Are We There Yet? arXiv preprint arXiv:2009.07235 (2020).

[3] Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in Neural Information Processing Systems. 10197–10207.
