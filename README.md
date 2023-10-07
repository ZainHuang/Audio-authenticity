# Audio-authenticity
2020之江杯全球人工智能大赛，语音鉴伪挑战赛TOP3方案


![](https://github.com/ZainHuang/Audio-authenticity/blob/main/images/%E3%80%90%E5%87%8C%E8%93%9D%E9%A3%8E%E3%80%91%E3%80%90%E9%9F%B3%E9%A2%91%E8%B5%9B%E9%81%93%E3%80%912020%E4%B9%8B%E6%B1%9F%E6%9D%AF%E5%85%A8%E7%90%83%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%A4%A7%E8%B5%9B.png)


运行环境
Linux版本：
Linux fuxilabor_labor0_S4_Odps_S96_dsw_prepaid_cnsh_891_2020100812331 4.9.65 #5 SMP Fri Mar 30 15:59:08 CST 2018 x86_64 x86_64 x86_64 GNU/Linux

Python版本：
Python 3.6.4 |Anaconda, Inc.| (default, Jan 16 2018, 18:10:19)
[GCC 7.2.0] on linux

GPU：
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.48                 Driver Version: 410.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+===================|

CUDA版本：
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130

CUDNN版本：
#define CUDNN_MAJOR 7
#define CUDNN_MINOR 6
#define CUDNN_PATCHLEVEL 3
#define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
#include "driver_types.h"

依赖包：
torch                     1.4.0+cu100
torchtext                 0.6.0
torchvision               0.5.0
six                       1.11.0
scikit-learn              0.23.2
librosa                            0.7.2
python-speech-features             0.6
joblib                             0.13.2


模型训练预测：
1、先运行python mian.py --ACTTION=feature生成特征，特征存放于user_dara/feature文件夹中
2、运行python mian.py --ACTTION=train，训练模型，训练完成后会在user_data目录下生成best_acc.txt文档记录了交叉验证后最好的5个模型编号
3、再运行python mian.py --ACTTION=train，输出预测结果
4、mian.py中的已固定随机因子，理论上在相同环境下可完美复现，随机因子固定如下
seed = 2020
np.random.seed(seed)
torch.manual_seed(seed)#为CPU设置随机种子
torch.cuda.manual_seed_all(seed)#为所有GPU设置随机种子
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic =True

