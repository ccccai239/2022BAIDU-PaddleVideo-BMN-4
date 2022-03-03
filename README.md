
# 基于飞桨实现乒乓球时序动作定位大赛 ：B榜第4名方案
##
github中空间有限，此项目中data与算法模型文件（PaddleVideo）参见https://aistudio.baidu.com/aistudio/projectdetail/3544826?contributionType=1 下载

## 赛题介绍
时序动作定位(提案生成)是计算机视觉和视频分析领域一个具有的挑战性的任务。本次比赛不同于以往的ActivityNet-TAL，FineAction等视频时序检测动作定位比赛，我们采用了更精细的动作数据集--乒乓球转播画面，该数据集具有动作时间跨度短，分布密集等特点，给传统模型精确定位细粒度动作带来了很大挑战。本次比赛的任务即针对乒乓球转播画面视频面对镜头的运动员定位其动作(时序动作提案生成)。

## 数据集介绍
本次比赛的数据集包含了19-21赛季兵乓球国际比赛（世界杯、世锦赛、亚锦赛，奥运会）和国内比赛（全运会，乒超联赛）中标准单机位高清转播画面的特征信息，共包含912条视频特征文件，每个视频时长在0～6分钟不等，特征维度为2048，以pkl格式保存。我们对特征数据中面朝镜头的运动员的回合内挥拍动作进行了标注，单个动作时常在0～2秒不等，其视频帧率为25帧/秒。训练数据为729条标注视频，B榜测试数据为92条视频，训练数据标签以json格式给出。
*标签格式如下：![](https://ai-studio-static-online.cdn.bcebos.com/ec8317f3d07449aca11bde656e72b7f9ce8f8616cf5b4ec386743a8f8e542a6e)*
*特征数据如下：![](https://ai-studio-static-online.cdn.bcebos.com/8384c3f2f6a24d8d82ffeec3a44644cefaf75e6809bf47cbabab4a55f2b6747e)*

## 项目整体架构
* 该项目主要分为数据处理、模型训练、模型导出以及推理预测四部分。
* 在数据处理上，训练集与测试集采用不同的分割方式。
* 在模型构建上，本方案采用PaddleVideo中的BMN模型。此模型引入边界匹配(Boundary-Matching, BM)机制来评估proposal的置信度，按照proposal开始边界的位置及其长度将所有可能存在的proposal组合成一个二维的BM置信度图，图中每个点的数值代表其所对应的proposal的置信度分数。网络由三个模块组成，基础模块作为主干网络处理输入的特征序列，TEM模块预测每一个时序位置属于动作开始、动作结束的概率，PEM模块生成BM置信度图。
![](https://ai-studio-static-online.cdn.bcebos.com/1a8deac1f6fb4f7abc6ff65b692a9e16c76ccb6d890949ea98aa1324f0df30dd)



## 数据处理
本赛题中的数据包含912条ppTSM抽取的视频特征，特征保存为pkl格式，文件名对应视频名称，读取pkl之后以(num_of_frames, 2048)向量形式代表单个视频特征。其中num_of_frames是不固定的，同时数量也比较大，所以pkl的文件并不能直接用于训练。同时由于乒乓球每个动作时间非常短，为了可以让模型更好的识别动作，所以这里将数据进行分割。

### 1.解压数据集
执行以下命令解压数据集，解压之后将压缩包删除，保证项目空间小于100G。否则项目会被终止。


```python
%cd /home/aistudio/data/
!tar xf data122998/Features_competition_train.tar.gz
!cp data122998/label_cls14_train.json /home/aistudio/work/data/.
#删除原始数据集，减少内存
!rm -rf data122998/
#查看其训练数据，是否为729条数据
%cd /home/aistudio/data/Features_competition_train
!ls | wc -l
```
    

### 2.生成paddlevideo相关的依赖包


```python
%cd /home/aistudio/PaddleVideo/                                  # 进入PaddleVideo文件夹
!pip install -r requirements.txt                                 # 安装环境配置
    

### 3.训练数据预处理
解压后使用get_instance_for_bmn进行数据分割，以每4s进行分割（例如1-4，4-8...依次进行），将这些固定长度的动作片段中标注成.npy文件。这是按照真是动作的开始时间结束时间进行分割，保证分割片段中都是完整的动作。
#### 生成训练数据和标签
运行脚本get_instance_for_bmn.py，提取二分类的proposal，windows=4，根据gts和特征得到BMN训练所需要的数据集

```python
# 数据预处理
%cd /home/aistudio/PaddleVideo/applications/TableTennis/
# 生成bmn训练数据和标签
!python3.7 get_instance_for_bmn.py
%rm -rf /home/aistudio/data/Features_competition_train
``` 

#### 运行标签修正脚本
训练BMN前矫正标签和数据是否一一对应，数据中一些无对应标签的feature将不参与训练


```python
!python3.7 fix_bad_label.py
```

*执行后/home/aistudio/work/data/Inputforbmn/feature目录下生成了训练用的numpy数据，该目录下还生成了labelfixed.json标签文件*
下次运行该项目时，可以无须再解压分割训练集

## 训练模型
在不同的train step范围设定递减的学习率，使模型训练更加收敛。
resume_epoch可以设置为指定的checkpoint并继续训练，训练后权重会保存在output文件夹中。


```python
%cd /home/aistudio/PaddleVideo/
!python main.py -c configs/localization/bmn.yaml

## 模型导出
将训练好的模型导出用于推理预测，执行以下脚本。
-p 选择想要导出的模型参数，可自行修改。

```python
%cd /home/aistudio/PaddleVideo/
!python tools/export_model.py -c configs/localization/bmn.yaml -p output/BMN/BMN_epoch_00015.pdparams -o inference/BMN
```

## 解压并分割测试集
由于单个动作长度在0~2秒，若直接采用每4秒一划分的方法，可能会出现一个动作被划分到两个文件的可能。
因此考虑其动作的完整性，在这种划分方法的基础上，将滑窗的移动步长改为2秒；这样在划分测试集时，每次都会产生2秒的重叠，则在这2秒的重叠中就包括了某个完整动作。这样分割方法的改进，使动作特征数据更加完整，匹配精度有所提高。


```python
#解压测试集
%cd /home/aistudio/data/
!tar xf data123009/Features_competition_test_B.tar.gz
!rm -rf data123009/
#划分测试集
%cd /home/aistudio/PaddleVideo/applications/TableTennis/
# split_testdata_overlap.py 文件中的路径可能需要修改
!python split_testdata_overlap.py

!rm -rf /home/aistudio/data/Features_competition_test_B/
```

```python
# 统计分割的npy文件个数
%cd /home/aistudio/work/data/test_npy
%ls *.npy | wc -l
```

*在/home/aistudio/work/data/test_npy/文件夹下生成测试特征文件。*

## 推理预测
使用导出的模型进行推理预测，执行以下命令。

```python
#生成测试结果
%cd /home/aistudio/PaddleVideo/
!python tools/predict.py --input_file /home/aistudio/work/data/test_npy \
 --config configs/localization/bmn.yaml \
 --model_file inference/BMN/BMN.pdmodel \
 --params_file inference/BMN/BMN.pdiparams \
 --use_gpu=True \
 --use_tensorrt=False
```

上面程序输出的json文件是分割后的预测结果。执行以下脚本，运用**非极大抑制算法**将这些文件组合到一起，形成submission.json文件。

```python
#合并生成结果
%cd /home/aistudio/PaddleVideo/applications/TableTennis/
!python merge_result.py
```

## 提升思路
*1.在官方提供的[baseline](https://aistudio.baidu.com/aistudio/projectdetail/3389378)基础上增加训练的epoch数量
*2.采用多步长衰减与Warmup的余弦退火衰减等多种学习率调整方法，其A榜测试提交得分对比如下：
学习率调整策略	15epoch
decay 策略（learningrate：[0.001->0.0001->0.00001]）	44.83078
warm up策略（max_learningrate=[0.001]）	44.76696
warm up策略（max_learningrate=[0.002]）	44.52029
average checkpoint策略（最近5个epoch）	44.22303
可见其多种学习率调整策略对该数据结果相差不大。
*3.基于每4秒一分的划分方法上，将滑动窗口的步长改为2秒，保证测试集动作的完整性，优化特征数据。
*4.采用非极大抑制算法合并测试结果

