# RNN DNA Core Promoter Classification

该项目实现了一个简单的RNN模型，用于DNA核心启动子分类。模型使用Python和NumPy实现，并通过反向传播进行训练。

## 文件结构

- `promoter.py`：主代码文件，包含数据加载、特征提取和训练代码。
-  `rnn.py`:
-  `dna_core_promoter.xlsx`:数据文件
## 依赖项
在运行代码之前，请确保已安装以下Python库：
- numpy
- pandas
- scikit-learn
可以使用以下命令安装这些库：
```sh
pip install numpy pandas scikit-learn
```
## 使用方法
1.将数据集文件dna_core_promoter.xlsx放置在与代码相同的目录中。
2.运行promoter.py文件。
```sh
python promoter.py
```
## 代码说明
代码首先加载数据集，并将序列数据转换为k-mer特征。利用rnn.py中的rnn模型训练并评估性能。

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
