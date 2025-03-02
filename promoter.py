import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from rnn import RNN  

# 加载数据
df = pd.read_excel("dna_core_promoter.xlsx")
sequences = df['sequence'].values
labels = df['label'].values

# 检查类别分布
print("类别分布:\n", pd.Series(labels).value_counts())

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels, test_size=0.2, stratify=labels, random_state=42
)

# 特征提取
def canonical_kmer(kmer):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    rc = ''.join([complement[nt] for nt in reversed(kmer)])
    return min(kmer, rc)

def kmer_analyzer(seq, k=3):
    return [canonical_kmer(seq[i:i+k]) for i in range(len(seq) - k + 1)]

vectorizer = CountVectorizer(analyzer=lambda x: kmer_analyzer(x, k=3))
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# 将稀疏矩阵转换为数组并调整维度
X_train_padded = X_train_counts.toarray().reshape(-1, 1, X_train_counts.shape[1])  # (n_samples, 1, n_features)
X_test_padded = X_test_counts.toarray().reshape(-1, 1, X_test_counts.shape[1])

# 训练和评估
input_dim = X_train_padded.shape[2] 
hidden_dim = 100
output_dim = 1
learning_rate = 0.01
epochs = 20
batch_size = 32

rnn = RNN(input_dim, hidden_dim, output_dim, learning_rate)
rnn.train(X_train_padded, y_train, epochs, batch_size)

y_pred = []
for x in X_test_padded:
    y = rnn.forward(x)
    y_pred.append(1 if y > 0.5 else 0)

print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("分类报告:\n", classification_report(y_test, y_pred))