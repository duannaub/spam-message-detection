from data_process import data_process
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
import jieba

# 调用数据处理函数
adata, data_after_stop, labels = data_process()

# 分割训练样本和测试样本，训练样本占80%，测试样本占20%
data_tr, data_te, labels_tr, labels_te = train_test_split(adata, labels, test_size=0.2, random_state=26)

# 创建CountVectorizer对象
countVectorizer = CountVectorizer()

# 对训练集的文本数据进行拟合和转换，生成词频矩阵
data_tr = countVectorizer.fit_transform(data_tr)

# 使用已有的词汇表对测试集进行转换
data_te = countVectorizer.transform(data_te)

# 计算训练集、测试集的TF-IDF特征向量
tfidf_transformer = TfidfTransformer()
X_tr = tfidf_transformer.fit_transform(data_tr).toarray()
X_te = tfidf_transformer.transform(data_te).toarray()

# 创建高斯朴素贝叶斯分类器
model = GaussianNB()

# 使用训练集数据进行模型训练
model.fit(X_tr, labels_tr)

# 使用训练好的模型对测试集进行预测
predictions = model.predict(X_te)

# 计算预测准确率
accuracy = accuracy_score(labels_te, predictions)
print(f"预测准确率: {accuracy * 100:.2f}%")

# 计算混淆矩阵
cm = confusion_matrix(labels_te, predictions)
print("混淆矩阵:")
print(cm)

# 假设是二分类任务，提取TP, FP, TN, FN
# 混淆矩阵的排列方式：
# [[TN, FP],
#  [FN, TP]]

tn, fp, fn, tp = cm.ravel()
print(f"真阴性 (TN): {tn}")
print(f"假阳性 (FP): {fp}")
print(f"假阴性 (FN): {fn}")
print(f"真阳性 (TP): {tp}")

# 计算精确率、召回率和F1分数
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"精确率 (Precision): {precision:.2f}")
print(f"召回率 (Recall): {recall:.2f}")
print(f"F1 分数: {f1:.2f}")

# 用户输入短信的分类
def classify_message(message):
    # 分词
    words = " ".join(jieba.lcut(message))
    # 转换为特征向量
    message_vec = countVectorizer.transform([words])
    message_tfidf = tfidf_transformer.transform(message_vec).toarray()
    # 预测
    prediction = model.predict(message_tfidf)
    return "垃圾短信" if prediction[0] == 1 else "正常短信"

# 支持用户多次输入
while True:
    user_message = input("请输入一条短信（输入 'q' 退出）：")
    if user_message.lower() == 'q':
        print("程序已退出。")
        break
    result = classify_message(user_message)
    print(f"分类结果: {result}")