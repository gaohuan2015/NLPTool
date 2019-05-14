import os
import codecs
from sklearn import feature_extraction
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
corpus = []
path = '/Kmeans/pubmed_abstract 35.txt'
path = os.getcwd() + path
with open(path, encoding='utf-8') as f:
    for line in f.readlines():
        corpus.append(line)

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
word = vectorizer.get_feature_names()
weight = tfidf.toarray()
print('Features length: ' + str(len(word)))
resName = "BHTfidf_Result.txt"
result = codecs.open(resName, 'w', 'utf-8')
for j in range(len(word)):
    result.write(word[j] + ' ')
result.write('\r\n\r\n')
for i in range(len(weight)):
    for j in range(len(word)):
        #print weight[i][j],
        result.write(str(weight[i][j]) + ' ')
    result.write('\r\n\r\n')
result.close()
clf = KMeans(n_clusters=5)  #景区 动物 人物 国家
s = clf.fit(weight)
print(s)
print(clf.cluster_centers_)
label = []
print(clf.labels_)
i = 1
while i <= len(clf.labels_):
    label.append(clf.labels_[i - 1])
    i = i + 1

pca = PCA(n_components=2)  #输出两维
newData = pca.fit_transform(weight)  #载入N维
print(newData)
x1 = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []
x4 = []
y4 = []
x5 = []
y5 = []
for i in range(len(newData)):
    if label[i] == 0:
        x1.append(newData[i][0])
        y1.append(newData[i][1])
    if label[i] == 1:
        x2.append(newData[i][0])
        y2.append(newData[i][1])
    if label[i] == 2:
        x3.append(newData[i][0])
        y3.append(newData[i][1])
    if label[i] == 3:
        x4.append(newData[i][0])
        y4.append(newData[i][1])
    if label[i] == 4:
        x5.append(newData[i][0])
        y5.append(newData[i][1])
#四种颜色 红 绿 蓝 黑
plt.plot(x1, y1, 'or')
plt.plot(x2, y2, 'og')
plt.plot(x3, y3, 'ob')
plt.plot(x4, y4, 'ok')
plt.plot(x5, y5, 'oy')
plt.show()
