from pandas import DataFrame, read_csv, concat
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


df=pd.read_csv('/Users/baoh/PycharmProjects/Thesis2018/toefl11_tokenized_test.tsv',delimiter='\t')
#print (df)
#df2=df.loc[:,['Text']]
#text = df2.values[0]
#text = (df2.values[0],df2.values[1]) ?

#df.Text[0:12099]

#data['w']  #选择表格中的'w'列，使用类字典属性,返回的是Series类型

#data.w    #选择表格中的'w'列，使用点属性,返回的是Series类型

#data[['w']]  #选择表格中的'w'列，返回的是DataFrame类型




from sklearn.feature_extraction.text import CountVectorizer
# 文本文档列表
#text = ["The quick brown fox jumped over the lazy dog."]
text=df.Text[0:15]
# 构造变换函数
vectorizer = CountVectorizer()
# 词条化以及建立词汇表
vectorizer.fit(text)
# 总结
print(vectorizer.vocabulary_)
# 编码文档
vector = vectorizer.transform(text)
# 总结编码文档
print(vector.shape)
print(type(vector))
print(vector.toarray())




from sklearn.feature_extraction.text import TfidfVectorizer
# 文本文档列表
#text = ["The quick brown fox jumped over the lazy dog.",
#"The dog.",
#"The fox"]
# 创建变换函数
text=df.Text[0:15]
vectorizer = TfidfVectorizer()
# 词条化以及创建词汇表
vectorizer.fit(text)
# 总结
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
print(vectorizer.tf_)
# 编码文档
#vector = vectorizer.transform([text[0]])
vector = vectorizer.transform(text)
# 总结编码文档
print(vector.shape)
print(vector.toarray())



from sklearn.feature_extraction.text import HashingVectorizer
# 文本文档列表
#text = ["The quick brown fox jumped over the lazy dog."]
#df.Text[0:12099]
text=df.Text[0:15]
# 创建变换函数
vectorizer = HashingVectorizer(n_features=20)
# 编码文档
vector = vectorizer.transform(text)
# 总结编码文档
print(vector.shape)
print(vector.toarray())

