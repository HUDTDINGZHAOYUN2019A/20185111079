import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train = pd.read_csv(r'D:\微信文件\WeChat Files\hu948086277\FileStorage\File\2019-05\pfm_train.csv')
test = pd.read_csv(r'D:\微信文件\WeChat Files\hu948086277\FileStorage\File\2019-05\pfm_test.csv')
print('train size:{}'.format(train.shape))  # train size:(1100, 31)
print('test size:{}'.format(test.shape))  # test size:(350, 30)

# EmployeeNumber为员工ID，将其删除
train.drop(['EmployeeNumber'], axis=1, inplace=True)

# 将Attrition（该字段为标签）移至最后一列，方便索引
Attrition = train["Attrition"]

train.drop(['Attrition'], axis=1, inplace=True)
train.insert(0, 'Attrition', Attrition)

# 在分析中发现有一些字段的值是单一的,进一步验证
single_value_feature = []
for col in train.columns:
    lenght = len(train[col].unique())
    if lenght == 1:
        single_value_feature.append(col)

# 'Over18', 'StandardHours'这两个字段的值是唯一的，删除这两个字段 删除这两个字段
train.drop(['Over18', 'StandardHours'], axis=1, inplace=True)
# train.shape  # (1100, 28)

# 使用pandas的cut进行分组，分为10组
train['MonthlyIncome'] = pd.cut(train['MonthlyIncome'], bins=10)

# 将数据类型为‘object’的字段名提取出来，并使用one-hot-encode对其进行编码
col_object = []
for col in train.columns[1:]:
    if train[col].dtype == 'object':
        col_object.append(col)

train_encode = pd.get_dummies(train)

X = train_encode.iloc[:, 1:]
y = train_encode.iloc[:, 0]

# 划分训练集以及测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_train, y_train)

pred = lr.predict(X_test)
np.mean(pred == y_test)

# predict
# test数据集处理
test.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
test_MonthlyIncome = pd.concat((pd.Series([1009, 19999]), test['MonthlyIncome']))
# 在指定位置插入与train中MonthlyIncome的max、min一致的数值，之后再删除
test['MonthlyIncome'] = pd.cut(test_MonthlyIncome, bins=10)[2:]  # 分组并去除对应的值
test_encode = pd.get_dummies(test)
# test_encode.drop(['TotalWorkingYears', 'YearsWithCurrManager'], axis=1, inplace=True)  # 输出结果
# sample = pd.DataFrame(lr.predict(test_encode))
predRes = lr.predict(test_encode)
print("预测准确率为：", 1 - np.mean(predRes))

# sample.to_csv('sample.csv')
