import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate,GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,StackingClassifier

# 读取数据
all_data = pd.read_csv('data.csv')

# height 数值类型 为object 需要转化为 数值型
all_data = all_data.astype({'Height':'float64'})

# 缺失值
replace_list = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Height']
all_data.loc[:,replace_list] = all_data.loc[:,replace_list].replace({0:np.nan})

# 删除相关性低的Height
#all_data.drop('Height',1,inplace = True)

# remove the outliers
# 异常点 上须的计算公式为Q3+1.5(Q3-Q1)；下须的计算公式为Q1-1.5(Q3-Q1)
# 极端异常点 ：上限的计算公式为Q3+3(Q3-Q1) 下界的计算公式为Q1-3(Q3-Q1)
# 由于数据量比较少 所以选择删除极端异常值
def remove_outliers(feature,all_data):
    first_quartile = all_data[feature].describe()['25%']
    third_quartile = all_data[feature].describe()['75%']
    iqr = third_quartile - first_quartile
    # 异常值下标
    index = all_data[(all_data[feature] < (first_quartile - 3*iqr)) | (all_data[feature] > (first_quartile + 3*iqr))].index
    all_data = all_data.drop(index)
    return all_data
outlier_features = ['Insulin', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction']
for feat in outlier_features:
    all_data = remove_outliers(feat,all_data)
	
# 直接删除处理 中值填充 分区间种植填充 KNNImputer 填充 相关特征预测填充
def drop_method():
    all_data.dropna(inplace = True)
	
def median_method():
    for column in list(all_data.columns[all_data.isnull().sum() > 0]):
        median = all_data[column].median()
        all_data[column].fillna(median, inplace=True)

def knn_method():
    # 先将缺失值比较少的特征用中值填充
    values = {'Glucose': all_data['Glucose'].median(),
              'BloodPressure':all_data['BloodPressure'].median(),
              'BMI':all_data['BMI'].median(),
              'Height':all_data['Height'].median()}
    all_data.fillna(value=values,inplace=True)

    # 用KNNImputer 填充 Insulin SkinThickness
    corr_SkinThickness = ['BMI', 'Glucose','BloodPressure', 'SkinThickness']
    # 权重按距离的倒数表示。在这种情况下，查询点的近邻比远处的近邻具有更大的影响力
    SkinThickness_imputer = KNNImputer(n_neighbors = 16,weights = 'distance')
    all_data[corr_SkinThickness] = SkinThickness_imputer.fit_transform(all_data[corr_SkinThickness])

    corr_Insulin = ['Glucose', 'BMI','BloodPressure', 'Insulin']
    Insulin_imputer = KNNImputer(n_neighbors = 16,weights = 'distance')
    all_data[corr_Insulin] = Insulin_imputer.fit_transform(all_data[corr_Insulin])
	
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer  # 用来填补缺失值
def predict_method(feature):
    # 复制一份数据 避免对原数据做出不必要的修改
    copy_data = all_data.copy()
    # 缺失了的下标
    predict_index = copy_data[copy_data[feature].isnull()].index
    # 没缺失的下标
    train_index = copy_data[feature].dropna().index
    
    # 用作预测 的训练集标签
    train_label = copy_data.loc[train_index,feature]
    
    copy_data = copy_data.drop(feature,axis=1)
    
    # 对特征先用中值填充
    imp_median = SimpleImputer(strategy='median')
    
    # 用作预测的训练集特征
    train_feature = copy_data.loc[train_index]
    train_feature = imp_median.fit_transform(train_feature)

    # 需要进行预测填充处理的缺失值
    pre_feature = copy_data.loc[predict_index]
    pre_feature = imp_median.fit_transform(pre_feature)
    
    # 选取随机森林模型
    fill_model = RandomForestRegressor()
    fill_model = fill_model.fit(train_feature,train_label)
    # 预测 填充
    pre_value = fill_model.predict(pre_feature)
    all_data.loc[predict_index,feature] = pre_value

# drop_method()
median_method()
# knn_method()

"""
#用随机森林的方法填充缺失值较多的 SkinThickness 和 Insulin 缺失值
predict_method("Insulin")
predict_method("SkinThickness")
# 其余值中值填充
for column in list(all_data.columns[all_data.isnull().sum() > 0]):
    median = all_data[column].median()
    all_data[column].fillna(median, inplace=True)
"""

# 特征
feture_data = all_data.drop('Outcome',1)
# 标签
label = all_data['Outcome']

# 利用BMI和身高构造weight特征
# BMI = weight(kg) / height(m)**2
feture_data['weight'] = (0.01*feture_data['Height'])**2 * feture_data['BMI']

# 标准化
Std = StandardScaler()
feture_data = Std.fit_transform(feture_data)

def train(model, params):
    grid_search = GridSearchCV(estimator = model, param_grid = params, cv = 5)
    grid_search.fit(feture_data,label)
    print(grid_search.best_params_)
    model_score = cross_validate(grid_search.best_estimator_,feture_data, label, cv=5)
    print(model_score['test_score'])
    print("mean test score :{}".format(model_score['test_score'].mean()))
    return grid_search

model = SVC()
params  =  {'C':np.linspace(0.1, 2, 100)}
SVC_grid_search = train(model,params)
plt.figure()
sns.lineplot(x=[x for x in range(100)],y=SVC_grid_search.cv_results_['mean_test_score'])
plt.show()

params = {"C":np.linspace(0.1,2,100)}
model = LogisticRegression()
LR_grid_search = train(model,params)
plt.figure()
sns.lineplot(x=[x for x in range(100)],y=LR_grid_search.cv_results_['mean_test_score'])
plt.show()

params = {"n_estimators":[x for x in range(30,50,2)],'min_samples_split':[x for x in range(4,10)]}
model = RandomForestClassifier()
RFC_grid_search = train(model,params)
plt.figure()
sns.lineplot(x=[x for x in range(len(RFC_grid_search.cv_results_['mean_test_score']))],
             y=RFC_grid_search.cv_results_['mean_test_score'])
plt.show()

estimators = [
    ('SVC',SVC_grid_search.best_estimator_),
    ('NB', LR_grid_search.best_estimator_),
    ('RFC', RFC_grid_search.best_estimator_)
]
model = StackingClassifier(estimators=estimators, final_estimator=SVC())
model_score = cross_validate(model,feture_data, label, cv=5)
print(model_score['test_score'])
print("mean test score :{}".format(model_score['test_score'].mean()))
