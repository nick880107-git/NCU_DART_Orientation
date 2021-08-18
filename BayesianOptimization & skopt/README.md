# Tuning SVM Hyperparameters With Bayesian Optimization
在機器學習中，除了模型建構外，超參數的調整也是很重要的，不過我想多數人一開始都是用「人工調整」慢慢找出最好的結果

其實超參數還有其他的調整方法，這次要實作的便是利用scikit-optimization中的BayesSearchCV進行調參


## 1. 安裝scikit-optimization (skopt)
```
conda install -c conda-forge scikit-optimize
```
or
```
pip install scikit-optimize
```
然後import
```
import skopt
```

## 2. 資料集
這次使用的資料集是 [Ionosphere Dataset](http://archive.ics.uci.edu/ml/datasets/Ionosphere)，是一個關於雷達偵測從電離層返回訊號好壞的二元分類資料集，內含351筆資料，34種參數

下載下來的資料集是.data檔案，直接將副檔名改為.csv即可用pandas讀取

## 3. 實作Bayesian Optimization
引入skopt中的BayesSearchCV
```
from skopt import BayesSearchCV
```
我們這次要調整SVM模型有以下幾個超參數
- C, the regularization parameter.
- kernel, the type of kernel used in the model.
- degree, used for the polynomial kernel.
- gamma, used in most other kernels.

宣告要調整的超參數名稱及範圍，資料型別須為dict, list of dict or list of tuple 

```
params = dict()
params['C'] = (1e-6, 100.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['degree'] = (1,5)
params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
```

評估方法使用cross-validation，避免只取單一驗證集導致驗證結果過於片面

宣告RepeatedStratifiedKFold作為cross-validation的generator
```
# 將資料集切成n_splits份後，輪流取其中1份作驗證，n-1份進行訓練，重複n_repeats次
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

宣告BayesSearchCV，傳入模型種類 & 欲調整的超參數，以及我們的generator，fit後就可以找到最佳超參數及分數了
```
search = BayesSearchCV(estimator=SVC(), search_spaces=params, n_jobs=-1, cv=cv)

search.fit(X, y)
print(search.best_score_)
print(search.best_params_)

>>>0.9515669515669516
>>>OrderedDict([('C', 9.595708891378774), ('degree', 5), ('gamma', 0.06458522648595245), ('kernel', 'rbf')])
```

如果宣告BayesSearchCV時出現以下錯誤：
```
TypeError: __init__() got an unexpected keyword argument 'iid'
```
那是因為scikit-learn在新版已經捨棄iid這個參數了，將scikit-learn降至0.23.2即可

## 4. 比較
未進行超參數調整前，accuracy為0.937，可見超參數的調整是有助於模型訓練的優化。

**完整程式碼可見skopt.ipynb**