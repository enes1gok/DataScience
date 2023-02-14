'''from sklearn.feature_selection import VarianceThreshold
X = [[0,0,1],[0,1,0],[1,0,0],[0,1,1],[0,1,0],[0,1,1]]

sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8))) #var(x) = p(1-p)
sel.fit_transform(X)
#We are waiting for the first column to be eliminated.
#The probability of having a zero value is 5/6 > 0.8'''

'''#Statistical model selection
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X, y = load_iris(return_X_y=True)
print(X.shape)
X_new = SelectKBest(chi2,k=2).fit_transform(X, y)
print(X_new.shape)'''

'''#model-based feature selection
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
X, y = load_iris(return_X_y=True)
print(X.shape)'''

'''#L1 norm based feature selection
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print(X_new.shape)

#L2 norm based feature selection
lsvc = LinearSVC(C=0.01, penalty="l2", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print(X_new.shape)'''

'''#Tree-based feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
X, y = load_iris(return_X_y=True)

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
clf.feature_importances_

model = SelectFromModel(clf, prefit = True)
X_new = model.transform(X)
print(X_new.shape)'''