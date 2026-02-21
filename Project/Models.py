from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import pandas as pd

class AbstractModel():
    all_question_answer = 42

    def __init__(self, data_provider):
        self.data_provider = data_provider

class DimensionReduction(AbstractModel):

    def get_TCNA(self):
        tsne = TSNE(n_components=2, random_state=self.all_question_answer)
        return pd.DataFrame(tsne.fit_transform(self.data_provider.X_train))
    
    def get_PCA(self):
        pca = PCA(n_components=2)
        return pd.DataFrame(pca.fit_transform(self.data_provider.X))
    
class PlainClassifier(AbstractModel):

    def compare_models(self):
        models = (
           { "Gaussian": "get_gaussian_clf"},
           { "DecisionTree": "get_decision_tree_clf"},
           { "RandomForest": "get_random_forest_clf"},
           { "AdaBoost": "get_adaboost_clf"},
           { "Bagging": "get_bagging_clf"},
           { "Naive": "get_naive_clf"}
        )
        for model in models:
            print(f"Accuracy of {list(model.keys())[0]}: {getattr(self, list(model.values())[0])()}")


    def get_gaussian_clf(self):
        X_train, X_test = self.data_provider.get_quantile_transformed_X()
        clf = GaussianNB()
        clf = clf.fit(X_train,self.data_provider.y_train)
        y_pred = clf.predict(X_test)
        return metrics.accuracy_score(self.data_provider.y_test, y_pred)
      
    def get_naive_clf(self):
        X_train, X_test = self.data_provider.get_quantile_transformed_X()
        clf = GaussianProcessClassifier(1.0 * RBF(1.0), random_state=self.all_question_answer)
        clf = clf.fit(X_train,self.data_provider.y_train)
        y_pred = clf.predict(X_test)
        return metrics.accuracy_score(self.data_provider.y_test, y_pred)
    
    def get_decision_tree_clf(self):
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        clf = clf.fit(self.data_provider.X_train,self.data_provider.y_train)
        y_pred = clf.predict(self.data_provider.X_test)
        self.decision_tree = clf
        return metrics.accuracy_score(self.data_provider.y_test, y_pred)
    
    def get_adaboost_clf(self):
        X_train, X_test = self.data_provider.get_quantile_transformed_X()
        clf = AdaBoostClassifier(algorithm="SAMME", random_state=self.all_question_answer)
        clf.fit(X_train,self.data_provider.y_train)
        y_pred = clf.predict(X_test)
        return metrics.accuracy_score(self.data_provider.y_test, y_pred)
    
    def get_bagging_clf(self):
        X_train, X_test = self.data_provider.get_quantile_transformed_X()
        clf = BaggingClassifier(estimator=SVC(shrinking=True), 
                                n_estimators=10, random_state=self.all_question_answer)
        clf.fit(X_train,self.data_provider.y_train)
        y_pred = clf.predict(X_test)
        return metrics.accuracy_score(self.data_provider.y_test, y_pred)

    def get_random_forest_clf(self):
        clf = RandomForestClassifier(max_depth=3, random_state=self.all_question_answer)
        clf.fit(self.data_provider.X_train,self.data_provider.y_train)
        y_pred = clf.predict(self.data_provider.X_test)
        return metrics.accuracy_score(self.data_provider.y_test, y_pred)