import pandas as pd 
import numpy as np
from distfit import distfit
import scipy.stats as st
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import QuantileTransformer

class DataProvider:
    
    def __init__(self, path, is_class_to_dummies = False, is_save_desc = False,
                is_check_distr = False):
        data = pd.read_parquet(path,engine='pyarrow')
        self.features = list(filter(lambda x: x != "target", data.columns))
        self.preprocess_data(data,is_class_to_dummies,is_save_desc)
        if is_check_distr:
            self.check_distributions()
        else:
            self.distributions = pd.read_csv('./data/distributions.csv')

    def preprocess_data(self, data, is_class_to_dummies, is_save_desc=True ):
        data = data.dropna()

        self.complexity_distr = data.groupby('complexity')['target'].mean()

        data = data[data['complexity'] != 4]
        data = data[data['complexity'] != 12]

        if is_save_desc:
            desc = data.describe()
            desc.to_csv('./data/desc.csv')
        
        if is_class_to_dummies:
            one_hot = pd.get_dummies(data['class'])
            data = data.join(one_hot)
            
        data = data.drop('class',axis = 1)
        self.features.remove('class')
        
        self.X = data[self.features]
        self.y = data.target 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                             test_size=0.2, random_state=1)
    
    def check_distributions(self):
        dist_names = ["norm", "exponweib", "weibull_max",
                      "weibull_min", "pareto", "genextreme"]
        distributions = []
        for feat in self.features:
            dist_results = []
            params = {}
            for dist_name in dist_names:
                dist = getattr(st, dist_name)
                param = dist.fit(self.X[feat])
                params[dist_name] = param
                _, p = st.kstest(self.X[feat], dist_name, args=param)
                dist_results.append((dist_name, p))

            best_dist, _ = (max(dist_results, key=lambda item: item[1]))
            
            distributions.append({"feature": feat, "distr": best_dist})
        self.distributions = pd.DataFrame(distributions)
        self.distributions.to_csv('./data/distributions.csv')
        
    def get_random_non_norm_features(self,n):
        return self.distributions[self.distributions['distr']!='norm'].sample(n=n)[['feature','distr']].values
    
    def get_target_class_count(self, cls):
        return len(self.y[self.y == cls])
    
    def get_quantile_transformed_X(self):
        return (QuantileTransformer(output_distribution='normal').fit_transform(self.X_train.values),
                 QuantileTransformer(output_distribution='normal').fit_transform(self.X_test.values))
    