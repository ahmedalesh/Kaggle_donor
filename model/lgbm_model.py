import numpy as np
import os

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from data_preprocessing.lgbm_data_preprocessing import LGBMDataPreprocessing

class LGBMModel(object):
    DP = LGBMDataPreprocessing

    def __init__(self):
        self.n_splits = 5
        self.n_repeats = 1
        self.params = {
            'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'max_depth': 16, 'num_leaves': 31,
            'learning_rate': 0.025, 'feature_fraction': 0.85, 'bagging_fraction': 0.85, 'bagging_freq': 5,
            'verbose': 0, 'num_threads': 1, 'lambda_l2': 1, 'min_gain_to_split': 0}

    def load_model(self, file_name):
        return lgb.Booster(model_file=file_name)

    def save_model(self, model, file_name):
        model.save_model(file_name)

    def train(self, df):
        cols_to_drop = [
            'id',
            'project_title',
            'project_essay',
            'project_resource_summary',
            'project_is_approved',
        ]
        X = df.drop(cols_to_drop, axis=1, errors='ignore')
        y = df['project_is_approved']
        feature_names = list(X.columns)
        kf = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats,random_state=0)
        for index,(train_index, valid_index) in enumerate(kf.split(X)):
            print('Fold {}/{}'.format(index + 1, self.n_splits))
            model = lgb.train(
                self.params,
                lgb.Dataset(X.loc[train_index], y.loc[train_index], feature_name=feature_names),
                num_boost_round=10000,
                valid_sets=[lgb.Dataset(X.loc[valid_index], y.loc[valid_index])],
                early_stopping_rounds=100,
                verbose_eval=100,
            )
            if index == 0:
                importance = model.feature_importance()
                model_fnames = model.feature_name()
                tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
                tuples = [x for x in tuples if x[1] > 0]
                with open('../lgbm_weights/feature_importance.txt', 'wb') as f:
                    for feat in tuples[:50]:
                        f.write(feat+'\n')

            p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)
            auc = roc_auc_score(y.loc[valid_index], p)
            print('{} AUC: {}'.format(index+1, auc))
            file_name = '../lgbm_weights/model_{}.txt'.format(index+1)
            self.save_model(model, file_name)

    def test(self, df):
        cols_to_drop = [
            'id',
            'project_title',
            'project_essay',
            'project_resource_summary'
        ]
        X = df.drop(cols_to_drop, axis=1, errors='ignore')
        y = np.zeros(X.shape[0])
        for index in enumerate(self.n_splits):
            file_name = '../lgbm_weights/model_{}.txt'.format(index+1)
            if not os.path.isfile(file_name):
                continue
            model = self.load_model(file_name)
            fold_prediction = model.predict(X)
            y+=(fold_prediction/float(self.n_splits))
        df['project_is_approved'] = y
