import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import os
import multiprocessing as mp

from sklearn.preprocessing import LabelEncoder

class DataPreprocessing(object):

    '''
    This class is going to form the abstract class for the preprocessing
    step for different models including 
    '''
    def __init__(self):
        self.resources = pd.read_csv('data/resources.csv')
        self.resources['total_requested'] = self.resources['quantity'] * self.resources['price']

    def process(self, df, test_options):
        print('processing data.........')
        file_name = 'data/test_processed.csv' if test_options else 'data/train_processed.csv'
        if os.path.isfile(file_name):
            return pd.read_csv(file_name)
        df = self.groupby_operation(self.resources, df, 'id', 'total_requested', ['sum', 'mean', 'max', 'min', 'count'])
        df = self.groupby_operation(df, df, 'teacher_id', 'teacher_number_of_previously_posted_projects', ['sum', 'mean', 'max', 'min', 'count'])
        df = self.process_datetime(df, 'project_submitted_datetime')
        df = self.handle_missing_data(df, 'teacher_prefix')
        for column in ['teacher_id', 'school_state', 'project_grade_category',
                    'project_subject_categories', 'project_subject_subcategories', 'teacher_prefix']:
            df = self.label_encode(df, column, test_options)
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(self.process_df, np.array_split(df, mp.cpu_count()))
            pool.close()
            pool.join()
        df = pd.concat(results)
        self.save_processed_data(df, file_name)
        return df

    def save_processed_data(self, df, file_name):
        df.to_csv(file_name, index=False)

    def groupby_operation(self, groupby_df, df, groupby_column, column_name, aggs):
        '''
        Groupby aggs in and merge with a df
        :param column_name (string)
        :param aggs: (list)
        :return: df
        '''
        for agg in aggs:
            tmp = groupby_df.groupby(groupby_column, as_index=False)[column_name].agg(agg).rename(
                columns={column_name: '%s_%s'%(str(agg),column_name)})
            df = pd.merge(df, tmp, on=groupby_column, how='left')
        return df

    def process_datetime(self, df, column):
        '''
        Groupby aggs in and merge with a df
        :param column (string)
        :param df: dataframe
        :return: df
        '''
        assert df[column].dtype == 'object'
        df[column] = pd.to_datetime(df[column], format='%Y-%m-%d %H:%M:%S')
        df['day'] = df[column].dt.day
        df['month'] = df[column].dt.month
        df['year'] = df[column].dt.year
        df['hour'] = df[column].dt.hour
        df['minute'] = df[column].dt.minute
        df['second'] = df[column].dt.second
        return df

    def handle_missing_data(self, df, column):
        if df[column].isnull().sum() == 0:
            return df
        if column == 'teacher':
            mode_value = 'Mrs.'
        else:
            mode_value = df[column].value_counts().keys()[0]
        df[column].fillna(mode_value, inplace=True)
        return df

    def label_encode(self, df, column, test_options):
        if test_options:
            le = LabelEncoder()
            le.classes_ = np.load('label_encoder/'+column+'.npy')
            df[column] = le.transform(list(df[column].values))
        else:
            le = LabelEncoder()
            le.fit(list(df[column].values))
            np.save('label_encoder/'+column+'.npy', le.classes_)
            df[column] = le.transform(list(df[column].values))
        return df

    def process_df(self, df):
        remaining_columns = [col for col in df.columns if df[col].dtype == 'object' and col != 'id']
        for col in remaining_columns:
            print('start processing column {}'.format(col))
            df[col].fillna('', inplace=True)
            df[col] = df[col].apply(lambda x: self.clean_sentences(x))
            print('done processing column {}'.format(col))
            print('\n')
        return df

    def clean_sentences(self, string):
        """
        Tokenization/string cleaning for dataset
        Every dataset is lower cased except
        """
        string = BeautifulSoup(string)
        string = string.get_text().encode('ascii', 'ignore')
        string = string.decode('utf-8')
        string = re.sub(r"\r\n\r\n", " ", string)
        string = re.sub(r"\r\n", " ", string)
        string = re.sub(r"\\r\\n", " ", string)
        string = re.sub(r"\\r\\n\\r\\n", " ", string)
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)
        string = string.strip('rn')
        string = string.strip().lower()
        return string