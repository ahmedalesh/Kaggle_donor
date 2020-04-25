from data_preprocessing.data_preprocessing import DataPreprocessing
from textblob import TextBlob
import multiprocessing as mp
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class LGBMDataPreprocessing(DataPreprocessing):

    def __init__(self):
        super(LGBMDataPreprocessing, self).__init__()

    def process(self, df, test_options):
        file_name = 'data/lgbm_test_processed.csv' if test_options else 'data/lgbm_train_processed.csv'
        if os.path.isfile(file_name):
            return pd.read_csv(file_name)
        df['project_essay'] = df.apply(lambda row: ' '.join([
            str(row['teacher_prefix']),
            str(row['school_state']),
            str(row['project_grade_category']),
            str(row['project_subject_categories']),
            str(row['project_subject_subcategories']),
            str(row['project_essay_1']),
            str(row['project_essay_2']),
            str(row['project_essay_3']),
            str(row['project_essay_4']),
        ]), axis=1)
        with mp.Pool(mp.cpu_count()) as pool:
            result = pool.map(self.extract_features, np.array_split(df, mp.cpu_count()))
            pool.close()
            pool.join()
        df = pd.concat(result)
        df = self.groupby_operation(self.resources, df, 'id', 'total_requested', ['sum', 'mean', 'max', 'min', 'count'])
        df = self.groupby_operation(df, df, 'teacher_id', 'teacher_number_of_previously_posted_projects', ['sum', 'mean', 'max', 'min', 'count'])
        df = self.groupby_operation(self.resources, df, 'id', 'price', ['sum', 'mean', 'max', 'min', 'count','std'])
        for column in ['teacher_id', 'school_state', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories', 'teacher_prefix']:
            df = self.label_encode(df, column, test_options)
        df = self.process_datetime(df, 'project_submitted_datetime')
        with mp.Pool(mp.cpu_count()) as pool:
            result = pool.map(self.process_df, np.array_split(df, mp.cpu_count()))
            pool.close()
            pool.join()
        cols = [
            'project_title',
            'project_essay',
            'project_resource_summary'
        ]
        n_features = [
            400,
            5000,
            400
        ]
        for column, feats in zip(cols,n_features):
                df = self.tfidf_vectorizer(df, column, feats, test_options)
        self.save_processed_data(df, file_name)
        return df

    def process_df(self, df):
        for col in ['project_title', 'project_essay', 'project_resource_summary']:
            print('start processing column {}'.format(col))
            df[col].fillna('', inplace=True)
            df[col] = df[col].apply(lambda x: self.clean_sentences(x))
            print('done processing column {}'.format(col))
            print('\n')
        return df

    def get_polarity(self, text):
        textblob = TextBlob(text.encode('utf-8'))
        pol = textblob.sentiment.polarity
        return round(pol, 3)

    def get_subjectivity(self, text):
        textblob = TextBlob(text.encode('utf-8'))
        subj = textblob.sentiment.subjectivity
        return round(subj, 3)

    def extract_features(self, df):
        df['project_title_len'] = df['project_title'].apply(lambda x: len(str(x)))
        df['project_essay_1_len'] = df['project_essay_1'].apply(lambda x: len(str(x)))
        df['project_essay_2_len'] = df['project_essay_2'].apply(lambda x: len(str(x)))
        df['project_essay_3_len'] = df['project_essay_3'].apply(lambda x: len(str(x)))
        df['project_essay_4_len'] = df['project_essay_4'].apply(lambda x: len(str(x)))
        df['project_resource_summary_len'] = df['project_resource_summary'].apply(lambda x: len(str(x)))
        df['polarity'] = df['project_essay'].apply(lambda x: self.get_polarity(x))
        df['subjectivity'] = df['project_essay'].apply(lambda x: self.get_subjectivity(x))
        return df

    def tfidf_vectorizer(self, df, column_name, n_features, test_options):
        if test_options:
            with open('tfidfvectorizer/'+column_name+str(n_features)+'.pickle', 'wb') as f:
                tfidf = pickle.load(f)
            tfidf_test = np.array(tfidf.transform(df[column_name]).todense(), dtype=np.float16)
            for i in range(n_features):
                df[column_name + '_tfidf_' + str(i)] = tfidf_test[:, i]
        else:
            tfidf = TfidfVectorizer(max_features=n_features, min_df=3)
            tfidf.fit(df[column_name])
            tfidf_train = np.array(tfidf.transform(df[column_name]).todense(), dtype=np.float16)
            for i in range(n_features):
                df[column_name + '_tfidf_' + str(i)] = tfidf_train[:, i]
            if not os.path.isdir('tfidfvectorizer/'):
                os.mkdir('tfidfvectorizer')
            with open('tfidfvectorizer/'+column_name+str(n_features)+'.pickle', 'wb') as f:
                pickle.dump(tfidf, f)
        return df