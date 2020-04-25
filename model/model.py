import tensorflow as tf
import pickle
import numpy as np
import os

from sklearn.model_selection import train_test_split
from utils import read_glove_vecs
from data_preprocessing.data_preprocessing import DataPreprocessing

class Model(object):
    DP = DataPreprocessing

    def __init__(self):
        self.glove_file  = 'glove/glove.6B.50d.txt'

    def load_model(self):
        with open('../weights/tokenizer_project_essay_1.pickle', 'rb') as f:
            self.tokenizer_1 = pickle.load(f)
        with open('../weights/tokenizer_project_essay_2.pickle', 'rb') as f:
            self.tokenizer_2 = pickle.load(f)
        self.model = tf.keras.models.load_model('../weights/weights/model_1.h5')

    def save_model(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(os.path.join(path,'tokenizer_project_essay_1.pickle'), 'wb') as f:
            pickle.dump(self.tokenizer_1, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path,'tokenizer_project_essay_2.pickle'), 'wb') as f:
            pickle.dump(self.tokenizer_1, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.model.save(os.path.join(path, 'model_weights.h5'))

    def construct_model(self, input_shape_1,input_shape_2, embedding_layer_1, embedding_layer_2):
        essay_1 = tf.keras.layers.Input(shape=input_shape_1)
        essay_2 = tf.keras.layers.Input(shape=input_shape_2)
        embeddings_essay_1 = embedding_layer_1(essay_1)
        embeddings_essay_2 = embedding_layer_2(essay_2)
        embeddings = tf.keras.layers.Concatenate(axis=1)([embeddings_essay_1,embeddings_essay_2])
        X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True))(embeddings)
        X = tf.keras.layers.Dropout(0.25)(X)
        X = tf.keras.layers.LSTM(units=128, return_sequences=False)(X)
        X = tf.keras.layers.Dropout(0.25)(X)
        X = tf.keras.layers.Dense(1)(X)
        X = tf.keras.layers.Activation(activation='sigmoid')(X)
        self.model = tf.keras.models.Model(inputs=[essay_1,essay_2], outputs=X)
        self.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001),
                                metrics=['accuracy', tf.keras.metrics.AUC()])

    def train(self, df):
        _, _, word_to_vec_map = read_glove_vecs(self.glove_file)
        dev_essay1, self.tokenizer_1, embedding_layer_essay_1 = self.tokenize(df['project_essay_1'], word_to_vec_map)
        dev_essay2, self.tokenizer_2, embedding_layer_essay_2 = self.tokenize(df['project_essay_2'], word_to_vec_map)
        dev_otherfeatures = df[['teacher_number_of_previously_posted_projects', 'teacher_prefix', 'project_subject_categories', 'project_is_approved']]
        train_essay1, val_essay1, train_essay2, val_essay2, train_otherfeatures, val_otherfeatures = train_test_split(
            dev_essay1, dev_essay2, dev_otherfeatures, test_size=0.2, random_state=42)
        train_target, val_target = train_otherfeatures['project_is_approved'].values, val_otherfeatures['project_is_approved'].values
        train_otherfeatures.drop('project_is_approved', axis=1, inplace=True)
        val_otherfeatures.drop('project_is_approved', axis=1, inplace=True)
        assert ('project_is_approved' not in train_otherfeatures.columns and 'project_is_approved' not in val_otherfeatures.columns)
        self.construct_model((train_essay1.shape[-1],), (train_essay2.shape[-1],),embedding_layer_essay_1, embedding_layer_essay_2)
        self.save_model('weights')
        self.model.fit(
            [train_essay1, train_essay2], train_target, validation_data=([val_essay1, val_essay2],val_target), epochs=50,
            batch_size=32, use_multiprocessing=True, workers=4, max_queue_size=3,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=2),
                tf.keras.callbacks.ModelCheckpoint('weights/model_weights.h5', save_best_only=True, monitor='val_auc')
            ])


    def test(self, df):
        self.load_model()
        essay_array_1 = self.tokenizer_1.texts_to_sequences(df['project_essay_1'])
        essay_array_1 = tf.keras.preprocessing.sequence.pad_sequences(essay_array_1, padding='post')
        essay_array_2 = self.tokenizer_1.texts_to_sequences(df['project_essay_2'])
        essay_array_2 = tf.keras.preprocessing.sequence.pad_sequences(essay_array_2, padding='post')
        df['project_is_approved'] = self.model.predict([essay_array_1,essay_array_2])
        return df

    def tokenize(self, lang, word_to_vec_map):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)
        array = lang_tokenizer.texts_to_sequences(lang)
        array = tf.keras.preprocessing.sequence.pad_sequences(array,padding='post')
        embedding_layer = self.pretrained_embedding_layer(word_to_vec_map, lang_tokenizer.word_index)
        return array, lang_tokenizer, embedding_layer

    def pretrained_embedding_layer(self, word_to_vec_map, word_to_index):
        vocab_len = len(word_to_index) + 1
        emb_dim = word_to_vec_map["cucumber"].shape[0]
        emb_matrix = np.zeros([vocab_len, emb_dim])
        for word, index in word_to_index.items():
            emb_matrix[index, :] = word_to_vec_map.get(word, np.random.random(emb_dim))
        embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=False)
        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])
        return embedding_layer
