import tensorflow as tf
import pickle
import numpy as np
import os

from sklearn.model_selection import train_test_split
from utils import read_glove_vecs
from data_preprocessing.gru_att_data_preprocessing import GRUATTDataPreprocessing

K = tf.keras.backend

class AttLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.init = tf.keras.initializers.get('normal')
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.add_weight(name='kernel',
                                  shape=(input_shape[-1],1),
                                  initializer='normal',
                                  trainable=True)
        super(AttLayer, self).build(input_shape)

    def call(self, x):
        eij = K.tanh(K.dot(x, self.W))
        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1, keepdims=True)
        weighted_input = K.sum(x * weights, axis=1)
        return weighted_input

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = {}
        base_config = super(AttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class GruAttModel(object):
    DP = GRUATTDataPreprocessing
    def __init__(self):
        self.glove_file  = 'glove/glove.6B.50d.txt'
        self.max_sent = 50
        self.max_sent_length = 100

    def load_model(self):
        with open('../weights/tokenizer_text.pickle', 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.model = tf.keras.models.load_model('../weights/model_gru_att.h5')

    def save_model(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(os.path.join(path,'tokenizer_text.pickle'), 'wb') as f:
            pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.model.save(os.path.join(path, 'model_gru_att.h5'))

    def construct_model(self):
        essay = tf.keras.layers.Input(shape=(self.max_sent_length,))
        embeddings_essay = self.embedding_layer(essay)
        l_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, return_sequences=True))(embeddings_essay)
        l_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(200))(l_lstm)
        l_att = AttLayer()(l_dense)
        sentEncoder = tf.keras.models.Model(essay, l_att)

        review_input = tf.keras.layers.Input(shape=(self.max_sent, self.max_sent_length), dtype='int32')
        review_encoder = tf.keras.layers.TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(50, return_sequences=True))(review_encoder)
        l_dense_sent = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(200))(l_lstm_sent)
        l_att_sent = AttLayer()(l_dense_sent)
        preds = tf.keras.layers.Dense(1, activation='sigmoid')(l_att_sent)
        self.model = tf.keras.models.Model(review_input, preds)
        self.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001),
                                metrics=['accuracy',tf.keras.metrics.AUC()])


    def prepare_data(self, df):
        data = np.zeros((len(df), self.max_sent, self.max_sent_length), dtype='int32')
        for i, sentences in enumerate(df['text']):
            for j, sent in enumerate(sentences):
                if j < self.max_sent:
                    array_1 = self.tokenizer.texts_to_sequences([sent])
                    array_1 = tf.keras.preprocessing.sequence.pad_sequences(array_1, maxlen=self.max_sent_length, padding='post')
                    data[i, j, :] = array_1
        return data

    def train(self, df):
        _, _, word_to_vec_map = read_glove_vecs(self.glove_file)
        self.tokenizer, self.embedding_layer = self.tokenize(df['text'], word_to_vec_map)
        dev_data = self.prepare_data(df)
        dev_label = df['project_is_approved'].values
        train_data, val_data, train_label, val_label = train_test_split(
            dev_data, dev_label, test_size=0.2, random_state=42)
        self.construct_model()
        print(train_data.shape, val_data.shape, train_label.shape, val_label.shape)
        print(self.model.summary())
        self.model.fit(
            train_data, train_label, validation_data=(val_data, val_label), epochs=50,
            batch_size=32, use_multiprocessing=True, workers=4, max_queue_size=3,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=2),
                tf.keras.callbacks.ModelCheckpoint('../weights/model_gru_att.h5', save_best_only=True, monitor='val_auc')
            ])

    def test(self, df):
        self.load_model()
        data = self.prepare_data(df)
        df['project_is_approved'] = self.model.predict(data)
        return df

    def tokenize(self, lang, word_to_vec_map):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer()
        sequences = []
        if type(lang[0]) is list:
            for docs in lang:
                sequences.extend(docs)
        else:
            sequences = lang
        lang_tokenizer.fit_on_texts(sequences)
        print('found words')
        print(lang_tokenizer.word_index)
        embedding_layer = self.pretrained_embedding_layer(word_to_vec_map, lang_tokenizer.word_index)
        return lang_tokenizer, embedding_layer

    def pretrained_embedding_layer(self, word_to_vec_map, word_to_index):
        vocab_len = len(word_to_index) + 1
        emb_dim = word_to_vec_map["cucumber"].shape[0]
        emb_matrix = np.zeros([vocab_len, emb_dim])
        for word, index in word_to_index.items():
            emb_matrix[index, :] = word_to_vec_map.get(word, np.random.random(emb_dim))
        embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=False, weights=[emb_matrix])
        return embedding_layer
