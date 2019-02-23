import keras
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Dot,Conv2D,Flatten,MaxPool2D
from keras.optimizers import Adam
from layers.attention import Position_Embedding, Attention


class SELF_ATT():
    def __init__(self,config):
        self.config = config
        self.model = self.build()
    
    def build(self):
        seq1 = Input(name='seq1', shape=[self.config['seq1_maxlen']])
        seq2 = Input(name='seq2', shape=[self.config['seq2_maxlen']])
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.config['embed_trainable'])

        seq1_embed = embedding(seq1)
        seq1_embed = Dropout(0.5)(seq1_embed)
        seq2_embed = embedding(seq2)
        seq2_embed = Dropout(0.5)(seq2_embed)

        share_lstm = Bidirectional(LSTM(self.config['hidden_size'], return_sequences=True, dropout=self.config['dropout_rate']))
        seq1_rep_rnn = share_lstm(seq1_embed)
        seq2_rep_rnn = share_lstm(seq2_embed)

        pos_emb = Position_Embedding()
        att = Attention(8, 16)

        seq1_embed = pos_emb(seq1_rep_rnn)
        seq2_embed = pos_emb(seq2_rep_rnn)

        seq1_seq = att([seq1_embed,seq1_embed,seq1_embed])
        seq1_seq = GlobalAveragePooling1D()(seq1_seq)
        seq1_seq = Dropout(0.5)(seq1_seq)

        seq2_seq = att([seq2_embed,seq2_embed,seq2_embed])
        seq2_seq = GlobalAveragePooling1D()(seq2_seq)
        seq2_seq = Dropout(0.5)(seq2_seq)

        sum_vec = add([seq1_seq,seq2_seq])
        mul_vec = multiply([seq1_seq,seq2_seq])

        mlp_input = concatenate([sum_vec, mul_vec])

        output = Dense(1, activation="sigmoid")(mlp_input)
        model = Model(inputs=[seq1, seq2], outputs=output)
        return model