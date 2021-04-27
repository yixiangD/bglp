import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, LSTM, GRU, Lambda, dot, concatenate, Activation, Input, Conv1D, Multiply
from keras_pos_embd import TrigPosEmbedding

class LinearModel:
    def __init__(self, input_shape=(6,), nb_output_units=1):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units

    def __repr__(self):
        return 'Linear'

    def build(self):
        i = Input(shape=self.input_shape)
        x = Dense(self.nb_output_units, activation=None)(i)

        return Model(inputs=[i], outputs=[x])

class MLPModel:
    def __init__(self, input_shape=(6,), nb_output_units=1, nb_hidden_units=128, nb_layers=1, hidden_activation='relu'):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units
        self.nb_hidden_units = nb_hidden_units
        self.nb_layers = nb_layers
        self.hidden_activation = hidden_activation

    def __repr__(self):
        return 'MLP_{0}_units_{1}_layers'.format(self.nb_hidden_units, self.nb_layers)

    def build(self):
        # input
        i = Input(shape=self.input_shape)

        # add first LSTM layer
        x = Dense(self.nb_hidden_units, input_shape=self.input_shape, activation=self.hidden_activation)(i)

        if self.nb_layers > 1:
            for _ in range(self.nb_layers - 1):
                x = Dense(self.nb_hidden_units, input_shape=self.input_shape, activation=self.hidden_activation)(x)

        x = Dense(self.nb_output_units, activation=None)(x)

        return Model(inputs=[i], outputs=[x])

class GRUModel:
    def __init__(self, input_shape=(6, 1), nb_output_units=1, nb_hidden_units=128, nb_layers=1, dropout=0.0, recurrent_dropout=0.0):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units
        self.nb_hidden_units = nb_hidden_units
        self.nb_layers = nb_layers
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    def __repr__(self):
        return 'GRU_{0}_units_{1}_layers_dropout={2}_{3}'.format(self.nb_hidden_units, self.nb_layers, self.dropout, self.recurrent_dropout)

    def build(self):
        # input
        i = Input(shape=self.input_shape)

        # add first LSTM layer
        x = GRU(self.nb_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=self.nb_layers > 1)(i)

        if self.nb_layers > 1:
            for _ in range(self.nb_layers - 2):
                x = GRU(self.nb_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=True)(x)

            # add final GRU layer
            x = GRU(self.nb_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=False)(x)

        x = Dense(self.nb_output_units, activation=None)(x)

        return Model(inputs=[i], outputs=[x])

class LSTMModel:
    def __init__(self, input_shape=(6, 1), nb_output_units=1, nb_hidden_units=128, nb_layers=1, dropout=0.0, recurrent_dropout=0.0):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units
        self.nb_hidden_units = nb_hidden_units
        self.nb_layers = nb_layers
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    def __repr__(self):
        return 'LSTM_{0}_units_{1}_layers_dropout={2}_{3}'.format(self.nb_hidden_units, self.nb_layers, self.dropout, self.recurrent_dropout)

    def build(self):
        # input
        i = Input(shape=self.input_shape)

        # add first LSTM layer
        x = LSTM(self.nb_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=self.nb_layers > 1)(i)

        if self.nb_layers > 1:
            for _ in range(self.nb_layers - 2):
                x = LSTM(self.nb_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=True)(x)

            # add final LSTM layer
            x = LSTM(self.nb_hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=False)(x)

        x = Dense(self.nb_output_units, activation=None)(x)

        return Model(inputs=[i], outputs=[x])

'''
Attention mechanism code based on: https://github.com/philipperemy/keras-attention-mechanism
(Apache License 2.0)
'''
class LSTMAttentionModel:
    def __init__(self, input_shape=(6, 1), nb_output_units=1, nb_hidden_units=128, dropout=0.0, recurrent_dropout=0.0,
                 nb_attention_units=64):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units
        self.nb_hidden_units = nb_hidden_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.nb_attention_units = nb_attention_units

    def __repr__(self):
        return 'LSTMAttention_{0}_units_dropout={1}_{2}_{3}_attention_units'.format(self.nb_hidden_units, self.dropout, self.recurrent_dropout, self.nb_attention_units)

    def build(self):
        # input
        i = Input(shape=self.input_shape)

        # LSTM layer
        x = LSTM(self.nb_hidden_units, return_sequences=True)(i)

        # attention mechanism
        score_first_part = Dense(int(x.shape[2]), use_bias=False)(x)
        hidden_state = Lambda(lambda x: x[:, -1, :], output_shape=(self.nb_hidden_units,))(x)
        score = dot([score_first_part, hidden_state], [2, 1])
        attention_weights = Activation('softmax')(score)
        context_vector = dot([x, attention_weights], [1, 1])
        pre_activation = concatenate([context_vector, hidden_state])
        x = Dense(self.nb_attention_units, use_bias=False, activation='tanh')(pre_activation)

        # output
        x = Dense(1, activation=None)(x)

        return Model(inputs=[i], outputs=[x])

class Seq2seqModel:
    def __init__(self, input_shape=(6, 1), kernel_size=4, n_block=4, nb_hidden_units=64, nb_layers=2):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.n_block = n_block
        self.nb_hidden_units = nb_hidden_units
        self.nb_layers = nb_layers

    def __repr__(self):
        return 'Seq2seq_{0}_seq2seq_units_{1}_layers={2}_fnn_units_{3}_layers'.format(self.kernel_size, self.n_block, self.nb_hidden_units, self.nb_layers)

    def build(self):
        # input
        i = Input(shape=self.input_shape)

        # position embedding
        embed = TrigPosEmbedding(mode=TrigPosEmbedding.MODE_CONCAT,output_dim=2)
        x = embed(i)
        x = Dense(1, activation=None)(x)
        # Seq2seq block
        for _ in range(self.n_block):
            A = Conv1D(self.input_shape[1], self.kernel_size, padding="same", activation=None, input_shape=self.input_shape[1:])(x)
            B = Conv1D(self.input_shape[1], self.kernel_size, padding="same", activation="sigmoid", input_shape=self.input_shape[1:])(x)
            x = Multiply()([A, B]) + x

        # FNN block
        for _ in range(self.nb_layers):
            x = Dense(self.nb_hidden_units, activation="relu")(x)

        # output
        x = Dense(1, activation=None)(x)

        return Model(inputs=[i], outputs=[x])
