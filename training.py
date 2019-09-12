import json
from Tokenizer import BlanketTokenizer
from ArgsManager import ArgsManager
import numpy as np
from os import path
from keras import layers, models, optimizers, regularizers
from keras.callbacks import TensorBoard
from TFLogs import TrainValTensorBoard
import os

def create_dataset(args):
    data_dict = json.load(open(args["processed-data"], 'r'))
    data = data_dict['data']
    labels = np.array(data_dict['labels'])
    print(labels.shape)
    pipeline = BlanketTokenizer(args)
    data = pipeline.parse_texts(data)
    data , labels = shuffle_arrays_in_unison([data, labels])
    print('saving')
    np.save(path.join('processed_data', 'x.npy'), data)
    np.save(path.join('processed_data', 'y.npy'), labels)
    print('done')

if __name__ == "__main__":

    args= ArgsManager(use_app_data= True)
    data = np.load(path.join('processed_data', 'x.npy'))
    labels = np.load(path.join('processed_data', 'y.npy'))
    emb_matrix = np.load(path.join(args['active-embedding-lookup-dir'],'emb.npy'))
    vocab_size = emb_matrix.shape[0]

    chinker = DataChinker()
    _ = chinker.prepare_slices(labels, spans=[0.8, 0.2], is_one_hot= False)
    x_train, x_test = chinker.slice_matrix(data)
    y_train, y_test = chinker.slice_matrix(labels)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    x= layers.Input(shape = (args['sequence_size'],), name='input')
    a = layers.Embedding(input_dim = vocab_size, output_dim = 100, weights = [emb_matrix], trainable = True)(x)
    a = layers.Bidirectional(layers.LSTM(64, return_sequences = False, name='lstm', recurrent_dropout = 0.15))(a)
    a = layers.Dense(64, activation='relu',name ='hidden_dense', kernel_regularizer = regularizers.l2(l = 0.03))(a)
    y_hat = layers.Dense(1, activation = 'relu', name='output')(a)

    model = models.Model(x,y_hat, name='rnn_model1.5')
    model.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics = ['acc'])
    print(model.summary())

    log_dir = path.join('logs', model.name)
    if not path.exists(log_dir):
        os.makedirs(log_dir)

    tb_callback = TrainValTensorBoard(log_dir, write_graph = True)

    model.fit( x = x_train, y = y_train, batch_size= 64, epochs= 7, callbacks= [tb_callback], 
            validation_data= (x_test, y_test))

    model.save('model.h5')

    print('evaluating model..')
    scores = model.evaluate(x = x_test, y = y_test)
    print(scores)
    print('done')




