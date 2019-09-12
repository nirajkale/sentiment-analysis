from keras.preprocessing.text import Tokenizer
from ArgsManager import ArgsManager
import json
from os import path
import numpy as np
from TextProcessing import SimpleProcessing
from DataChinker import DataChinker
from common import shuffle_arrays_in_unison, save_tokenizer, load_tokenizer, ensure_dir
from keras import layers, models, optimizers ,callbacks, regularizers
from TFLogs import TrainValTensorBoard

def create_dataset(args):
    data_dict = json.load(open( args["processed-data"], 'r'))
    data = data_dict['data']
    labels = np.array(data_dict['labels'])
    labels -= labels.min()
    labels /= labels.max()
    text_pipe = SimpleProcessing( args)
    print('parsing texts')
    data = text_pipe.parse_texts( data)
    tokenizer = Tokenizer( num_words= args['bow_num_words'])
    print('updating tokenizer vocab')
    tokenizer.fit_on_texts( data)
    print('creating bow matrix')
    x= tokenizer.texts_to_matrix(data) #(m , n)
    print('saving')
    np.save(path.join(args['processed-data-dir'], 'x.npy'), data)
    np.save(path.join(args['processed-data-dir'], 'y.npy'), labels)
    save_tokenizer( tokenizer, args['processed-data-dir'])
    print('done')

if __name__ == "__main__":
    
    args= ArgsManager(use_app_data= True)
    app_data_dir = app_data_dir = path.join( path.dirname(__file__), 'app_data')
    # create_dataset(args)
    print('loading data')
    data = np.load(path.join(args['processed-data-dir'], 'x.npy'))
    data = data.tolist()
    labels = np.load(path.join(args['processed-data-dir'], 'y.npy'))
    tokenizer = load_tokenizer(args['processed-data-dir'])
    print('doing texts_to_matrix')
    x= tokenizer.texts_to_matrix(data) #(m , n)

    # chinker = DataChinker()
    # _ = chinker.prepare_slices( labels, spans=[0.8, 0.2], is_one_hot= False)
    x_train, x_test = x[:40000], x[40000:]
    y_train, y_test  = labels[:40000], labels[40000:]

    x_train, y_train = shuffle_arrays_in_unison([x_train, y_train])
    x_test, y_test = shuffle_arrays_in_unison([x_test, y_test])

    n= x_train.shape[1]

    x= layers.Input( shape= (n,))
    a = layers.Dense(units= 512, activation= 'relu', kernel_regularizer= regularizers.l2())(x)
    a = layers.Dense(units= 64, activation= 'relu')(a)
    y = layers.Dense(units= 1, activation= 'relu')(a)

    model = models.Model(x,y, name= 'bow_v1')
    model.compile( optimizer = optimizers.adam(lr= 0.001), loss='mse')

    variation = 'lr_0.0001_e25'
    model_log_dir = ensure_dir( path.join( app_data_dir, 'logs'), model.name, variation)
    model_chk_path = ensure_dir( path.join( app_data_dir, 'checkpoints'), model.name, variation, '{epoch:02d}-{val_loss:.2f}.hdf5', ignore_filename= True)
    
    cb_checkpoint = callbacks.ModelCheckpoint(filepath= model_chk_path,
                                            monitor= 'val_loss',
                                            save_best_only= True,
                                            save_weights_only= False,
                                            mode='auto',
                                            period=1)
    cb_logs = TrainValTensorBoard(log_dir= model_log_dir, write_graph= True)

    _ = model.fit( x = x_train, y = y_train,
                    batch_size = 32,
                    epochs = 15,
                    verbose = True,
                    callbacks= [ cb_checkpoint, cb_logs],
                    validation_data = ( x_test, y_test)
                )
    print('done')




    



