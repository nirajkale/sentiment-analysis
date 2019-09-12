import os
import tensorflow as tf
from keras.callbacks import TensorBoard
import keras.backend as K

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='logs', write_graph = True,**kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir,'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
        check_dir(training_log_dir)
        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir,'validation')
        check_dir(self.val_log_dir)
        self.write_graph = write_graph

    def set_model(self, model, add_graph = True):
        self.sess = K.get_session()                                                                                                                                                                                                                                   
        # if self.write_graph:                                                                                                                                                                                                                                          
        #     self.val_writer = tf.summary.FileWriter(self.val, self.sess.graph)                                                                                                                                                                                        
        # else:                                                                                                                                                                                                                                                         
        #     self.val_writer = tf.summary.FileWriter(self.log_dir)  
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
