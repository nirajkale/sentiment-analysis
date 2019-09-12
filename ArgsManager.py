import json
import os
from collections.abc import Sequence
from os import path

'''
Niraj Kale: to efficiently store, retrieve & transfer args
'''

class ArgsManager(Sequence):
    
    def __init__(self, filepath=None,use_app_data = False):
        if use_app_data or filepath is None:
            filepath = path.join(path.dirname(__file__), 'args.json')
        self.filepath = filepath
        self.args = {}
        super().__init__()
        if filepath:
            if path.isfile(filepath):
                self.args= json.load(open(filepath, 'r'))
            else:
                raise Exception('Configuration file is missing')

    def add_directory(self, name, dirpath, mkdir = True):
        if mkdir and (not path.exists(dirpath)):
            os.makedirs(dirpath)
        self.args[name] = dirpath

    def set(self, name, value):
        self.args[name] = value

    def get(self,name):
        if name not in self.args:
            raise Exception('Configuration with name '+name+' is not available')
        return self.args[name]

    def __getitem__(self, name):
        return self.get(name)
    
    def __len__(self):
        return len(self.args)

    def save(self, filepath=None):
        if not filepath:
            filepath = self.filepath
        json_str = json.dumps(self.args, indent=4)
        with open(filepath,'w') as f:
            f.write(json_str)

    def remove(self,name):
        del self.args[name]
    

if __name__ == '__main__':

    filename = 'args.json'
    app_data = 'app_data'
    model_data = path.join(app_data,'model_data')
    
    args = ArgsManager(None,use_app_data= True)
    args.set('epochs', 30)
    args.set('batch_size', 64)
    args.set('vocab_size', 100000)
    args.set('validation_split', 0)
    args.set('verbose', 1)
    args.set('sequence_size', 20)
    args.set('padding_type','pre')
    args.set('num_words', 2000)
    args.set('active-dataset','ds_no_lemma_emb100_v2.1')
    args.set('active-dataset-type','embeddings')
    args.set('active-model','')
    args.set('log-dir', path.join(model_data, 'logs'))
    args.set('save-dir', path.join(model_data, 'saved_models'))
    args.set('summary-dir', path.join(model_data, 'summaries'))
    args.set('plot-dir', path.join(model_data, 'model_plots'))
    args.set('checkpoint-dir', path.join(model_data, 'model_checkpoints'))
    args.set('transition-dir', path.join(model_data, 'model_transition'))
    args.set('binaries-dir', path.join(model_data, 'model_binaries'))
    args.set('dataset-dir', path.join(app_data,'processed-datasets'))
    args.set('master-model-dir', path.join(app_data,'master_models'))
    args.set('embedding-dir', path.join(app_data,'embeddings'))
    args.add_directory('embedding-lookup-dir', path.join(app_data,'embedding-lookups'))
    args.set('active-embedding-lookup-dir','')
    args.set('save_model_plots', True)
    args.set('log_device_placement', False)
    args.set('generate_summary', True)
    args.set('padding_type', 'pre')
    args.set('nlp_lowercase', True)
    args.set('nlp_apply_lemma', False)
    args.set('nlp_simplify_entities', False)
    args.set('nlp_remove_stop_words', False)
    args.set('nlp_cleanse_text', True)
    args.set('nlp_model_name','en_core_web_sm')

    # args.save(path.join(app_data, 'args.json'))
    args.save()
    print(args.args)
    input('Press enter to close..')
