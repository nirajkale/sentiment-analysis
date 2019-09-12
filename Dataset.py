from os import path
import os
import json
from common import ensure_dir
from ArgsManager import ArgsManager

#read positive
data = []
labels = []

def read_files(dirname, use_regression = True, label = 1):
    for file in os.listdir(dirname):
        f_name = path.join(dirname, file)
        with open(f_name,'r',encoding="utf8") as f:
            data.append(f.read())
            if use_regression:
                label = float(file[ file.index('_')+1:file.index('.')])
            labels.append(label)


if __name__ == "__main__":
    
    args= ArgsManager(use_app_data= True)
    basedir= args['raw-dataset-dir']
    
    print('collecting raw samples..')
    read_files( path.join(basedir,'train','pos'))
    read_files( path.join(basedir,'train','neg'), label=0)
    read_files( path.join(basedir,'test','pos'))
    read_files( path.join(basedir,'test','neg'), label=0)

    assert(len(data)==len(labels))
    print('total samples collected:', len(data))

    json_str = json.dumps({
        'data':data,
        'labels':labels
    }, indent=4)

    output_path = ensure_dir(args['processed-data'], ignore_filename= True)

    with open(output_path ,'w') as f:
        f.write(json_str)
    print('done')



