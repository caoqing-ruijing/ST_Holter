import sys
sys.path.append('../')
import yaml
from datetime import datetime
from tools import dir_utils


class Struct(object):
    '''
    trans dict to object
    input: dict    
    out: object
    
    {'a': 1, 'b': {'c': 2}, 'd': ['hi', {'foo': 'bar'}]}
    x.a >>> 1
    x.b.c >>> 2
    x.d[1].foo >>> 'bar'
    '''
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)): 
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value

def load_yaml(def_cfg_path,saveYaml2output=False):
    with open(def_cfg_path, 'r') as stream:
        default_cfg = yaml.safe_load(stream)
    
    # for train, not val
    if saveYaml2output is True:
        timestap = datetime.now().strftime("%Y_%m_%d-%H_%M")
        default_cfg['RUN_DATE']=timestap
        out_dir = '{}{}_{}/'.format(default_cfg['DATASET']['MODEL_SAVE_DIR'],default_cfg['TRY_TIME'],default_cfg['RUN_DATE']) 
        default_cfg['SAVE_DIR']=out_dir
        dir_utils.mkdir_without_del(out_dir)

        train_cfg_path = out_dir+'train_cfg.yaml'

        with open(train_cfg_path, 'w') as outfile:
            yaml.dump(default_cfg, outfile, default_flow_style=False)
            print('===>',train_cfg_path,'saved')
    
    opt = Struct(default_cfg)
    return opt


if __name__ == "__main__":

    file_path = '/home/yons/qiaoran/holter_ST/yaml/noise.yaml'

    # with open(file_path, 'r') as stream:
    #     data_loaded = yaml.safe_load(stream)

    # print('data_loaded',data_loaded)

    opt = load_yaml(file_path,saveYaml2output=True)
    print(opt.DATASET.MODEL_SAVE_DIR)
    # for k,v in data_loaded.items():
    #     print(k,v)



