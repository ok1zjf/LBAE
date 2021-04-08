__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.8'
__status__ = "Research"
__date__ = "2/1/2020"
__license__= "MIT License"

import random
import torch
import json
import numpy as np
import sys
import torch
import json

# ===================================================
class Params:
    def __init__(self):
        rnd_seed = 12345
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)
        torch.cuda.manual_seed(rnd_seed)

        self.log_stdout = True
        self.use_cuda = True
        self.cuda_device = 1   # -1 for all available GPUs
        # self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.lr = [0.00005]
        self.lr_epoch = [0]
        self.batch_size = 256  # train batch size
        self.epoch_start = None
        self.iter_start = -1
        self.print_every_batch = 10
        self.keep_last_models = 15

    def diff(self, params):
        ''' shows changes between params -> self '''

        diffs = []
        vars = [attr for attr in dir(self) if not callable(getattr(self,attr)) and not (attr.startswith("__") or attr.startswith("_"))]

        info_str = ''
        for i, var in enumerate(vars):
            val = getattr(self, var)
            # if isinstance(val, torch.Tensor):
                # val = val.data.cpu().numpy().tolist()[0]
            info_str += '['+str(i)+'] '+var+': '+str(val)+'\n'

            pval = getattr(params, var)
            if pval != val:
                diffs.append([var, pval, val])

        return diffs

    def diff_str(self, params):
        diff = self.diff(params)
        out = ''
        for var, past, current in diff:
            out += var+': '+str(past)+' -> '+str(current)+'\n'
        return out

    def __getattr__(self, item):
        return None

    def save(self, filename='config.json'):
        with open(filename, 'w') as f: 
            json.dump(self.__dict__, f, indent=3, sort_keys=True)
        return

    def load(self, filename='config.json'):
        try:
            with open(filename, 'r') as f: 
                args = json.load(f)
                self.load_from_args(args)
        except:
            return False
        return True

    def load_from_args(self, args):
        for key in args:
            # print(key, args[key])
            setattr(self, key, args[key])

    def load_from_sys_args(self, args):
        args ={}
        for i, val in enumerate(sys.argv):
            if i == 0: continue
            toks = val.split('=')
            if len(toks)==1:
                val = True
            else:
                try:
                    val = float(toks[1])
                except:
                    val = toks[1]
                    pass
            
            args[toks[0]] = val
        self.load_from_args(args)
        return


    def __str__(self):
        # Ignore all functions and variables starting with _ and __
        vars = [attr for attr in dir(self) if not callable(getattr(self,attr)) and not (attr.startswith("__") or attr.startswith("_"))]

        info_str = ''
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, torch.Tensor):
                val = val.data.cpu().numpy().tolist()[0]
            info_str += '['+str(i)+'] '+var+': '+str(val)+'\n'

        return info_str

#=================================================================================
if __name__ == "__main__":
    print("NOT AN EXECUTABLE!")

