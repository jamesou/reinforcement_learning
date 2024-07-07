import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from collections import namedtuple,OrderedDict
import joblib
from joblib import Parallel, delayed
from bgp.rl.reward_functions import *
import bgp.simglucose.envs.simglucose_gym_env as bgp_env
import bgp.simglucose.controller.basal_bolus_ctrller as bbc
from bgp.rl.bb import bb_test
"""
This script runs the BB baseline,用药物治疗
"""
Seeds = namedtuple('seeds', ['numpy_seed', 'sensor_seed', 'scenario_seed'])
n_jobs = 2
# n_jobs = 10
name = 'basal_bolus'
server = 'mld4'
source_path = '/root/projects/reinforcement_learning/bgp'
save_path = 'saves'
full_path = '{}/{}'.format(save_path, name)
residual_bolus = True 

if not os.path.exists(full_path):
    os.mkdir(full_path)

# patients = (['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
#             ['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
#             ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
patients = (['adolescent#0{}'.format(str(i).zfill(2)) for i in [7]] +
            ['child#0{}'.format(str(i).zfill(2)) for i in [7]] +
            ['adult#0{}'.format(str(i).zfill(2)) for i in [7]])
n_days = 14
# seeds = [i for i in range(50)]
seeds = [i for i in range(5)]
q = pd.read_csv('{}/simglucose/params/Quest2.csv'.format(source_path))
p = pd.read_csv('{}/simglucose/params/vpatient_params.csv'.format(source_path))
residual_bolus = True
settings = itertools.product(patients, seeds)
if __name__=='__main__':
    res_list = Parallel(n_jobs=n_jobs)(delayed(bb_test)(name=s[0], seed=s[1],
                                                        n_days=n_days,
                                                        full_path=full_path,
                                                        q=q,p=p,
                                                        residual_bolus=residual_bolus,
                                                        source_path=source_path
                                                        ) for s in settings)

    
                                                  