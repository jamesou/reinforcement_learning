import os
import pandas as pd
import itertools
from collections import namedtuple
from joblib import Parallel, delayed
from bgp.rl.reward_functions import *
from bgp.rl.bb import bb_test
import csv
"""
This script runs the BB baseline
"""
Seeds = namedtuple('seeds', ['numpy_seed', 'sensor_seed', 'scenario_seed'])
n_jobs = 2
name = 'basal_bolus'
server = 'mld4'
source_path = '/root/projects/reinforcement_learning'
save_path = 'saves'
full_path = '{}/{}'.format(save_path, name)
if not os.path.exists(full_path):
    os.mkdir(full_path)
patients = (['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
            ['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
            ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
n_days = 3
# patients = (['adult#004'])
seeds = [i for i in range(5)]
q = pd.read_csv('{}/bgp/simglucose/params/Quest2.csv'.format(source_path))
p = pd.read_csv('{}/bgp/simglucose/params/vpatient_params.csv'.format(source_path))
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
    csv_file = full_path+'/summary.csv'
    # Write to CSV
    with open(csv_file, mode='w', newline='') as file:
        fieldnames =res_list[0]['summary'].keys()
        # Create a DictWriter object, passing the file and fieldnames
        writer = csv.DictWriter(file, fieldnames=fieldnames)
         # Write the header
        writer.writeheader()
        for res in res_list:
            writer.writerow(res['summary'])
    
         
 

    
                                                  