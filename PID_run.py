import numpy as np
import pandas as pd
import joblib
import os
from joblib import Parallel, delayed
from bgp.rl import pid
from bgp.rl.reward_functions import risk_diff,magni_reward_duration
import bgp.simglucose.envs.simglucose_gym_env as bgp_env
from datetime import datetime
import csv
"""
This script was used to tune the PID and PID-MA baselines. It performs an iterative grid search with exponential
refinement over possible parameters.
"""
data_dir = 'saves' # '/data/dir'
source_dir = '/root/projects/reinforcement_learning'  # '/source/dir'
name = 'pid'
person_grid = (['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
               ['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
               ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
# person_grid = (['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
# person_grid = (['adolescent#0{}'.format(str(i).zfill(2)) for i in [7]] +
#                ['child#0{}'.format(str(i).zfill(2)) for i in [7]] +
#                ['adult#0{}'.format(str(i).zfill(2)) for i in [7]])
# person_grid=(['adolescent#001'])
n_iter = 1
n_days = 3
n_seeds = 5
n_dim = 5
full_save = True
n_jobs = 2
residual_bolus = True
start_time = datetime(2024, 8, 8, 0, 0, 0)
best_setting_dict_prev = {}
if __name__=='__main__':
    pid_para_file = '{}/bgp/simglucose/params/pid_params.csv'.format(source_dir)
    pid_df = pd.read_csv(pid_para_file,engine='python')
    for k in range(n_iter):
        # print(k)
        itername = name+'_{}'.format(k)
        full_path = '{}/{}'.format(data_dir, itername)
        if os.path.exists(full_path):
            os.removedirs(full_path)
        os.mkdir(full_path)
        for person in person_grid:
            print(f'Running person:{person}')
            config_arr = []
            env = bgp_env.RLT1DEnv( reward_fun=magni_reward_duration,patient_name=person,
                                    seeds={'numpy': 0,
                                          'sensor': 0,
                                          'scenario': 0}, bw_meals=True, n_hours=4,
                                    meal_announce=None,meal_duration=5,
                                    residual_bolus=residual_bolus,
                                    termination_penalty=1e5,
                                    update_seed_on_reset=True,
                                    hist_init=True,
                                    start_date=start_time, harrison_benedict=True, 
                                    source_dir=source_dir)
            if person not in pid_df.name.values:
                raise ValueError('{} not in PID csv'.format(person))
            # setpoint is the target blood glucose value
            pid_params = pid_df.loc[pid_df.name == person].squeeze()
            # set PID
            config_arr = []
            for seed in range(n_seeds):
                seed += n_seeds
                controller = pid.PID(setpoint=pid_params.setpoint,
                           kp=pid_params.kp, ki=pid_params.ki, kd=pid_params.kd)
                config_arr.append({'controller': controller, 'seed': seed})
            res_arr = Parallel(n_jobs=n_jobs)(delayed(pid.pid_test)(env=env,
                                                pid=config['controller'],
                                                n_days=n_days,
                                                save_path= data_dir+"/"+itername,
                                                patient=person,                                                           
                                                seed=config['seed']) for config in config_arr)
            res_grid = {}
            for res in res_arr:
                key = (res['kp'], res['ki'], res['kd'])
                if key not in res_grid:
                    res_grid[key] = []
                res_grid[key].append(res['hist'])
            filepath = '{}/{}/{}.pkl'.format(data_dir, itername, person)
            joblib.dump(res_grid, filepath)
            csv_file = full_path+'/summary.csv'
            # Write to CSV
            print(f"csv_file:{csv_file}")
            with open(csv_file, mode='a', newline='') as file:
                fieldnames =res_arr[0]['summary'].keys()
                # Create a DictWriter object, passing the file and fieldnames
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                # Write the header
                writer.writeheader()
                for res in res_arr:
                    writer.writerow(res['summary'])
            print(f'Finished running:{person}')
        # generate next grid    
        # per_patient_perf = []
        # for pat in person_grid:
        #     filepath = '{}/{}/{}.pkl'.format(data_dir, itername, pat)
        #     dat = joblib.load(filepath)
        #     for key in dat:
        #         for seed in range(n_seeds):
        #             d = {'name': pat, 'kp': key[0], 'ki': key[1], 'kd': key[2], 'seed': seed}
        #             d['risk'] = dat[key][seed]['Magni_Risk'].mean()
        #             d['euglycemic'] = np.logical_and(dat[key][seed]['BG'] < 180, dat[key][seed]['BG'] > 70).sum() / len(
        #                 dat[key][seed]['BG'])
        #             per_patient_perf.append(d)
        # df_pid = pd.DataFrame.from_dict(per_patient_perf)
        # best_setting_dict = {}
        # for person_name in df_pid['name'].unique():
        #     df_pat = df_pid.query('name == "{}"'.format(person_name))
        #     best_perf = np.infty
        #     best_settings = None
        #     for kp in df_pat['kp'].unique():
        #         for ki in df_pat['ki'].unique():
        #             for kd in df_pat['kd'].unique():
        #                 perf = df_pat.query('kp == {} and ki == {} and kd == {}'.format(kp, ki, kd))['risk']
        #                 assert len(perf) == n_seeds
        #                 if perf.mean() < best_perf:
        #                     best_perf = perf.mean()
        #                     best_settings = (kp, ki, kd)
        #     # Get the best parameters for each patient
        #     best_setting_dict[person_name] = (best_settings, best_perf)
        # k_args = ['kp', 'ki', 'kd']
        # patient_grid_dict = {}
        # for person_name in df_pid['name'].unique():
        #     patient_grid_dict[person_name] = {'kp': None, 'ki': None, 'kd': None}
        #     for k_ind in range(3):
        #         k_type = k_args[k_ind]
        #         grid = list(np.sort(df_pid.query('name=="{}"'.format(person_name))[k_type].unique())[::-1])
        #         prev_best = best_setting_dict_prev[person_name][0][k_ind]
        #         curr_best = best_setting_dict[person_name][0][k_ind]
        #         perf_grid = []
        #         for k_val in grid:
        #             perf_grid.append(df_pid.query('name=="{}" and {}=={}'.format(person_name, k_type, k_val))['euglycemic'].mean())
        #         patient_grid_dict[person_name][k_type] = curr_best #rh.update_grid_dict(grid, prev_best, curr_best, n_dim, perf_grid)
        # joblib.dump((patient_grid_dict, best_setting_dict), '{}/{}/grid_and_settings.pkl'.format(data_dir, itername))
        # grid = patient_grid_dict    
        # best_setting_dict_prev = best_setting_dict