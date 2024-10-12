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
from datetime import datetime
"""
This script runs the manual BB baseline 
"""
def bb_test(name, seed, n_days, full_path,q,p,residual_bolus,source_path):
    carb_error_mean = 0
    carb_error_std = .2  
    carb_miss_prob = .05  
    sample_time = 5
    u2ss = p.query('Name=="{}"'.format(name)).u2ss.item()
    bw = p.query('Name=="{}"'.format(name)).BW.item()
    basal = u2ss * bw / 6000
    cr = q.query('Name=="{}"'.format(name)).CR.item()
    cf = q.query('Name=="{}"'.format(name)).CF.item()
    cnt = bbc.ManualBBController(target=140, cr=cr, cf=cf, 
                                 basal=basal, 
                                 sample_rate=sample_time,
                                 use_cf=True, use_bol=True, 
                                 cooldown=180, corrected=True,
                                 use_low_lim=True, low_lim=140)
    # reward_fun=risk_bg
    reward_fun=magni_reward_duration
    # reward_fun=risk_diff
    start_time = datetime(2024, 8, 8, 0, 0, 0)
    env = bgp_env.RLT1DEnv(reward_fun=reward_fun, patient_name=name, seeds={'numpy': seed,
                                       'sensor': seed,
                                       'scenario': seed}, bw_meals=True, n_hours=4,
                                meal_announce=None,meal_duration=5,
                                residual_bolus=residual_bolus,
                                termination_penalty=1e5, 
                                hist_init=True,start_date=start_time,
                                harrison_benedict=True, 
                                update_seed_on_reset=True,
                                source_dir=source_path)
    action = cnt.manual_bb_policy(carbs=0, glucose=140)
    ep_r=0
    for i in tqdm(range(n_days * int(1440/sample_time))):
        o, r, d, info = env.step(action=action.basal+action.bolus)
        ep_r+=r
        bg = env.env.CGM_hist[-1]
        carbs = info['meal'] * 5
        if np.random.uniform() < carb_miss_prob:
            carbs = 0
        err = np.random.normal(carb_error_mean, carb_error_std)
        carbs = carbs + carbs * err
        action = cnt.manual_bb_policy(carbs=carbs, glucose=bg)
    print(f'patient:{name},seed:{seed},average risk:{ep_r/(n_days * int(1440/sample_time))}')
    hist = env.env.show_history()
    hist.to_csv(f'{full_path}/{name}_{seed}.csv')
    joblib.dump(hist, '{}/bb_{}_seed{}.pkl'.format(full_path, name, seed))
    summary = env.env.summary()
    print(summary)
    return {'hist': hist, 'summary':summary}
   

 