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

"""
This script runs the manual BB baseline,用药物治疗
"""

def bb_test(name, seed, n_days, full_path,q,p,residual_bolus,source_path):
    carb_error_mean = 0
    carb_error_std = .2
    carb_miss_prob = .05
    sample_time = 5
    # name='child#001'
    u2ss = p.query('Name=="{}"'.format(name)).u2ss.item()
    bw = p.query('Name=="{}"'.format(name)).BW.item()
    basal = u2ss * bw / 6000
    cr = q.query('Name=="{}"'.format(name)).CR.item()
    cf = q.query('Name=="{}"'.format(name)).CF.item()
    cnt = bbc.ManualBBController(target=140, cr=cr, cf=cf, basal=basal, sample_rate=sample_time,
                                 use_cf=True, use_bol=True, cooldown=180, corrected=True,
                                 use_low_lim=True, low_lim=140)

    res_dict = {}
    # reward_fun=risk_bg
    # reward_fun=magni_reward
    reward_fun=risk_diff

    env = bgp_env.DeepSACT1DEnv(reward_fun=reward_fun,
                                patient_name=name,
                                seeds={'numpy': seed,
                                       'sensor': seed,
                                       'scenario': seed},
                                reset_lim={'lower_lim': 10, 'upper_lim': 1000},
                                time=False, meal=False, bw_meals=True,
                                load=False, gt=False, n_hours=4,
                                norm=False, time_std=None, action_cap=None, action_bias=0,
                                action_scale=1, meal_announce=None,
                                residual_basal=False, residual_bolus=residual_bolus,
                                residual_PID=False,
                                fake_gt=False, fake_real=False,
                                suppress_carbs=False, limited_gt=False,
                                termination_penalty=None, hist_init=True, harrison_benedict=True, meal_duration=5,
                                source_path=source_path,source_dir='/root/projects/reinforcement_learning')
    action = cnt.manual_bb_policy(carbs=0, glucose=140)
    print(name)
    print(action)
    print(env.ideal_basal)
    ep_r=0
    for i in tqdm(range(n_days * int(1440/sample_time))):
        o, r, d, info = env.step(action=action.basal+action.bolus)
        print(f"i:{i}")
        print(r)
        ep_r+=r
        print(ep_r)
        bg = env.env.CGM_hist[-1]

        carbs = info['meal'] * 5
        if np.random.uniform() < carb_miss_prob:
            carbs = 0
        err = np.random.normal(carb_error_mean, carb_error_std)
        carbs = carbs + carbs * err
        action = cnt.manual_bb_policy(carbs=carbs, glucose=bg)
        if action.bolus>0:
            # print((i%(1440/5))/12,action.bolus/action.basal)
            print((i%(1440/5))/12,action.bolus/action.basal)
        print(action)
        print(bg)
    print(f'患者为{name},seed为{seed},平均risk为{ep_r/(n_days * int(1440/sample_time))}')
    hist = env.env.show_history()[288:]
    res_dict['person'] = name
    res_dict['seed'] = seed
    res_dict['bg'] = hist['BG'].mean()
    res_dict['risk'] = hist['Risk'].mean()
    res_dict['hyper'] = (hist['BG'] > 180).sum() / len(hist['BG'])
    res_dict['hypo'] = (hist['BG'] < 70).sum() / len(hist['BG'])
    res_dict['event'] = res_dict['hyper'] + res_dict['hypo']
    res_dict['accum_r'] = ep_r
    print(hist)
    joblib.dump(hist, '{}/bb_{}_seed{}.pkl'.format(full_path, name, seed))
    statistics = OrderedDict()
    statistics['Risk'] = [env.avg_risk()]
    statistics['MagniRisk'] = [env.avg_magni_risk()]
    bg, euglycemic, hypo, hyper, ins = env.glycemic_report()
    statistics['Glucose'] = [np.mean(bg)]
    statistics['MinBG'] = [min(bg)]
    statistics['MaxBG'] = [max(bg)]
    statistics['Insulin'] = [np.mean(ins)]
    statistics['MinIns'] = [min(ins)]
    statistics['MaxIns'] = [max(ins)]
    statistics['GLen'] = [len(bg)]
    statistics['Euglycemic'] = [euglycemic]
    statistics['Hypoglycemic'] = [hypo]
    statistics['Hyperglycemic'] = [hyper]
    print(statistics)
    # return res_dict
    joblib.dump(statistics, '{}/bb_{}_seed{}.pkl'.format(full_path, name, seed))
    return statistics

 