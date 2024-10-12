import numpy as np
from bgp.simglucose.analysis.risk import risk_index
import joblib
import torch

def reward_event(cgm_hist, **kwargs):
    if len(cgm_hist) < 1:
        return 0
    if cgm_hist[-1] > 180 or cgm_hist[-1] < 70:
        return -1
    else:
        return 1
    
def reward_day(cgm_hist, **kwargs):
    cgm_hist = np.array(cgm_hist)
    hypo_cnt = (cgm_hist < 70).sum()
    hyper_cnt = (cgm_hist > 180).sum()
    return -1 * (hypo_cnt * 0.2 + hyper_cnt * 0.1)

def risk_diff(cgm_hist, **kwargs):
    if len(cgm_hist) < 2:
        return 0
    _, _, risk_current = risk_index([cgm_hist[-1]], 1)
    _, _, risk_prev = risk_index([cgm_hist[-2]], 1)
    return risk_prev - risk_current

def risk_diff_bg(bg_hist, **kwargs):
    if len(bg_hist) < 2:
        return 0
    _, _, risk_current = risk_index([bg_hist[-1]], 1)
    _, _, risk_prev = risk_index([bg_hist[-2]], 1)
    return risk_prev - risk_current

def risk_bg(bg_hist, **kwargs):
    return -risk_index([bg_hist[-1]], 1)[-1]

def risk_diff_duration(cgm_hist, **kwargs):
    # 在Reward中加入高血糖或低血糖的持续时间
    # 70-180 定义为正常血糖
    if len(cgm_hist) < 2:
        return 0
    _, _, risk_current = risk_index([cgm_hist[-1]], 1)
    _, _, risk_prev = risk_index([cgm_hist[-2]], 1)
    duration_time=min(len(cgm_hist),12)
    duration_risk=0
    count=0
    if cgm_hist[-1]>180:
        count=1
        sign=1
    elif cgm_hist[-1]<70:
        count=1
        sign=0
    if count>0:
        for i in range(2,duration_time+1):
            if sign==1:
                if cgm_hist[-i]<180:
                    break
                else:
                    duration_risk+=risk_index([cgm_hist[-i]],1)[2]
            elif sign==0:
                if cgm_hist[-i]>70:
                    break
                else:
                    duration_risk+=risk_index([cgm_hist[-i]],1)[2]
    return risk_prev - risk_current-duration_time/12

def magni_reward(bg_hist, **kwargs):
    bg = max(1, bg_hist[-1])
    fBG = 3.5506*(np.log(bg)**.8353-3.7932)
    risk = 10 * (fBG)**2
    return -1*risk

def magni_risk(bg, **kwargs):
    bg = max(1, bg)
    fBG = 3.5506*(np.log(bg)**.8353-3.7932)
    risk = 10 * (fBG)**2
    return risk

def magni_reward_duration(bg_hist, **kwargs):
    # The duration of high or low blood sugar is added to the Reward
    bg = max(1, bg_hist[-1])
    fBG = 3.5506*(np.log(bg)**.8353-3.7932)
    risk = 10 * (fBG)**2
    duration_time = min(len(bg_hist), 12)
    duration_risk = 0
    count=0
    sign=-1
    if bg_hist[-1] > 180:
        count = 1
        sign = 1
    elif bg_hist[-1] < 70:
        count = 1
        sign = 0
    if count > 0:
        for i in range(2, duration_time + 1):
            if sign == 1:
                if bg_hist[-i] < 180:
                    break
                else:
                    duration_risk += magni_risk(bg_hist[-i])
            elif sign == 0:
                if bg_hist[-i] > 70:
                    break
                else:
                    duration_risk += magni_risk(bg_hist[-i])
    return -1*risk-duration_risk/12