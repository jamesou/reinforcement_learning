import numpy as np
from tqdm import tqdm
from collections import OrderedDict

class PID:
    def __init__(self, setpoint, kp, ki, kd, basal=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0
        self.basal = basal
        self.setpoint = setpoint
    def step(self, value):
        error = self.setpoint - value
        p_act = self.kp * error
        # print('p: {}'.format(p_act))
        self.integral += error
        i_act = self.ki * self.integral
        # print('i: {}'.format(i_act))
        d_act = self.kd * (error - self.previous_error)
        try:
            if self.basal is not None:
                b_act = self.basal
            else:
                b_act = 0
        except:
            b_act = 0
        # print('d: {}'.format(d_act))
        self.previous_error = error
        action = p_act + i_act + d_act + b_act
        return action
    def reset(self):
        self.integral = 0
        self.previous_error = 0

def pid_test(pid, env, n_days, save_path,patient,seed, full_save=False):
    env.seeds['sensor'] = seed
    env.seeds['scenario'] = seed
    env.reset()
    full_patient_state = []
    for i in tqdm(range(n_days*288)):
        act = pid.step(env.env.CGM_hist[-1])
        state, reward, done, info = env.step(action=act)
        full_patient_state.append(info['patient_state'])
    full_patient_state = np.stack(full_patient_state)
    # statistics = OrderedDict()
    # statistics['Risk'] = [env.avg_risk()]
    # statistics['MagniRisk'] = [env.avg_magni_risk()]
    # bg, euglycemic, hypo, hyper, ins = env.glycemic_report()
    # statistics['Glucose'] = [np.mean(bg)]
    # statistics['MinBG'] = [min(bg)]
    # statistics['MaxBG'] = [max(bg)]
    # statistics['Insulin'] = [np.mean(ins)]
    # statistics['MinIns'] = [min(ins)]
    # statistics['MaxIns'] = [max(ins)]
    # statistics['GLen'] = [len(bg)]
    # statistics['Euglycemic'] = [euglycemic]
    # statistics['Hypoglycemic'] = [hypo]
    # statistics['Hyperglycemic'] = [hyper]
    if full_save:
        return env.env.show_history(), full_patient_state
    else:
        hist = env.env.show_history()
        hist.to_csv(f'{save_path}/{patient}_{seed}.csv')
        statistics = env.env.summary()
        return {'hist': hist, 'kp': pid.kp, 'ki': pid.ki, 'kd': pid.kd,'summary':statistics}
