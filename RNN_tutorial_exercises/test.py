# from tasks import PerceptualDecisionMakingMod1

# env = PerceptualDecisionMakingMod1()
# trial = env.new_trial()
# ob, gt = env.ob, env.gt
# print(env.start_t, env.end_t)
# print(env.dt)
# print(env.tmax)
# print(env.start_ind, env.end_ind)
# print(env.in_period)

# print('Trial information', trial)
# print('Observation shape is ', ob.shape)
# print('Groundtruth shape is ', gt.shape)

import numpy as np

# 创建一个一维数组
arr = np.array([1, 2, 3])

# 定长
length = 5

# 在数组的末尾补零到定长
arr_padded = np.pad(arr, (0, length - len(arr)), 'constant', constant_values=0)

print(arr_padded)  # 输出：[1 2 3 0 0]