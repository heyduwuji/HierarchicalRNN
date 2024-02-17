from tasks import PerceptualDecisionMakingMod1

env = PerceptualDecisionMakingMod1()
trial = env.new_trial()
ob, gt = env.ob, env.gt
print(env.start_t, env.end_t)
print(env.dt)
print(env.tmax)
print(env.start_ind, env.end_ind)
print(env.in_period)

print('Trial information', trial)
print('Observation shape is ', ob.shape)
print('Groundtruth shape is ', gt.shape)