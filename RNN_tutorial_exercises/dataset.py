import gym
import neurogym as ngym

def get_dataset(task_name, seq_len, batch_size, **keywords):
    task_name = 'PerceptualDecisionMaking-v0'
    keywords = {'dt': 20, 'timing': {'stimulus': 1000}}
    env = gym.make(task_name, keywords)

    seq_len = 100
    batch_size = 16
    dataset = ngym.Dataset(env, batch_size, seq_len)
    return dataset