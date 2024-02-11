import gym
import neurogym as ngym

def get_dataset(task_name, seq_len, batch_size, **keywords) -> ngym.Dataset:
    # env = gym.make(task_name, keywords)
    dataset = ngym.Dataset(task_name, env_kwargs=keywords, batch_size=batch_size, seq_len=seq_len)
    return dataset