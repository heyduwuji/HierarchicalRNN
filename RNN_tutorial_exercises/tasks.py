import neurogym as ngym
import numpy as np

class PerceptualDecisionMaking(ngym.TrialEnv):
    """Two-alternative forced choice task in which the subject has to
    integrate two stimuli to decide which one is higher on average.

    Args:
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)
        sigma: float, input noise level
        dim_ring: int, dimension of ring input and output
    """
    metadata = {
        'paper_link': 'https://www.jneurosci.org/content/12/12/4745',
        'paper_name': '''The analysis of visual motion: a comparison of
        neuronal and psychophysical performance''',
        'tags': ['perceptual', 'two-alternative', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, stim_scale=1.,
                 sigma=1.0, dim_ring=2):
        super().__init__(dt=dt)
        # The strength of evidence, modulated by stim_scale
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2]) * stim_scale
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 100,
            'stimulus': 2000,
            'delay': 0,
            'decision': 100}
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.theta = np.linspace(0, 2*np.pi, dim_ring+1)[:-1]
        self.choices = np.arange(dim_ring)

        name = {'fixation': 0, 'stimulus': range(1, dim_ring+1)}
        self.observation_space = ngym.spaces.Box(
            -np.inf, np.inf, shape=(1+dim_ring,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, dim_ring+1)}
        self.action_space = ngym.spaces.Discrete(1+dim_ring, name=name)

    def _new_trial(self, **kwargs):
        # Trial info
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
        }
        trial.update(kwargs)

        coh = trial['coh']
        ground_truth = trial['ground_truth']
        stim_theta = self.theta[ground_truth]

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        stim = np.cos(self.theta - stim_theta) * (coh/200) + 0.5
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')

        # Ground truth
        self.set_groundtruth(ground_truth, period='decision', where='choice')

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('fixation'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
    
class PerceptualDecisionMakingMod1(ngym.TrialEnv):

    def __init__(self, dt=100, rewards=None, timing=None):
        super().__init__(dt=dt)

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 100,
            'stimulus1': 2000,
            'delay': 0,
            'stimulus2': 0,
            'go': 200}
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.coherence_space = [-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08]
        self.stim_time_space = [400, 800, 1600]

        name = {'fixation': 0, 'stim_mod1': range(1, 33), 'stim_mod2': range(33, 65)}
        self.observation_space = ngym.spaces.Box(
            -np.inf, np.inf, shape=(1+ 2 * 32,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'target': range(1, 33)}
        self.action_space = ngym.spaces.Box(
            -np.inf, np.inf, shape=(1+ 32,), dtype=np.float32, name=name)
        self.n_eachring = 32     
        self.pref = np.arange(0,2*np.pi,2*np.pi/self.n_eachring)
        
    def new_batch(self):
        self.fixation_time = self.rng.uniform(100, 400)
        self.stim_time = self.rng.choice(self.stim_time_space)
        self.go_time = 500

    def _get_dist(self, original_dist):
        return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))
    
    def _get_loc(self, loc):
        dist = self._get_dist(loc - self.pref)
        dist /= np.pi / 8
        y = 0.8*np.exp(-dist**2/2)
        return y

    def _new_trial(self, **kwargs):
        # new batch
        if kwargs.get('new_batch', True):
            self.new_batch()

        # all trials in the same batch have same length
        average_strength = self.rng.uniform(0.8, 1.2)
        coherence = self.rng.choice(self.coherence_space)
        strength1 = average_strength + coherence
        strength2 = average_strength - coherence
        angle1 = self.rng.uniform(0, 2*np.pi)
        angle2 = self.rng.uniform(angle1 + 0.5 * np.pi, (angle1 + 0.5 * np.pi) % (2 * np.pi))
        target_angle = angle1 if strength1 > strength2 else angle2
        self.target_angle = target_angle
        u_mod2 = np.zeros(32)
        u_mod1_stim1 = strength1 * self._get_loc(angle1)
        u_mod1_stim2 = strength2 * self._get_loc(angle2)
        u_mod1 = u_mod1_stim1 + u_mod1_stim2
        target = self._get_loc(target_angle) + 0.05

        # set random stimulus time
        timing = {'stimulus1': int(self.stim_time), 'fixation': int(self.fixation_time), 'go': int(self.go_time)}
        self.timing.update(timing)

        # Trial info
        trial = {
            'angle 1': angle1,
            'angle 2': angle2,
            'strength 1': strength1,
            'strength 2': strength2,
            'coh': coherence
        }
        trial.update(kwargs)

        # Periods
        self.add_period(['fixation', 'stimulus1', 'delay', 'stimulus2', 'go'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus1', 'delay', 'stimulus2'], where='fixation')
        self.add_ob(0, 'go', where='fixation')
        self.add_ob(u_mod1, period=['fixation', 'stimulus1', 'delay', 'stimulus2', 'go'], where='stim_mod1')
        self.add_ob(u_mod2, period=['fixation', 'stimulus1', 'delay', 'stimulus2', 'go'], where='stim_mod2')

        # Ground truth
        self.set_groundtruth(0.85, period=['fixation', 'stimulus1', 'delay', 'stimulus2'], where='fixation')
        self.set_groundtruth(0.05, period=['go'], where='fixation')
        self.set_groundtruth(np.array([0.05] * 32), period=['fixation', 'stimulus1', 'delay', 'stimulus2'], where='target')
        self.set_groundtruth(target, period=['go'], where='target')

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('fixation'):
            if action.all() != 0:  # action = 0 means fixating
                new_trial = self.abort
        elif self.in_period('go'):
            if action.all() != 0:
                new_trial = True

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
    
class PerceptualDecisionMakingMod2(ngym.TrialEnv):

    def __init__(self, dt=100, rewards=None, timing=None):
        super().__init__(dt=dt)

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 100,
            'stimulus1': 2000,
            'delay': 0,
            'stimulus2': 0,
            'go': 200}
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.coherence_space = [-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08]
        self.stim_time_space = [400, 800, 1600]

        name = {'fixation': 0, 'stim_mod1': range(1, 33), 'stim_mod2': range(33, 65)}
        self.observation_space = ngym.spaces.Box(
            -np.inf, np.inf, shape=(1+ 2 * 32,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'target': range(1, 33)}
        self.action_space = ngym.spaces.Box(
            -np.inf, np.inf, shape=(1+ 32,), dtype=np.float32, name=name)
        self.n_eachring = 32     
        self.pref = np.arange(0,2*np.pi,2*np.pi/self.n_eachring)
        
    def new_batch(self):
        self.fixation_time = self.rng.uniform(100, 400)
        self.stim_time = self.rng.choice(self.stim_time_space)
        self.go_time = 500

    def _get_dist(self, original_dist):
        return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))
    
    def _get_loc(self, loc):
        dist = self._get_dist(loc - self.pref)
        dist /= np.pi / 8
        y = 0.8*np.exp(-dist**2/2)
        return y

    def _new_trial(self, **kwargs):
        # new batch
        if kwargs.get('new_batch', True):
            self.new_batch()

        # all trials in the same batch have same length
        average_strength = self.rng.uniform(0.8, 1.2)
        coherence = self.rng.choice(self.coherence_space)
        strength1 = average_strength + coherence
        strength2 = average_strength - coherence
        angle1 = self.rng.uniform(0, 2*np.pi)
        angle2 = self.rng.uniform(angle1 + 0.5 * np.pi, (angle1 + 0.5 * np.pi) % (2 * np.pi))
        target_angle = angle1 if strength1 > strength2 else angle2
        self.target_angle = target_angle
        u_mod1 = np.zeros(32)
        u_mod2_stim1 = strength1 * self._get_loc(angle1)
        u_mod2_stim2 = strength2 * self._get_loc(angle2)
        u_mod2 = u_mod2_stim1 + u_mod2_stim2
        target = self._get_loc(target_angle) + 0.05

        # set random stimulus time
        timing = {'stimulus1': int(self.stim_time), 'fixation': int(self.fixation_time), 'go': int(self.go_time)}
        self.timing.update(timing)

        # Trial info
        trial = {
            'angle 1': angle1,
            'angle 2': angle2,
            'strength 1': strength1,
            'strength 2': strength2,
            'coh': coherence
        }
        trial.update(kwargs)

        # Periods
        self.add_period(['fixation', 'stimulus1', 'delay', 'stimulus2', 'go'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus1', 'delay', 'stimulus2'], where='fixation')
        self.add_ob(0, 'go', where='fixation')
        self.add_ob(u_mod1, period=['fixation', 'stimulus1', 'delay', 'stimulus2', 'go'], where='stim_mod1')
        self.add_ob(u_mod2, period=['fixation', 'stimulus1', 'delay', 'stimulus2', 'go'], where='stim_mod2')

        # Ground truth
        self.set_groundtruth(0.85, period=['fixation', 'stimulus1', 'delay', 'stimulus2'], where='fixation')
        self.set_groundtruth(0.05, period=['go'], where='fixation')
        self.set_groundtruth(np.array([0.05] * 32), period=['fixation', 'stimulus1', 'delay', 'stimulus2'], where='target')
        self.set_groundtruth(target, period=['go'], where='target')

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('fixation'):
            if action.all() != 0:  # action = 0 means fixating
                new_trial = self.abort
        elif self.in_period('go'):
            if action.all() != 0:
                new_trial = True

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}