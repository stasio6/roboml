ALGO_NAME = 's2v-DAgger-AAC-SAC-Q-filter'
# State-to-Visual DAgger, plus SAC loss, the SAC uses AAC

import os
import argparse
import random
import time
from distutils.util import strtobool

os.environ["OMP_NUM_THREADS"] = "1"

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import DictReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import datetime
from collections import defaultdict
from functools import partial


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=None,
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="s2v-DAgger-baselines",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="PickCube-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=300_000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.8,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.01,
        help="target smoothing coefficient (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=64,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=4000,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=1,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--warmup-steps", type=int, default=0,
        help="the number of warmup steps")
    parser.add_argument("--bc-loss-th", type=float, default=0.01, # important for training time
        help="if the bc loss is smaller than this threshold, then stop training and collect new data")
    parser.add_argument("--warmup-policy", type=int, default=1, # important for not crashing
        help="when in warmup, 2 and 3 do not update q-functions, 1 and 3 do not update alpha")
    parser.add_argument("--load-alpha", type=float, default=None)

    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--eval-freq", type=int, default=30_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--sync-venv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--num-steps-per-update", type=float, default=1) # TODO: tune this
    parser.add_argument("--training-freq", type=int, default=64)
    parser.add_argument("--log-freq", type=int, default=2000)
    parser.add_argument("--save-freq", type=int, default=500_000)
    parser.add_argument("--value-always-bootstrap", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="in ManiSkill variable episode length setting, set to True if positive reawrd, False if negative reward.")
    parser.add_argument("--control-mode", type=str, default='pd_ee_delta_pos')
    parser.add_argument("--from-ckpt", type=str, default=None)
    parser.add_argument("--expert-ckpt", type=str, default='output/PickCube-v1/SAC-ms2-new/230329-142137_1_profile/checkpoints/600000.pt')
    parser.add_argument("--image-size", type=int, default=64, # we have not implemented memory optimization for replay buffer, so use small image for now
        help="the size of observation image, e.g. 64 means 64x64")


    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.script = __file__
    if args.buffer_size is None:
        args.buffer_size = args.total_timesteps
    args.buffer_size = min(args.total_timesteps, args.buffer_size)
    args.num_eval_envs = min(args.num_eval_envs, args.num_eval_episodes)
    assert args.num_eval_episodes % args.num_eval_envs == 0
    assert args.training_freq % args.num_envs == 0
    assert args.training_freq % args.num_steps_per_update == 0
    assert args.policy_frequency == 1, 'We only support 1 in s2v DAgger'
    # fmt: on
    return args

from dm_control import suite
from dm_control.suite.wrappers import action_scale
# from dm_control.mujoco.engine import Camera


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        self.num_frames = num_frames
        self.obs_mode = getattr(self.env, "obs_mode", "rgb")
        self.frames = []
        self.metadata.update({'num_frames': num_frames})
        new_rgb_shape = list(self.observation_space["rgb"].shape)
        new_rgb_shape[0] *= num_frames
        rgb_obs_dim = spaces.Box(
            low = np.repeat(self.observation_space["rgb"].low, num_frames, axis=0), 
            high = np.repeat(self.observation_space["rgb"].high, num_frames, axis=0), 
            shape=new_rgb_shape, 
            dtype=self.observation_space["rgb"].dtype
        )
        self.observation_space["rgb"] = rgb_obs_dim
    
    def observation(self):
        return {
            'rgb': np.concatenate([obs['rgb'] for obs in self.frames], axis=0),
            'state': self.frames[-1]['state'],
        }

    def step(self, actions):
        next_obs, rewards, dones, infos = self.env.step(actions)
        self.frames = self.frames[1:] + [next_obs]
        return self.observation(), rewards, dones, infos

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.frames = [obs] * self.num_frames
        return self.observation()

class DMCVisualStateGymWrapper(gym.Wrapper):
    def __init__(self, env, domain, task, action_repeat, width=84, height=84, camera_id=0):
        super().__init__(env)
        act_shp = env.action_spec().shape
        state_shape = (int(sum(np.prod(v.shape) for v in env.observation_spec().values())), )
        self.observation_space = spaces.Dict(spaces={
            "rgb" : spaces.Box(low=0, high=255, shape=(3, height, width), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.full(act_shp, env.action_spec().minimum),
			high=np.full(act_shp, env.action_spec().maximum),
            shape=act_shp, dtype=np.float32,
        )
        self.domain = domain
        self.task = task
        self.action_repeat = action_repeat
        self.ep_len = 1000 // action_repeat
        self.t = 0
        self.metadata = {'render_modes': 'rgb_array'}
        self.width = width
        self.height = height
        self.camera_id = camera_id 
        
    def _get_visual_obs(self):
        return self.render(
            "rgb",
            height=self.height,
            width=self.width,
            camera_id=self.camera_id
        ).transpose(2, 0, 1)

    def _obs_to_array(self, obs):
        return np.concatenate([v.flatten() for v in obs.values()])

    def reset(self):
        self.t = 0
        time_step = self.env.reset()
        return {
            "rgb": self._get_visual_obs(),
            "state": self._obs_to_array(time_step.observation)
        }

    def step(self, action):
        self.t += 1

        # action repeat
        reward = 0.0
        discount = 1.0
        for _ in range(self.action_repeat):
            time_step = self.env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        # convert time_step to gym format
        obs_dict = {
            "rgb": self._get_visual_obs(),
            "state": self._obs_to_array(time_step.observation)
        }
        done = time_step.last() or self.t == self.ep_len
        info = {'TimeLimit.truncated': self.t == self.ep_len} if done else {}
        return obs_dict, reward, done, info

    def render(self, mode='rgb_array', width=84, height=84, camera_id=0):
        camera_id = dict(quadruped=2).get(self.domain, camera_id)
        return self.env.physics.render(height, width, camera_id)

def make_env(env_id, seed, action_repeat=None, video_dir=None, video_trigger=None):
    """
    Adapted from https://github.com/facebookresearch/drqv2 and https://github.com/nicklashansen/tdmpc
    """
    DEFAULT_ACTION_REPEAT = {
        "cartpole": 8, 
        "reacher3d": 4, "cheetah": 4, "ball_in_cup": 4,
        "humanoid": 2, "dog": 2, "walker": 2, "finger": 2, 
    } # Same as DrQ, PLaNet, TD-MPC
    domain, task = env_id.replace('-', '_').split('_', 1)
    domain = dict(cup='ball_in_cup').get(domain, domain)
    assert (domain, task) in suite.ALL_TASKS
    if action_repeat is None:
        action_repeat = DEFAULT_ACTION_REPEAT.get(domain, 4)

    def thunk():
        # make dmc env with gym interface
        env = suite.load(domain, task, task_kwargs={'random': seed})
        env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
        env = DMCVisualStateGymWrapper(env, domain, task, action_repeat)
        env = FrameStackWrapper(env, 3)

        # other regular wrappers
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if video_dir:
            env = gym.wrappers.RecordVideo(env, video_dir, video_trigger)
        env = gym.wrappers.ClipAction(env)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)

class DrQConv(nn.Module):
    def __init__(self, in_channel, feature_dim=50):
        super().__init__()
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, self.num_filters, 3, stride=2), nn.ReLU(),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1), nn.ReLU(),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1), nn.ReLU(),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self.num_filters * 35 * 35, self.feature_dim),
            nn.LayerNorm(self.feature_dim))
        self.out_dim = feature_dim

    def forward(self, obs):
        rgb = obs['rgb'].float() / 255.0 # (B, H, W, 3*k)
        x = self.cnn(rgb)
        x = x.flatten(1)
        x = self.fc(x)
        return x

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_dimension=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space['state'].shape).prod() + np.prod(env.single_action_space.shape), hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, envs, visual_feature_dim=256, mlp_cfg=[1024, 1024], num_frames=3):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape)
        self.encoder = DrQConv(in_channel=3 * num_frames, feature_dim=visual_feature_dim)
        self.mlp = make_mlp(visual_feature_dim, mlp_cfg, last_act=True)
        self.fc_mean = nn.Linear(mlp_cfg[-1], action_dim)
        self.fc_logstd = nn.Linear(mlp_cfg[-1], action_dim)
        # action rescaling
        self.action_scale = torch.FloatTensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0)
    
    def get_feature(self, obs):
        visual_feature = self.encoder(obs)
        return self.mlp(visual_feature), visual_feature

    def forward(self, obs):
        x, visual_feature = self.get_feature(obs)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, visual_feature

    def get_eval_action(self, obs):
        x, _ = self.get_feature(obs)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, obs):
        mean, log_std, visual_feature = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean#, visual_feature

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

def to_tensor(x, device):
    if isinstance(x, dict):
        return {k: to_tensor(v, device) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)

def unsqueeze(x, dim):
    if isinstance(x, dict):
        return {k: unsqueeze(v, dim) for k, v in x.items()}
    return x.unsqueeze(dim)

def collect_episode_info(info, result=None):
    if result is None:
        result = defaultdict(list)
    for item in info:
        if "episode" in item.keys():
            print(f"global_step={global_step}, episodic_return={item['episode']['r']:.4f}")
            result['return'].append(item['episode']['r'])
            result['len'].append(item["episode"]["l"])
    return result

def evaluate(n, agent, eval_envs, device):
    print('======= Evaluation Starts =========')
    agent.eval()
    result = defaultdict(list)
    obs = eval_envs.reset()
    while len(result['return']) < n:
        with torch.no_grad():
            action = agent.get_eval_action(to_tensor(obs, device))
        obs, rew, done, info = eval_envs.step(action.cpu().numpy())
        collect_episode_info(info, result)
    print('======= Evaluation Ends =========')
    return result


import importlib.util
import sys

def import_file_as_module(path, module_name='tmp_module'):
    spec = importlib.util.spec_from_file_location(module_name, path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = foo
    spec.loader.exec_module(foo)
    return foo

if __name__ == "__main__":
    args = parse_args()

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    tag = '{:s}_{:d}'.format(now, args.seed)
    if args.exp_name: tag += '_' + args.exp_name
    log_name = os.path.join(args.env_id, ALGO_NAME, tag)
    log_path = os.path.join(args.output_dir, log_name)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=log_name.replace(os.path.sep, "__"),
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    import json
    with open(f'{log_path}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    VecEnv = gym.vector.SyncVectorEnv if args.sync_venv else gym.vector.AsyncVectorEnv
    envs = VecEnv(
        [make_env(args.env_id, args.seed + i) for i in range(args.num_envs)]
    )
    eval_envs = VecEnv(
        [make_env(args.env_id, args.seed + 1000 + i,
                video_dir=f'{log_path}/videos' if args.capture_video and i == 0 else None, 
                video_trigger=lambda x: x % (args.num_eval_episodes // args.num_eval_envs) == 0 ) 
        for i in range(args.num_eval_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    print(envs.single_action_space)
    print(envs.single_observation_space)

    # agent setup
    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    if args.from_ckpt is not None:
        raise NotImplementedError()

    # expert setup
    from os.path import dirname as up
    args.expert_ckpt = 'checkpoints/' + args.env_id + '/checkpoints/' + args.env_id + ".pt"
    assert args.env_id in args.expert_ckpt, 'Expert checkpoint should be trained on the same env'
    expert_dir = up(up(args.expert_ckpt))
    import json
    with open(f'{expert_dir}/args.json', 'r') as f:
        expert_args = json.load(f)
    m = import_file_as_module(expert_args['script'])

    class DummyObject: pass
    dummy_env = DummyObject()
    dummy_env.single_observation_space = envs.single_observation_space['state']
    dummy_env.single_action_space = envs.single_action_space
    
    for key in ['Agent', 'Actor', 'QNetwork']:
        if hasattr(m, key):
            Expert = m.__dict__[key]
            break
    expert = Expert(dummy_env).to(device)
    checkpoint = torch.load(args.expert_ckpt)
    for key in ['agent', 'actor', 'q']:
        if key in checkpoint:
            expert.load_state_dict(checkpoint[key])
            break
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1.load_state_dict(checkpoint['qf1']) # load critic from expert ckpt
    qf2.load_state_dict(checkpoint['qf2'])
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        if args.load_alpha is not None:
            log_alpha = torch.Tensor([np.log(args.load_alpha)]).to(device)
            log_alpha.requires_grad = True
        else:
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    envs.single_observation_space['expert_action'] = envs.single_action_space
    rb = DictReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=True,
    )

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs = envs.reset() # obs has both tensor and ndarray
    global_step = 0
    global_update = 0
    learning_has_started = False
    num_updates_per_training = int(args.training_freq // args.num_steps_per_update)
    result = defaultdict(list)
    collect_time = training_time = eval_time = 0

    start_time = time.time()
    while global_step < args.total_timesteps:

        # Collect samples from environemnts
        tic = time.time()
        for local_step in range(args.training_freq // args.num_envs):
            global_step += 1 * args.num_envs

            # ALGO LOGIC: put action logic here
            if not learning_has_started:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions, _, _ = actor.get_action(to_tensor(obs, device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            result = collect_episode_info(infos, result)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = {k:v.copy() if isinstance(v, np.ndarray) else v.clone() for k,v in next_obs.items()}
            for idx, d in enumerate(dones):
                if d:
                    t_obs = infos[idx]["terminal_observation"]
                    for key in real_next_obs:
                        real_next_obs[key][idx] = t_obs[key]
            # expert actions
            obs['expert_action'] = expert.get_eval_action(torch.Tensor(obs['state']).to(device)).detach().cpu().numpy()
            real_next_obs['expert_action'] = np.ones_like(obs['expert_action']) * np.nan # dummpy expert actions
            rb.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
        collect_time += time.time() - tic

        # ALGO LOGIC: training.
        if global_step < args.learning_starts:
            continue

        learning_has_started = True
        tic = time.time()
        for local_update in range(num_updates_per_training):
            global_update += 1
            data = rb.sample(args.batch_size)
            warmup = global_step <= args.warmup_steps
            update_q = args.warmup_policy == 0 or args.warmup_policy == 1
            update_alpha = args.warmup_policy == 0 or args.warmup_policy == 2

            if not warmup or update_q:
                # update the value networks
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations["state"], next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations["state"], next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    if args.value_always_bootstrap:
                        next_q_value = data.rewards.flatten() + args.gamma * (min_qf_next_target).view(-1)
                    else:
                        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(data.observations["state"], data.actions).view(-1)
                qf2_a_values = qf2(data.observations["state"], data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # update the policy network (SAC + DAgger)
                pi, log_pi, pi_mean = actor.get_action(data.observations)
                qf1_pi = qf1(data.observations["state"], pi)
                qf2_pi = qf2(data.observations["state"], pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                qf1_exp = qf1(data.observations["state"], data.observations["expert_action"])
                qf2_exp = qf2(data.observations["state"], data.observations["expert_action"])
                min_qf_exp = torch.min(qf1_pi, qf2_pi).view(-1)

            if warmup:
                # pure imitation to warm up
                _, _, pi_mean = actor.get_action(data.observations)
                imitation_loss = F.mse_loss(pi_mean, data.observations['expert_action'])
                actor_rl_loss = imitation_loss * 0
                actor_total_loss = imitation_loss
            else:
                # It's no more warmup - we use Q-filter
                rl_coefs = torch.le(min_qf_exp, min_qf_pi).long()
                imitation_coefs = 1 - rl_coefs
                actor_rl_loss = (((alpha * log_pi) - min_qf_pi)*rl_coefs).mean()

                # imitation_loss = F.mse_loss(pi_mean, data.observations['expert_action'])
                imitation_loss = torch.mean(torch.mean((pi_mean - data.observations['expert_action'])**2, dim=1) * imitation_coefs)

                actor_total_loss = actor_rl_loss + imitation_loss

            actor_optimizer.zero_grad()
            actor_total_loss.backward()
            actor_optimizer.step()

            if args.autotune and (not warmup or update_alpha):
                with torch.no_grad():
                    _, log_pi, _ = actor.get_action(data.observations)
                alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                alpha = log_alpha.exp().item()

            # update the target networks
            if global_update % args.target_network_frequency == 0 and (not warmup or update_q):
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if imitation_loss < args.bc_loss_th and warmup:
                break
        training_time += time.time() - tic
        print('global step:', global_step, 'imitation_loss:', imitation_loss.item())

        # Log training-related data
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            if len(result['return']) > 0:
                for k, v in result.items():
                    writer.add_scalar(f"train/{k}", np.mean(v), global_step)
                result = defaultdict(list)
            if not warmup or update_q:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_rl_loss", actor_rl_loss.item(), global_step)
            writer.add_scalar("losses/alpha", alpha, global_step)
            writer.add_scalar("losses/actor_total_loss", actor_total_loss.item(), global_step)
            writer.add_scalar("losses/imitation_loss", imitation_loss.item(), global_step)
            
            # print("SPS:", int(global_step / (time.time() - start_time)))
            tot_time = time.time() - start_time
            writer.add_scalar("charts/SPS", int(global_step / (tot_time)), global_step)
            writer.add_scalar("charts/collect_time", collect_time / tot_time, global_step)
            writer.add_scalar("charts/training_time", training_time / tot_time, global_step)
            writer.add_scalar("charts/eval_time", eval_time / tot_time, global_step)
            writer.add_scalar("charts/collect_SPS", int(global_step / collect_time), global_step)
            writer.add_scalar("charts/training_SPS", int(global_step / training_time), global_step)
            if args.autotune and (not warmup or update_alpha):
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        # Evaluation
        if (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            tic = time.time()
            result = evaluate(args.num_eval_episodes, actor, eval_envs, device)
            eval_time += time.time() - tic
            for k, v in result.items():
                writer.add_scalar(f"eval/{k}", np.mean(v), global_step)
        
        # Checkpoint
        if args.save_freq and ( global_step >= args.total_timesteps or \
                (global_step - args.training_freq) // args.save_freq < global_step // args.save_freq):
            os.makedirs(f'{log_path}/checkpoints', exist_ok=True)
            torch.save({
                'actor': actor.state_dict(), # does NOT include action_scale
                'qf1': qf1_target.state_dict(),
                'qf2': qf2_target.state_dict(),
            }, f'{log_path}/checkpoints/{global_step}.pt')

    envs.close()
    writer.close()