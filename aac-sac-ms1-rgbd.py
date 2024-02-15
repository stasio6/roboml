ALGO_NAME = 'AAC-SAC-ms2-RGBD'

import os
import argparse
import random
import time
from distutils.util import strtobool

os.environ["OMP_NUM_THREADS"] = "1"

import gym
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
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ManiSkill2-dev",
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
    parser.add_argument("--buffer-size", type=int, default=200_000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.8,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.01,
        help="target smoothing coefficient (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=512,
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

    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--eval-freq", type=int, default=30_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--sync-venv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--num-steps-per-update", type=float, default=8)
    parser.add_argument("--training-freq", type=int, default=128)
    parser.add_argument("--log-freq", type=int, default=2000)
    parser.add_argument("--save-freq", type=int, default=200_000)
    parser.add_argument("--value-always-bootstrap", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="in ManiSkill variable episode length setting, set to True if positive reawrd, False if negative reward.")
    parser.add_argument("--control-mode", type=str, default=None)
    parser.add_argument("--from-actor-ckpt", type=str, default=None)
    parser.add_argument("--from-critic-ckpt", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=64,
        help="the size of observation image, e.g. 64 means 64x64")


    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.script = __file__
    if args.buffer_size is None:
        args.buffer_size = args.total_timesteps
        
    if "Cabinet" in args.env_id:
        args.gamma = 0.95 
    elif "Bucket" in args.env_id:
        args.gamma = 0.9
        
    args.buffer_size = min(args.total_timesteps, args.buffer_size)
    args.num_eval_envs = min(args.num_eval_envs, args.num_eval_episodes)
    assert args.num_eval_episodes % args.num_eval_envs == 0
    assert args.training_freq % args.num_envs == 0
    assert args.training_freq % args.num_steps_per_update == 0
    args.value_always_bootstrap = False
    # fmt: on
    return args

import mani_skill2.envs
from mani_skill2.utils.common import flatten_state_dict, flatten_dict_space_keys
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.vector.vec_env import VecEnvObservationWrapper
from gym.core import Wrapper, ObservationWrapper
from gym import spaces

class MS2_AAC_RGBDVecEnvObsWrapper(VecEnvObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert self.obs_mode == 'rgbd'
        self.observation_space = self.build_obs_space(env)
        example_oracle_state = self.venv.env_method('get_state_mode_obs')[0] # hack for AAC
        self.observation_space['oracle_state'] = spaces.Box(-float("inf"), float("inf"), shape=example_oracle_state.shape, dtype=np.float32)
        self.concat_fn = partial(torch.cat, dim=-1)

    def observation(self, obs):
        obs = self.convert_obs(obs, self.concat_fn)
        obs['oracle_state'] = np.vstack(self.venv.env_method('get_state_mode_obs')) # hack for AAC
        return obs
    
    @staticmethod
    def build_obs_space(env, depth_dtype=np.float16):
        obs_space = env.observation_space
        state_dim = 0
        for k in ['agent', 'extra']:
            state_dim += sum([v.shape[0] for v in flatten_dict_space_keys(obs_space[k]).spaces.values()])

        single_img_space = list(env.observation_space['image'].values())[0]
        h, w, _ = single_img_space['rgb'].shape
        k = len(env.observation_space['image'])

        return spaces.Dict({
            'state': spaces.Box(-float("inf"), float("inf"), shape=(state_dim,), dtype=np.float32),
            'rgb': spaces.Box(0, 255, shape=(h,w,k*3), dtype=np.uint8),
            'depth': spaces.Box(-float("inf"), float("inf"), shape=(h,w,k), dtype=depth_dtype),
        })
    # NOTE: We have to use float32 for gym AsyncVecEnv since it does not support float16, but we can use float16 for MS2 vec env
    
    @staticmethod
    def convert_obs(obs, concat_fn):
        img_dict = obs['image']
        new_img_dict = {
            key: concat_fn([v[key] for v in img_dict.values()])
            for key in ['rgb', 'depth']
        }
        if isinstance(new_img_dict['depth'], torch.Tensor): # MS2 vec env uses float16, but gym AsyncVecEnv uses float32
            new_img_dict['depth'] = new_img_dict['depth'].to(torch.float16)

        states = [flatten_state_dict(obs["agent"])]
        if len(obs["extra"]) > 0:
            states.append(flatten_state_dict(obs["extra"]))
        new_img_dict['state'] = np.hstack(states)

        return new_img_dict

class MS2_RGBDObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert self.obs_mode == 'rgbd'
        self.observation_space = MS2_AAC_RGBDVecEnvObsWrapper.build_obs_space(env, depth_dtype=np.float32)
        example_oracle_state = self.env.get_state_mode_obs() # hack for AAC
        self.observation_space['oracle_state'] = spaces.Box(-float("inf"), float("inf"), shape=example_oracle_state.shape, dtype=np.float32)
        self.concat_fn = partial(np.concatenate, axis=-1)

    def observation(self, obs):
        obs = MS2_AAC_RGBDVecEnvObsWrapper.convert_obs(obs, self.concat_fn)
        obs['oracle_state'] = self.env.get_state_mode_obs() # hack for AAC
        return obs


from mani_skill2.vector.wrappers.sb3 import select_index_from_dict
class AutoResetVecEnvWrapper(Wrapper):
    # adapted from https://github.com/haosulab/ManiSkill2/blob/main/mani_skill2/vector/wrappers/sb3.py#L25
    def step(self, actions):
        vec_obs, rews, dones, infos = self.env.step(actions)
        if not dones.any():
            return vec_obs, rews, dones, infos

        for i, done in enumerate(dones):
            if done:
                # NOTE: ensure that it will not be inplace modified when reset
                infos[i]["terminal_observation"] = select_index_from_dict(vec_obs, i)

        reset_indices = np.where(dones)[0]
        vec_obs = self.env.reset(indices=reset_indices)
        return vec_obs, rews, dones, infos

class GetStateObsWrapper(Wrapper):
    def get_state_mode_obs(self):
        raw_env = self.env.unwrapped
        original_obs_mode = raw_env._obs_mode
        raw_env._obs_mode = 'state'
        state_dict = self.env.unwrapped._get_obs_state_dict()
        state_obs = flatten_state_dict(state_dict)
        raw_env._obs_mode = original_obs_mode
        return state_obs

def seed_env(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

def make_vec_env(env_id, num_envs, seed, control_mode=None, image_size=None, video_dir=None, gym_vec_env=False):
    assert gym_vec_env or video_dir is None, 'Saving video is only supported for gym vec env'
    cam_cfg = {'width': image_size, 'height': image_size} if image_size else None
    vec_env_reward_mode = 'dense'
    if 'Cabinet' in env_id:
        cam_cfg = {'width': 125, 'height': 50}
        vec_env_reward_mode = 'sparse'
    elif 'Bucket' in env_id or 'Chair' in env_id:
        cam_cfg = {'width': 125, 'height': 50}
        gym_vec_env = True
    if 'Door_unified' in env_id:
        vec_env_reward_mode = 'dense'
        gym_vec_env = True
    wrappers = [
        gym.wrappers.RecordEpisodeStatistics,
        gym.wrappers.ClipAction,
        GetStateObsWrapper,
    ]
    if gym_vec_env:
        if video_dir:
            wrappers.append(partial(RecordEpisode, output_dir=video_dir, save_trajectory=False, info_on_video=True))
        wrappers.append(MS2_RGBDObsWrapper)
        def make_single_env(_seed):
            def thunk():
                env = gym.make(env_id, reward_mode='dense', obs_mode='rgbd', control_mode=control_mode, camera_cfgs=cam_cfg)
                for wrapper in wrappers: env = wrapper(env)
                seed_env(env, _seed)
                return env
            return thunk
        # must use AsyncVectorEnv, so that the renderers will be in different processes
        envs = gym.vector.AsyncVectorEnv([make_single_env(seed + i) for i in range(num_envs)], context='forkserver')
    else:
        envs = mani_skill2.vector.make(
            env_id, num_envs, obs_mode='rgbd', reward_mode=vec_env_reward_mode, control_mode=control_mode, wrappers=wrappers, camera_cfgs=cam_cfg,
        )
        envs.is_vector_env = True
        envs = MS2_AAC_RGBDVecEnvObsWrapper(envs) # must be outside of ms2_vec_env, otherwise obs will be raw
        envs = AutoResetVecEnvWrapper(envs) # must be outside of ObsWrapper, otherwise info['terminal_obs'] will be raw obs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        seed_env(envs, seed)
    envs.is_ms1_env = 'Cabinet' in env_id or 'Bucket' in env_id or 'Chair' in env_id
    envs.use_gym_vec_env = gym_vec_env
    return envs

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)

class PlainConv(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_dim=256,
                 max_pooling=False,
                 inactivated_output=False, # False for ConvBody, True for CNN
                 image_size=128,
                 ):
        assert image_size in [64, 128]
        super().__init__()
        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [64, 64]
            nn.Conv2d(16, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 128, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(128, 128, 1, padding=0, bias=True), nn.ReLU(inplace=True),
        )
        feature_size = int((image_size / 32) ** 2)
        if max_pooling:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=not inactivated_output)
        else:
            self.pool = None
            self.fc = make_mlp(128 * feature_size, [out_dim], last_act=not inactivated_output)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class PlainConv3(PlainConv): # for 50x125 image
    def __init__(self,
                 in_channels=3,
                 out_dim=256,
                 max_pooling=False,
                 inactivated_output=False, # False for ConvBody, True for CNN
                 ):
        nn.Module.__init__(self)
        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [25, 62]
            nn.Conv2d(16, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [12, 31]
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [6, 15]
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [3, 7]
            nn.Conv2d(64, 128, 3, padding=1, bias=True), nn.ReLU(inplace=True),
        )
        if max_pooling:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=not inactivated_output)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 3 * 7, [out_dim], last_act=not inactivated_output)

        self.reset_parameters()

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space['oracle_state'].shape).prod() + np.prod(env.single_action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape)
        state_dim = envs.single_observation_space['state'].shape[0]
        image_shape = envs.single_observation_space['rgb'].shape
        if image_shape[2] == 6: # ms2 envs
            self.encoder = PlainConv(in_channels=8, out_dim=256, max_pooling=False, inactivated_output=False, image_size=image_shape[0])
        else: # ms1 envs
            self.encoder = PlainConv3(in_channels=12, out_dim=256, max_pooling=False, inactivated_output=False)
        self.mlp = make_mlp(256+state_dim, [512, 256], last_act=True)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling
        self.action_scale = torch.FloatTensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0)

    def get_feature(self, obs):
        # Preprocess the obs before passing to the real network, similar to the Dataset class in supervised learning
        rgb = obs['rgb'].float() / 255.0 # (B, H, W, 3*k)
        depth = obs['depth'].float() # (B, H, W, 1*k)
        img = torch.cat([rgb, depth], dim=3) # (B, H, W, C)
        img = img.permute(0, 3, 1, 2) # (B, C, H, W)
        feature = self.encoder(img)
        return torch.cat([feature, obs['state']], dim=1)

    def forward(self, obs):
        x = self.get_feature(obs)
        x = self.mlp(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_eval_action(self, obs):
        mean, log_std = self(obs)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, obs):
        mean, log_std = self(obs)
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
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

def to_numpy_dirty(x):
    return { # we do not change dtype here
        'rgb': x['rgb'].cpu().numpy(),
        'depth': x['depth'].cpu().numpy(),
        'state': x['state'],
        'oracle_state': x['oracle_state'],
    }

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
            print(f"global_step={global_step}, episodic_return={item['episode']['r']:.4f}, success={item['success']}")
            result['return'].append(item['episode']['r'])
            result['len'].append(item["episode"]["l"])
            result['success'].append(item['success'])
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


if __name__ == "__main__":
    args = parse_args()
    print(args, flush=True)
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
    eval_envs = make_vec_env(args.env_id, args.num_eval_envs, args.seed+1000, args.control_mode, args.image_size,
                             video_dir=f'{log_path}/videos' if args.capture_video else None, gym_vec_env=True)
    envs = make_vec_env(args.env_id, args.num_envs, args.seed, args.control_mode, args.image_size)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    print(envs.single_action_space)
    print(envs.single_observation_space)

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    if args.from_actor_ckpt is not None:
        ckpt = torch.load(args.from_actor_ckpt)
        for key in ['agent', 'actor']:
            if key in ckpt:
                actor.load_state_dict(ckpt[key])
                break
    if args.from_critic_ckpt is not None:
        ckpt = torch.load(args.from_critic_ckpt)
        qf1.load_state_dict(ckpt['qf1'])
        qf2.load_state_dict(ckpt['qf2'])
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
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
            if envs.use_gym_vec_env:
                rb.add(obs, real_next_obs, actions, rewards, dones, infos)
            else:
                rb.add(to_numpy_dirty(obs), to_numpy_dirty(real_next_obs), actions, rewards, dones, infos)

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

            # update the value networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations["oracle_state"], next_state_actions)
                qf2_next_target = qf2_target(data.next_observations["oracle_state"], next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                if args.value_always_bootstrap:
                    next_q_value = data.rewards.flatten() + args.gamma * (min_qf_next_target).view(-1)
                else:
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations["oracle_state"], data.actions).view(-1)
            qf2_a_values = qf2(data.observations["oracle_state"], data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # update the policy network
            if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                pi, log_pi, _ = actor.get_action(data.observations)
                qf1_pi = qf1(data.observations["oracle_state"], pi)
                qf2_pi = qf2(data.observations["oracle_state"], pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.observations)
                    alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        training_time += time.time() - tic

        # Log training-related data
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            if len(result['return']) > 0:
                for k, v in result.items():
                    writer.add_scalar(f"train/{k}", np.mean(v), global_step)
                result = defaultdict(list)
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/alpha", alpha, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            tot_time = time.time() - start_time
            writer.add_scalar("charts/SPS", int(global_step / (tot_time)), global_step)
            writer.add_scalar("charts/collect_time", collect_time / tot_time, global_step)
            writer.add_scalar("charts/training_time", training_time / tot_time, global_step)
            writer.add_scalar("charts/eval_time", eval_time / tot_time, global_step)
            writer.add_scalar("charts/collect_SPS", int(global_step / collect_time), global_step)
            writer.add_scalar("charts/training_SPS", int(global_step / training_time), global_step)
            if args.autotune:
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