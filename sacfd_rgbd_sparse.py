ALGO_NAME = 'SACfd-ms2-RGBD-encoder'

import os
import argparse
import random
import time
from distutils.util import strtobool

os.environ["OMP_NUM_THREADS"] = "1"

import gym
import numpy as np
# import pybullet_envs  # noqa
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
    parser.add_argument("--demo-path", type=str, default='output/PickCube-v1/SAC-ms2-new/230329-142137_1_profile/evaluation/600000/PickCube-v1_trajectories_1000.rgbd.pd_ee_delta_pos.h5',
        help="the path of demo H5 file or pkl file")
    parser.add_argument("--num-demo-traj", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=300_000,
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
    parser.add_argument("--num-steps-per-update", type=float, default=4)
    parser.add_argument("--training-freq", type=int, default=64)
    parser.add_argument("--log-freq", type=int, default=2000)
    parser.add_argument("--save-freq", type=int, default=500_000)
    parser.add_argument("--value-always-bootstrap", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="in ManiSkill variable episode length setting, set to True if positive reawrd, False if negative reward.")
    parser.add_argument("--control-mode", type=str, default='pd_ee_delta_pos')
    parser.add_argument("--from-ckpt", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=64,
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
    # fmt: on
    return args

import mani_skill2.envs
from mani_skill2.utils.common import flatten_state_dict, flatten_dict_space_keys
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.vector.vec_env import VecEnvObservationWrapper
from gym.core import Wrapper, ObservationWrapper
from gym import spaces

class MS2_RGBDVecEnvObsWrapper(VecEnvObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert self.obs_mode == 'rgbd'
        self.observation_space = self.build_obs_space(env)
        self.concat_fn = partial(torch.cat, dim=-1)

    def observation(self, obs):
        return self.convert_obs(obs, self.concat_fn)
    
    @staticmethod
    def build_obs_space(env, depth_dtype=np.float16):
        obs_space = env.observation_space
        state_dim = 0
        for k in ['agent', 'extra']:
            state_dim += sum([v.shape[0] for v in flatten_dict_space_keys(obs_space[k]).spaces.values()])

        h, w, _ = env.observation_space['image']['hand_camera']['rgb'].shape
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

        state = np.hstack([
            flatten_state_dict(obs["agent"]),
            flatten_state_dict(obs["extra"]),
        ])
        new_img_dict['state'] = state

        return new_img_dict
    
class MS2_RGBDObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert self.obs_mode == 'rgbd'
        self.observation_space = MS2_RGBDVecEnvObsWrapper.build_obs_space(env, depth_dtype=np.float32)
        self.concat_fn = partial(np.concatenate, axis=-1)

    def observation(self, obs):
        return MS2_RGBDVecEnvObsWrapper.convert_obs(obs, self.concat_fn)
    
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
    
def seed_env(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

def make_vec_env(env_id, num_envs, seed, control_mode=None, image_size=None, video_dir=None, gym_vec_env=False):
    assert gym_vec_env or video_dir is None, 'Saving video is only supported for gym vec env'
    cam_cfg = {'width': image_size, 'height': image_size} if image_size else None
    wrappers = [
        gym.wrappers.RecordEpisodeStatistics,
        gym.wrappers.ClipAction,
    ]
    if gym_vec_env:
        if video_dir:
            wrappers.append(partial(RecordEpisode, output_dir=video_dir, save_trajectory=False, info_on_video=True))
        wrappers.append(MS2_RGBDObsWrapper)
        def make_single_env(_seed):
            def thunk():
                env = gym.make(env_id, reward_mode='sparse', obs_mode='rgbd', control_mode=control_mode, camera_cfgs=cam_cfg)
                for wrapper in wrappers: env = wrapper(env)
                seed_env(env, _seed)
                return env
            return thunk
        # must use AsyncVectorEnv, so that the renderers will be in different processes
        envs = gym.vector.AsyncVectorEnv([make_single_env(seed + i) for i in range(num_envs)], context='forkserver')
    else:
        envs = mani_skill2.vector.make(
            env_id, num_envs, obs_mode='rgbd', reward_mode='sparse', control_mode=control_mode, wrappers=wrappers, camera_cfgs=cam_cfg,
        )
        envs.is_vector_env = True
        envs = MS2_RGBDVecEnvObsWrapper(envs) # must be outside of ms2_vec_env, otherwise obs will be raw
        envs = AutoResetVecEnvWrapper(envs) # must be outside of ObsWrapper, otherwise info['terminal_obs'] will be raw obs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        seed_env(envs, seed)

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

def make_mlp_with_layer_norm(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(nn.LayerNorm(c_out))
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)

class PlainConv(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_dim=256,
                 max_pooling=True,
                 inactivated_output=False, # False for ConvBody, True for CNN
                 image_size=128,
                 ):
        assert image_size in [64, 128]
        super().__init__()
        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True), nn.LayerNorm((64, 64)), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 16, 3, padding=1, bias=True), nn.LayerNorm((32,32)), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.LayerNorm((16,16)), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.LayerNorm((8,8)), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(64, 128, 3, padding=1, bias=True), nn.LayerNorm((4,4)), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [2, 2]
            nn.Conv2d(128, 128, 1, padding=0, bias=True), nn.LayerNorm((2,2)), nn.ReLU(inplace=True),
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

    def forward(self, obs):
        # Preprocess the obs before passing to the real network, similar to the Dataset class in supervised learning
        rgb = obs['rgb'].float() / 255.0 # (B, H, W, 3*k)
        depth = obs['depth'].float() # (B, H, W, 1*k)
        img = torch.cat([rgb, depth], dim=3) # (B, H, W, C)
        img = img.permute(0, 3, 1, 2) # (B, C, H, W)
        x = self.cnn(img)
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
    def __init__(self, envs, encoder):
        super().__init__()
        self.encoder = encoder
        action_dim = np.prod(envs.single_action_space.shape)
        state_dim = envs.single_observation_space['state'].shape[0]
        self.mlp = make_mlp_with_layer_norm(encoder.out_dim+action_dim+state_dim, [512, 256, 1], last_act=False)

    def forward(self, obs, action, visual_feature=None, detach_encoder=False):
        if visual_feature is None:
            visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        x = torch.cat([visual_feature, obs["state"], action], dim=1)
        return self.mlp(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, envs, visual_feature_dim=256):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape)
        state_dim = envs.single_observation_space['state'].shape[0]
        image_size = envs.single_observation_space['rgb'].shape[0]
        image_shape = envs.single_observation_space['rgb'].shape
        if image_shape[2] == 6: # ms2 envs
            self.encoder = PlainConv(in_channels=8, out_dim=visual_feature_dim, max_pooling=False, inactivated_output=False, image_size=image_size)
        else: # ms1 envs
            self.encoder = PlainConv3(in_channels=12, out_dim=256, max_pooling=False, inactivated_output=False)
        
        self.mlp = make_mlp(visual_feature_dim+state_dim, [512, 256], last_act=True)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling
        self.action_scale = torch.FloatTensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0)

    def get_feature(self, obs, detach_encoder=False):
        visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        x = torch.cat([visual_feature, obs['state']], dim=1)
        return self.mlp(x), visual_feature

    def forward(self, obs, detach_encoder=False):
        x, visual_feature = self.get_feature(obs, detach_encoder)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, visual_feature

    def get_eval_action(self, obs):
        mean, log_std, _ = self(obs)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, obs, detach_encoder=False):
        mean, log_std, visual_feature = self(obs, detach_encoder)
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
        return action, log_prob, mean, visual_feature

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)
    
class DictArray(object):
    def __init__(self, buffer_shape, element_space, device, data_dict=None):
        self.buffer_shape = buffer_shape
        self.device = device
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device)
                else:
                    self.data[k] = torch.zeros(buffer_shape + v.shape).to(device)

    def keys(self):
        return self.data.keys()
    
    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v
    
    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, device=self.device, data_dict=new_dict)
    
def get_element(x, index):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray) or isinstance(x, list):
        return x[index]
    elif isinstance(x, dict):
        return {k: get_element(v, index) for k, v in x.items()}
    else:
        raise NotImplementedError()
    

def to_numpy_dirty(x):
    return { # we do not change dtype here
        'rgb': x['rgb'].cpu().numpy(),
        'depth': x['depth'].cpu().numpy(),
        'state': x['state'],
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

class SmallDemoDataset_RGBD(object): # load everything into memory
    def __init__(self, envs, data_path, process_fn, obs_space, device, buffer_size, num_envs, num_traj=None):
        if data_path[-4:] == '.pkl':
            raise NotImplementedError()
        else:
            from utils.ms2_data import load_demo_dataset
            demo_dataset = load_demo_dataset(data_path, keys=['observations', 'actions', 'rewards'], num_traj=num_traj)

            self.demo_size = demo_dataset['actions'].shape[0]
            obs_buffer = DictArray(buffer_shape=(self.demo_size,), element_space=obs_space, device='cpu')
            obs_buffer_next = DictArray(buffer_shape=(self.demo_size,), element_space=obs_space, device='cpu')
            obs_cnt = 0
            obs_index = 0
            print(len(demo_dataset['observations']))
            for obs_traj in demo_dataset['observations']:
                _obs_traj = process_fn(obs_traj)
                _obs_traj['depth'] = torch.Tensor(_obs_traj['depth'].astype(np.float32) / 1024).to(torch.float16)
                len_traj = _obs_traj['state'].shape[0]
                obs_cnt += len_traj - 1
                for i in range(len_traj - 1): # remove the final observation
                    _obs = get_element(_obs_traj, i)
                    _obs_next = get_element(_obs_traj, i+1)
                    _obs = to_tensor(_obs, device)
                    _obs_next = to_tensor(_obs_next, device)
                    # obs_buffer[i] = _obs
                    obs_buffer[obs_index] = _obs
                    obs_buffer_next[obs_index] = _obs_next
                    obs_index+=1
            assert obs_cnt == self.demo_size
            
        self.device = device

        self.demo_data = {
            'actions': torch.tensor(demo_dataset['actions']).to(device),
            'observations': obs_buffer,
            'next_observations': obs_buffer_next, 
            'rewards': torch.tensor(demo_dataset['rewards']).to(device)
        }
        
        self.num_envs = num_envs
        
        self.collect_data = DictReplayBuffer(
            buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            num_envs,
            handle_timeout_termination=True,
        )

    def __len__(self):
        return self.demo_size
    
    def sample(self, batch_size):
        total_sizes = self.demo_size + self.collect_data.size()*self.num_envs # TODO: Turn back into symmetric sampling
        # n_samples_demo = int(self.demo_size/total_sizes*batch_size)
        # n_samples_collect = batch_size - n_samples_demo
        
        n_samples_demo = int(batch_size/2)
        n_samples_collect = batch_size - n_samples_demo

        # print("demo size:", self.demo_size)
        # print("total size:", total_sizes)
        # print("collect size:", self.collect_data.size()*self.num_envs)
        # print("demo samples:", n_samples_demo)
        # print("collect samples:", n_samples_collect)
        
        idxs = np.random.randint(0, self.demo_size, size=n_samples_demo)
        demo_batch = dict(
            observations=self.demo_data['observations'][idxs],
            next_observations=self.demo_data['next_observations'][idxs],
            actions=self.demo_data['actions'][idxs],
            rewards=self.demo_data['rewards'][idxs]
        )
        
        collect_batch = self.collect_data.sample(n_samples_collect)
        
        total_observations = dict(
            rgb=torch.cat([demo_batch['observations']['rgb'], to_tensor(collect_batch.observations['rgb'], 'cpu')], dim=0),
            depth=torch.cat([demo_batch['observations']['depth'], to_tensor(collect_batch.observations['depth'], 'cpu')], dim=0),
            state=torch.cat([demo_batch['observations']['state'], to_tensor(collect_batch.observations['state'], 'cpu')], dim=0)
        )
        
        total_next_observations = dict(
            rgb=torch.cat([demo_batch['next_observations']['rgb'], to_tensor(collect_batch.next_observations['rgb'], 'cpu')], dim=0),
            depth=torch.cat([demo_batch['next_observations']['depth'], to_tensor(collect_batch.next_observations['depth'], 'cpu')], dim=0),
            state=torch.cat([demo_batch['next_observations']['state'], to_tensor(collect_batch.next_observations['state'], 'cpu')], dim=0)
        )
        
        batch = dict(
            observations=total_observations,
            next_observations=total_next_observations,
            actions=torch.cat([demo_batch['actions'], to_tensor(collect_batch.actions, 'cpu')], dim=0),
            rewards=torch.cat([demo_batch['rewards'].unsqueeze(1), to_tensor(collect_batch.rewards, 'cpu')], dim=0)
        )
        return batch
    
    def add(self, obs, next_obs, actions, rewards, dones, infos):
        self.collect_data.add(obs, next_obs, actions, rewards, dones, infos)

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
    
    # from utils.ms2_data import load_demo_dataset_separate, load_demo_dataset
    
    
    # demo_dataset = load_demo_dataset_separate(args.demo_path, keys=['observations', 'actions', 'rewards'])
    
    # print("demo dict:", demo_dataset.keys())
    # print("demo image:", MS2_RGBDVecEnvObsWrapper.convert_obs(demo_dataset['observations'][0], partial(np.concatenate, axis=-1))['rgb'].shape)
    # # print("get element:", get_element(demo_dataset['observations'][0], 0)['image'].keys())
    # print("actions:", len(demo_dataset['actions']), demo_dataset['actions'][0].shape)
    # print("rewards:", len(demo_dataset['rewards']), demo_dataset['rewards'][0].shape)
    # obj_traj = MS2_RGBDVecEnvObsWrapper.convert_obs(demo_dataset['observations'][0], partial(np.concatenate, axis=-1))
    
    # len_traj = obj_traj['state'].shape[0]
    
    # print("len_traj:", len_traj)
    
    # for i in range(len_traj-1):
    #     _obs = get_element(obj_traj, i)
    #     print("obs shape:", _obs.keys())
    #     print("obs rgb:", _obs['rgb'].shape)
    #     assert False
    
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs, actor.encoder).to(device)
    qf2 = SoftQNetwork(envs, actor.encoder).to(device)
    qf1_target = SoftQNetwork(envs, actor.encoder).to(device)
    qf2_target = SoftQNetwork(envs, actor.encoder).to(device)
    if args.from_ckpt is not None:
        ckpt = torch.load(args.from_ckpt)
        actor.load_state_dict(ckpt['actor'])
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
        
    # demo dataset setup
    obs_process_fn = partial(MS2_RGBDVecEnvObsWrapper.convert_obs, concat_fn=partial(np.concatenate, axis=-1))
    dataset = SmallDemoDataset_RGBD(envs, args.demo_path, obs_process_fn,
                                        envs.single_observation_space, 'cpu', args.buffer_size, args.num_envs, args.num_demo_traj) # don't put it in GPU!
    
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
                actions, _, _, _ = actor.get_action(to_tensor(obs, device))
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
            dataset.add(to_numpy_dirty(obs), to_numpy_dirty(real_next_obs), actions, rewards, dones, infos)

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
            data = dataset.sample(args.batch_size)
            
            data = to_tensor(data, device)

            # update the value networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _, visual_feature = actor.get_action(data['next_observations'])
                qf1_next_target = qf1_target(data['next_observations'], next_state_actions, visual_feature)
                qf2_next_target = qf2_target(data['next_observations'], next_state_actions, visual_feature)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                if args.value_always_bootstrap:
                    next_q_value = data['rewards'].flatten() + args.gamma * (min_qf_next_target).view(-1)
                else:
                    next_q_value = data['rewards'].flatten() + (1 - data['dones'].flatten()) * args.gamma * (min_qf_next_target).view(-1)

            visual_feature = actor.encoder(data['observations'])
            qf1_a_values = qf1(data['observations'], data['actions'], visual_feature).view(-1)
            qf2_a_values = qf2(data['observations'], data['actions'], visual_feature).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # update the policy network
            if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                pi, log_pi, _, visual_feature = actor.get_action(data['observations'], detach_encoder=True)
                qf1_pi = qf1(data['observations'], pi, visual_feature, detach_encoder=True)
                qf2_pi = qf2(data['observations'], pi, visual_feature, detach_encoder=True)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _, _ = actor.get_action(data['observations'])
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