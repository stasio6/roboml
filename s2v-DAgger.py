ALGO_NAME = 's2v-DAgger'
# State-to-Visual DAgger

import argparse
import os
import random
import time
from distutils.util import strtobool

os.environ["OMP_NUM_THREADS"] = "1"

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    parser.add_argument("--wandb-project-name", type=str, default="s2v-dagger",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="PickCube-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=60_000,
        help="the replay memory buffer size")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps-per-collect", type=int, default=8000, # this hp is pretty important
        help="the number of steps to run in all environment in total per policy rollout")
    parser.add_argument("--minibatch-size", type=int, default=50,
        help="the size of mini-batches")
    parser.add_argument("--num-steps-per-update", type=float, default=2, # should be tuned based on sim time and training time
        help="the ratio between env steps and num of gradient updates, lower means more updates")
    parser.add_argument("--bc-loss-th", type=float, default=0.01, # important for training time
        help="if the bc loss is smaller than this threshold, then stop training and collect new data")

    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--log-freq", type=int, default=4000)
    parser.add_argument("--save-freq", type=int, default=500_000)
    parser.add_argument("--control-mode", type=str, default='pd_ee_delta_pos')
    parser.add_argument("--expert-ckpt", type=str, default='output/PickCube-v1/SAC-ms2-new/230329-142137_1_profile/checkpoints/600000.pt')
    parser.add_argument("--image-size", type=int, default=None,
        help="the size of observation image, e.g. 64 means 64x64")


    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.script = __file__
    assert args.num_eval_envs == 1 or not args.capture_video, "Cannot capture video with multiple eval envs."
    assert args.num_steps_per_collect % args.num_envs == 0
    args.num_steps = int(args.num_steps_per_collect // args.num_envs)
    assert args.num_steps_per_collect % args.minibatch_size == 0
    args.num_minibatches = int(args.num_steps_per_collect // args.minibatch_size)
    args.num_updates_per_collect = int(args.num_steps_per_collect / args.num_steps_per_update)
    assert args.num_updates_per_collect % args.num_minibatches == 0
    args.update_epochs = int(args.num_updates_per_collect // args.num_minibatches)
    args.num_eval_envs = min(args.num_eval_envs, args.num_eval_episodes)
    assert args.num_eval_episodes % args.num_eval_envs == 0
    # assert args.env_id in args.expert_ckpt, 'Expert checkpoint should be trained on the same env'
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
    def build_obs_space(env):
        obs_space = env.observation_space
        state_dim = 0
        for k in ['agent', 'extra']:
            state_dim += sum([v.shape[0] for v in flatten_dict_space_keys(obs_space[k]).spaces.values()])

        h, w, _ = env.observation_space['image']['hand_camera']['rgb'].shape
        k = len(env.observation_space['image'])

        return spaces.Dict({
            'state': spaces.Box(-float("inf"), float("inf"), shape=(state_dim,), dtype=np.float32),
            'image': spaces.Dict({
                'rgb': spaces.Box(0, 255, shape=(h,w,k*3), dtype=np.uint8),
                'depth': spaces.Box(-float("inf"), float("inf"), shape=(h,w,k), dtype=np.float16),
            })
        })
    
    @staticmethod
    def convert_obs(obs, concat_fn):
        img_dict = obs['image']
        new_imgage_dict = {
            key: concat_fn([v[key] for v in img_dict.values()])
            for key in ['rgb', 'depth']
        }
        depth = new_imgage_dict['depth']
        if not isinstance(depth, np.ndarray):
            new_imgage_dict['depth'] = depth.to(torch.float16)
        else:
            new_imgage_dict['depth'] = depth.astype(np.float16)

        state = np.hstack([
            flatten_state_dict(obs["agent"]),
            flatten_state_dict(obs["extra"]),
        ])

        out_dict = {
            'image': new_imgage_dict,
            'state': state,
        }
        return out_dict

class MS2_RGBDObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert self.obs_mode == 'rgbd'
        self.observation_space = MS2_RGBDVecEnvObsWrapper.build_obs_space(env)
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

class GetStateObsWrapper(Wrapper):
    def get_state_mode_obs(self):
        raw_env = self.env.unwrapped
        original_obs_mode = raw_env._obs_mode
        raw_env._obs_mode = 'state'
        state_dict = self.env.unwrapped._get_obs_state_dict()
        state_obs = flatten_state_dict(state_dict)
        raw_env._obs_mode = original_obs_mode
        return state_obs

def make_vec_env(env_id, num_envs, seed, control_mode=None, image_size=None, video_dir=None):
    cam_cfg = {'width': image_size, 'height': image_size} if image_size else None
    wrappers = [
        gym.wrappers.RecordEpisodeStatistics,
        gym.wrappers.ClipAction,
    ]
    if video_dir:
        assert num_envs == 1, "Cannot capture video with multiple envs."
        wrappers += [
            partial(RecordEpisode, output_dir=video_dir, save_trajectory=False, info_on_video=True),
            MS2_RGBDObsWrapper,
        ]
        # RecordEpisode is not compatible with MS2 vec env, so we have to use the normal env
        def make_env_fn():
            env = gym.make(env_id, reward_mode='dense', obs_mode='rgbd', control_mode=control_mode, camera_cfgs=cam_cfg)
            for wrapper in wrappers: env = wrapper(env)
            return env
        envs = gym.vector.SyncVectorEnv([make_env_fn])
    else:
        wrappers.append(GetStateObsWrapper)
        envs = mani_skill2.vector.make(
            env_id, num_envs, obs_mode='rgbd', reward_mode='dense', control_mode=control_mode, wrappers=wrappers, camera_cfgs=cam_cfg,
        )
        envs.is_vector_env = True
        envs = MS2_RGBDVecEnvObsWrapper(envs) # must be outside of ms2_vec_env, otherwise obs will be raw
        envs = AutoResetVecEnvWrapper(envs) # must be outside of ObsWrapper, otherwise info['terminal_obs'] will be raw obs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space

    envs.seed(seed)
    envs.action_space.seed(seed)
    envs.observation_space.seed(seed)
    envs.single_action_space.seed(seed)
    envs.single_observation_space.seed(seed)

    return envs

def get_state_obs(envs):
    return np.vstack(envs.env_method('get_state_mode_obs'))


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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
                 ):
        super().__init__()

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

        if max_pooling:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=not inactivated_output)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4, [out_dim], last_act=not inactivated_output)

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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape)
        state_dim = envs.single_observation_space['state'].shape[0]
        self.encoder = PlainConv(in_channels=8, out_dim=256, max_pooling=False, inactivated_output=False)
        self.actor = make_mlp(256+state_dim, [512, 256, action_dim], last_act=False)
        self.get_eval_action = self.get_action = self.forward

    def get_feature(self, obs):
        # Preprocess the obs before passing to the real network, similar to the Dataset class in supervised learning
        rgb = obs['image']['rgb'].float() / 255.0 # (B, H, W, 3*k)
        depth = obs['image']['depth'].float() # (B, H, W, 1*k)
        img = torch.cat([rgb, depth], dim=3) # (B, H, W, C)
        img = img.permute(0, 3, 1, 2) # (B, C, H, W)
        feature = self.encoder(img)
        return torch.cat([feature, obs['state']], dim=1)

    def forward(self, obs):
        x = self.get_feature(obs)
        return self.actor(x)

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
            return
        if isinstance(value, dict):
            for k, v in value.items():
                self.data[k][index] = v
        else:
            for k, v in value.data.items():
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
    
    # def to(self, device):
    #     return DictArray(self.buffer_shape, None, device=device, data_dict={
    #         k: v.to(device) for k, v in self.data.items()
    #     })


class DAggerBuffer(object):
    def __init__(self, buffer_size, collect_size, obs_space, action_space, device='cpu'):
        self.buffer_size = max(buffer_size // collect_size, 1)
        self.collect_size = collect_size
        self.observations = DictArray(buffer_shape=(self.buffer_size, collect_size), element_space=obs_space, device=device)
        self.expert_actions = torch.zeros((self.buffer_size, collect_size, int(np.prod(action_space.shape))), 
                                          dtype=torch.float32, device=device)
        self.device = device
        self.pos = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.buffer_size if self.full else self.pos

    def add(self, obs: DictArray, expert_actions):
        self.observations[self.pos] = obs # TODO: check if we need to copy the obs
        self.expert_actions[self.pos] = expert_actions.clone().detach()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        if self.full:
            batch_inds = (torch.randint(1, self.buffer_size, size=(batch_size,)) + self.pos) % self.buffer_size
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size,))
        # Sample randomly the env idx
        env_indices = torch.randint(0, high=self.collect_size, size=(len(batch_inds),))

        return dict(
            observations=self.observations[batch_inds, env_indices, :],
            expert_actions=self.expert_actions[batch_inds, env_indices, :],
        )


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
    tmp_env = gym.make(args.env_id)
    if tmp_env.spec.max_episode_steps > args.num_steps:
        print("\033[93mWARN: num_steps is less than max episode length. "
            "Consider raise num_steps_per_collect or lower num_envs. Continue?\033[0m")
        aaa = input()
    del tmp_env
    envs = make_vec_env(args.env_id, args.num_envs, args.seed, args.control_mode, args.image_size)
    # if args.rew_norm:
    #     envs = gym.wrappers.NormalizeReward(envs, args.gamma)
    # eval_envs = make_vec_env(args.env_id, args.num_eval_envs, args.seed+1000, args.control_mode, args.image_size,
    #                          video_dir=f'{log_path}/videos' if args.capture_video else None)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    print(envs.single_action_space)
    print(envs.single_observation_space)

    # agent setup
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # expert setup
    from os.path import dirname as up
    args.expert_ckpt = 'checkpoints/' + args.env_id + '/checkpoints/' + args.env_id + ".pt"
    expert_dir = up(up(args.expert_ckpt))
    import json
    with open(f'{expert_dir}/args.json', 'r') as f:
        expert_args = json.load(f)
    m = import_file_as_module(expert_args['script'])

    class DummyObject: pass
    dummy_env = DummyObject()
    from mani_skill2.utils.common import convert_observation_to_space
    example_expert_obs = envs.env_method('get_state_mode_obs')[0]
    expert_obs_space = convert_observation_to_space(example_expert_obs)
    dummy_env.single_observation_space = expert_obs_space
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

    # DAgger buffer setup
    envs.single_observation_space.dtype = np.float32
    dagger_buf = DAggerBuffer(
        args.buffer_size,
        args.num_steps_per_collect,
        envs.single_observation_space,
        envs.single_action_space,
        device='cpu',
    )

    # ALGO Logic: Storage setup
    # each obs is like {'image': {'rgb': (B,H,W,6), 'depth': (B,H,W,2)}, 'state': (B,D)}
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device)
    expert_obs = torch.zeros((args.num_steps, args.num_envs) + expert_obs_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    global_step = 0
    next_obs = to_tensor(envs.reset(), device)
    next_expert_obs = torch.Tensor(get_state_obs(envs)) # no need to move to GPU
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = int(np.ceil(args.total_timesteps / args.num_steps_per_collect))
    result = defaultdict(list)
    collect_time = training_time = eval_time = 0

    for update in range(1, num_updates + 1):
        print('== Epoch:', update)

        tic = time.time()
        agent.eval()
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            expert_obs[step] = next_expert_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action = agent.get_action(next_obs)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            next_obs = to_tensor(next_obs, device)
            next_expert_obs = torch.Tensor(get_state_obs(envs)) # no need to move to GPU
            next_done = torch.Tensor(done).to(device)

            result = collect_episode_info(info, result)
        collect_time += time.time() - tic

        tic = time.time()

        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_expert_obs = expert_obs.reshape((-1,) + expert_obs_space.shape)
        # DAgger: save data to replay buffer
        b_expert_actions = expert.get_eval_action(b_expert_obs).detach()
        dagger_buf.add(b_obs, b_expert_actions)

        # Optimizing the policy and value network
        agent.train()
        # b_inds = np.arange(args.num_steps_per_collect)
        for epoch in range(args.update_epochs):
            mean_loss = 0.0
            # np.random.shuffle(b_inds)
            for start in range(0, args.num_steps_per_collect, args.minibatch_size):
                # end = start + args.minibatch_size
                # mb_inds = b_inds[start:end]

                # Behavior Cloning
                data = dagger_buf.sample(args.minibatch_size)
                pred_actions = agent.get_action(to_tensor(data['observations'], device))
                loss = F.mse_loss(pred_actions, data['expert_actions'].to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_loss += loss.item()
            mean_loss /= (args.num_steps_per_collect // args.minibatch_size)
            print('epoch:', epoch, 'loss:', mean_loss)
            if args.bc_loss_th is not None and mean_loss < args.bc_loss_th:
                break

        training_time += time.time() - tic

        # Log
        if (global_step - args.num_steps_per_collect) // args.log_freq < global_step // args.log_freq:
            if len(result['return']) > 0:
                for k, v in result.items():
                    writer.add_scalar(f"train/{k}", np.mean(v), global_step)
                result = defaultdict(list)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/bc_loss", mean_loss, global_step)
            writer.add_scalar("losses/update_epochs", epoch + 1, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            tot_time = time.time() - start_time
            writer.add_scalar("charts/SPS", int(global_step / tot_time), global_step)
            writer.add_scalar("charts/collect_time", collect_time / tot_time, global_step)
            writer.add_scalar("charts/training_time", training_time / tot_time, global_step)
            writer.add_scalar("charts/eval_time", eval_time / tot_time, global_step)
            writer.add_scalar("charts/collect_SPS", int(global_step / collect_time), global_step)
            writer.add_scalar("charts/training_SPS", int(global_step / training_time), global_step)

        # Evaluation
        # if (global_step - args.num_steps_per_collect) // args.eval_freq < global_step // args.eval_freq:
        #     tic = time.time()
        #     result = evaluate(args.num_eval_episodes, agent, eval_envs, device)
        #     eval_time += time.time() - tic
        #     for k, v in result.items():
        #         writer.add_scalar(f"eval/{k}", np.mean(v), global_step)
        
        # Checkpoint
        if args.save_freq and ( update == num_updates or \
                (global_step - args.num_steps_per_collect) // args.save_freq < global_step // args.save_freq):
            os.makedirs(f'{log_path}/checkpoints', exist_ok=True)
            torch.save({
                'agent': agent.state_dict(),
            }, f'{log_path}/checkpoints/{global_step}.pt')

    envs.close()
    writer.close()