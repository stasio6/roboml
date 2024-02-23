ALGO_NAME = 's2v-DAgger'
ENV_DOMAIN = 'DMC'
# State-to-Visual DAgger

import argparse
import os
import random
import time
from distutils.util import strtobool

os.environ["OMP_NUM_THREADS"] = "1"

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import datetime
from collections import defaultdict

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
    parser.add_argument("--wandb-project-name", type=str, default="Rookie-dev",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="walker-walk",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=3_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=300_000,
        help="the replay memory buffer size")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps-per-collect", type=int, default=2000, # TODO: yunchao, please try a very small number (like 64) to see how it works
        help="the number of steps to run in all environment in total per policy rollout")
    parser.add_argument("--minibatch-size", type=int, default=100,
        help="the size of mini-batches")
    parser.add_argument("--num-steps-per-update", type=float, default=1, # should be tuned based on sim time and training time
        help="the ratio between env steps and num of gradient updates, lower means more updates")
    parser.add_argument("--bc-loss-th", type=float, default=0.01, # important for training time
        help="if the bc loss is smaller than this threshold, then stop training and collect new data")
    parser.add_argument("--learning-starts", type=int, default=4000, # TODO: maybe also tune this to see if it is important?
        help="timestep to start learning")

    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=5)
    parser.add_argument("--sync-venv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--log-freq", type=int, default=4000)
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--expert-ckpt", type=str, default='output/DMC/walker-walk/SAC/230414-234951_1_0.5x-update/checkpoints/600000.pt')

    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.env_domain = ENV_DOMAIN
    args.script = __file__
    assert args.num_eval_envs == 1 or not args.capture_video, "Cannot capture video with multiple eval envs."
    assert args.num_steps_per_collect % args.num_envs == 0
    args.num_steps = int(args.num_steps_per_collect // args.num_envs)
    args.num_updates_per_collect = int(args.num_steps_per_collect / args.num_steps_per_update)
    args.num_eval_envs = min(args.num_eval_envs, args.num_eval_episodes)
    assert args.num_eval_episodes % args.num_eval_envs == 0
    # assert args.env_id in args.expert_ckpt, 'Expert checkpoint should be trained on the same env'
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


class Agent(nn.Module):
    def __init__(self, envs, visual_feature_dim=256, mlp_cfg=[1024, 1024], num_frames=3):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape)
        self.encoder = DrQConv(in_channel=3 * num_frames, feature_dim=visual_feature_dim)
        self.mlp = make_mlp(visual_feature_dim, mlp_cfg, last_act=True)
        self.fc_mean = nn.Linear(mlp_cfg[-1], action_dim)
        # action rescaling
        self.action_scale = torch.FloatTensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0)
        self.forward = self.get_action = self.get_eval_action

    def get_feature(self, obs):
        visual_feature = self.encoder(obs)
        return self.mlp(visual_feature)

    def get_eval_action(self, obs):
        x = self.get_feature(obs)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action


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
                    dtype = torch.uint8 if v.dtype == np.uint8 else torch.float32
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype).to(device)

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
            print(f"global_step={global_step}, episodic_return={item['episode']['r']:.4f}")
            result['return'].append(item['episode']['r'])
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
    log_name = os.path.join(ENV_DOMAIN, args.env_id, ALGO_NAME, tag)
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
    VecEnv = gym.vector.SyncVectorEnv if args.sync_venv else lambda x: gym.vector.AsyncVectorEnv(x, context='forkserver')
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
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # expert setup
    from os.path import dirname as up
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
    assert (expert.action_scale == agent.action_scale).all()
    assert (expert.action_bias == agent.action_bias).all()

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
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    global_step = 0
    next_obs = to_tensor(envs.reset(), device)
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
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action = agent.get_action(next_obs)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            next_obs = to_tensor(next_obs, device)
            next_done = torch.Tensor(done).to(device)

            result = collect_episode_info(info, result)
        collect_time += time.time() - tic

        # flatten the batch
        b_obs = obs.reshape((-1,))
        # DAgger: save data to replay buffer
        b_expert_actions = expert.get_eval_action(b_obs['state']).detach()
        dagger_buf.add(b_obs, b_expert_actions)

        # Optimizing the policy and value network
        if global_step < args.learning_starts:
            continue
        tic = time.time()
        agent.train()
        for i_update in range(args.num_updates_per_collect):
            # Behavior Cloning
            data = dagger_buf.sample(args.minibatch_size)
            pred_actions = agent.get_action(to_tensor(data['observations'], device))
            loss = F.mse_loss(pred_actions, data['expert_actions'].to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _loss = loss.item()

            print('i_update:', i_update,'loss:', _loss)
            if args.bc_loss_th is not None and _loss < args.bc_loss_th:
                break

        training_time += time.time() - tic

        # Log
        if (global_step - args.num_steps_per_collect) // args.log_freq < global_step // args.log_freq:
            if len(result['return']) > 0:
                for k, v in result.items():
                    writer.add_scalar(f"train/{k}", np.mean(v), global_step)
                result = defaultdict(list)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/bc_loss", _loss, global_step)
            writer.add_scalar("losses/num_updates", i_update + 1, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            tot_time = time.time() - start_time
            writer.add_scalar("charts/SPS", int(global_step / tot_time), global_step)
            writer.add_scalar("charts/collect_time", collect_time / tot_time, global_step)
            writer.add_scalar("charts/training_time", training_time / tot_time, global_step)
            writer.add_scalar("charts/eval_time", eval_time / tot_time, global_step)
            writer.add_scalar("charts/collect_SPS", int(global_step / collect_time), global_step)
            writer.add_scalar("charts/training_SPS", int(global_step / training_time), global_step)

        # Evaluation
        if (global_step - args.num_steps_per_collect) // args.eval_freq < global_step // args.eval_freq:
            tic = time.time()
            result = evaluate(args.num_eval_episodes, agent, eval_envs, device)
            eval_time += time.time() - tic
            for k, v in result.items():
                writer.add_scalar(f"eval/{k}", np.mean(v), global_step)
        
        # Checkpoint
        if args.save_freq and ( update == num_updates or \
                (global_step - args.num_steps_per_collect) // args.save_freq < global_step // args.save_freq):
            os.makedirs(f'{log_path}/checkpoints', exist_ok=True)
            torch.save({
                'agent': agent.state_dict(),
            }, f'{log_path}/checkpoints/{global_step}.pt')

    envs.close()
    writer.close()