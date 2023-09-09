ALGO_NAME = 'BC_state'

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
# import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import datetime
from collections import defaultdict

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from utils.sampler import IterationBasedBatchSampler
from utils.torch_utils import worker_init_fn


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
    parser.add_argument("--env-id", type=str, default="PickCube-v1",
        help="the id of the environment")
    parser.add_argument("--demo-path", type=str, default='checkpoints/PickSingleYCB-v1/evaluation/PickSingleYCB-v1/PickSingleYCB-v1_trajectories_100.pkl',
        help="the path of demo pkl")
    parser.add_argument("--num-demo-traj", type=int, default=100)
    parser.add_argument("--total-iters", type=int, default=100_000,
        help="total timesteps of the experiments")
    parser.add_argument("--lr", type=float, default=3e-4,
        help="the learning rate of the agent")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the replay memory")

    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--log-freq", type=int, default=2000)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=20000)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    parser.add_argument("--num-eval-envs", type=int, default=4)
    parser.add_argument("--sync-venv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--num-dataload-workers", type=int, default=0)
    parser.add_argument("--control-mode", type=str, default='pd_ee_delta_pos')

    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.script = __file__
    args.num_eval_envs = min(args.num_eval_envs, args.num_eval_episodes)
    assert args.num_eval_episodes % args.num_eval_envs == 0
    if args.num_eval_envs == 1:
        args.sync_venv = True
    # fmt: on
    return args


import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode

def make_env(env_id, seed, control_mode=None, video_dir=None):
    def thunk():
        env = gym.make(env_id, reward_mode='dense', obs_mode='state', control_mode=control_mode)
        if video_dir:
            env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=True)
            # env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

class SmallDemoDataset(Dataset):
    def __init__(self, data_path, device, num_traj):
        if data_path[-4:] == '.pkl':
            from utils.data import load_demo_dataset
            demo_dataset = load_demo_dataset(data_path, num_traj=num_traj)
        else:
            from utils.ms2_data import load_demo_dataset
            demo_dataset = load_demo_dataset(data_path, num_traj=num_traj)

        for k, v in demo_dataset.items():
            demo_dataset[k] = torch.Tensor(v).to(device)
        self.size = demo_dataset['actions'].shape[0]
        self.dataset = demo_dataset
        assert len(self.dataset['actions']) == len(self.dataset['observations'])

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.dataset.items()}

    def __len__(self):
        return self.size


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(env.single_action_space.shape))
        )
        self.get_eval_action = self.get_action = self.forward

    def forward(self, x):
        return self.net(x)

def collect_episode_info(info, result=None):
    if result is None:
        result = defaultdict(list)
    for item in info:
        if "episode" in item.keys():
            print(f"episodic_return={item['episode']['r']:.4f}, success={item['success']}")
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
            action = agent.get_eval_action(torch.Tensor(obs).to(device))
        obs, rew, done, info = eval_envs.step(action.cpu().numpy())
        collect_episode_info(info, result)
    print('======= Evaluation Ends =========')
    return result

def save_ckpt(tag):
    os.makedirs(f'{log_path}/checkpoints', exist_ok=True)
    torch.save({
        'agent': agent.state_dict(),
    }, f'{log_path}/checkpoints/{tag}.pt')

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
    kwargs = {} if args.sync_venv else {'context': 'forkserver'}
    eval_envs = VecEnv(
        [make_env(args.env_id, args.seed + 1000 + i, args.control_mode,
                f'{log_path}/videos' if args.capture_video and i == 0 else None) 
        for i in range(args.num_eval_envs)],
        **kwargs,
    )
    envs = eval_envs
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"


    # dataloader setup
    if len(args.demo_path) == 2:
        args.demo_path = "checkpoints_gail/" + args.env_id + "/evaluation/" + args.demo_path + "/" + args.env_id + "_trajectories_100.pkl"
    dataset = SmallDemoDataset(args.demo_path, device, args.num_demo_traj)
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )

    # agent setup
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    model = agent


    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    loss_fn = F.mse_loss
    best_success_rate = -1

    tic = time.time()

    for iteration, data_batch in enumerate(train_dataloader):
        cur_iter = iteration + 1
        data_time = time.time() - tic

        # # copy data from cpu to gpu
        # data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

        # forward
        pred_actions = model(data_batch['observations'])

        # update losses
        total_loss = loss_fn(pred_actions, data_batch['actions'])

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time = time.time() - tic

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if cur_iter % args.log_freq == 0: 
            print(cur_iter, total_loss.item())
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], cur_iter)
            writer.add_scalar("losses/total_loss", total_loss.item(), cur_iter)

        # Evaluation
        if cur_iter % args.eval_freq == 0:
            result = evaluate(args.num_eval_episodes, agent, eval_envs, device)
            for k, v in result.items():
                writer.add_scalar(f"eval/{k}", np.mean(v), cur_iter)
            sr = np.mean(result['success'])
            if sr > best_success_rate:
                best_success_rate = sr
                save_ckpt('best_eval_success_rate')
                print(f'### Update best success rate: {sr:.4f}')



        # Checkpoint
        if args.save_freq and cur_iter % args.save_freq == 0:
            save_ckpt(str(cur_iter))

        tic = time.time()

    envs.close()
    writer.close()