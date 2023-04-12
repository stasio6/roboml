ALGO_NAME = 's2s-DAgger'
# State-to-state DAgger, used to verify the correctness of the State-to-Visual implementation

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
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="roboml",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="PickCube-v2",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps-per-collect", type=int, default=4000, # this hp is pretty important
        help="the number of steps to run in all environment in total per policy rollout")
    parser.add_argument("--minibatch-size", type=int, default=2000,
        help="the size of mini-batches")
    parser.add_argument("--num-steps-per-update", type=float, default=1, # should be tuned based on sim time and tranining time
        help="the ratio between env steps and num of gradient updates, lower means more updates")
    parser.add_argument("--bc-loss-th", type=float, default=0.01, # Stan, you can also set this to None
        help="if the bc loss is smaller than this threshold, then stop training and collect new data")

    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--log-freq", type=int, default=10000)
    parser.add_argument("--sync-venv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--save-freq", type=int, default=200_000)
    parser.add_argument("--control-mode", type=str, default='pd_ee_delta_pos')
    parser.add_argument("--expert-ckpt", type=str, default='output/PickCube-v2/SAC-ms2-new/230411-170847_1_SAC/checkpoints/300032.pt')

    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.script = __file__
    assert args.num_steps_per_collect % args.num_envs == 0
    args.num_steps = int(args.num_steps_per_collect // args.num_envs)
    assert args.num_steps_per_collect % args.minibatch_size == 0
    args.num_minibatches = int(args.num_steps_per_collect // args.minibatch_size)
    args.num_updates_per_collect = int(args.num_steps_per_collect / args.num_steps_per_update)
    assert args.num_updates_per_collect % args.num_minibatches == 0
    args.update_epochs = int(args.num_updates_per_collect // args.num_minibatches)
    args.num_eval_envs = min(args.num_eval_envs, args.num_eval_episodes)
    assert args.num_eval_episodes % args.num_eval_envs == 0
    # fmt: on
    return args

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
gym.logger.set_level(gym.logger.DEBUG)

def make_env(env_id, seed, control_mode=None, video_dir=None):
    def thunk():
        env = gym.make(env_id, reward_mode='dense', obs_mode='state', control_mode=control_mode)
        if video_dir:
            env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

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
            action = agent.get_eval_action(torch.Tensor(obs).to(device))
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
    tmp_env = make_env(args.env_id, seed=0)()
    if tmp_env.spec.max_episode_steps > args.num_steps:
        print("\033[93mWARN: num_steps is less than max episode length. "
            "Consider raise num_steps_per_collect or lower num_envs. Continue?\033[0m")
        aaa = input()
    del tmp_env
    VecEnv = gym.vector.SyncVectorEnv if args.sync_venv else gym.vector.AsyncVectorEnv
    kwargs = {} if args.sync_venv else {'context': 'forkserver'}
    envs = VecEnv(
        [make_env(args.env_id, args.seed + i, args.control_mode) for i in range(args.num_envs)],
        **kwargs
    )
    eval_envs = VecEnv(
        [make_env(args.env_id, args.seed + 1000 + i, args.control_mode,
                f'{log_path}/videos' if args.capture_video and i == 0 else None) 
        for i in range(args.num_eval_envs)],
        **kwargs,
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
    
    for key in ['Agent', 'Actor', 'QNetwork']:
        if hasattr(m, key):
            Expert = m.__dict__[key]
            break
    expert = Expert(envs).to(device)
    checkpoint = torch.load(args.expert_ckpt)
    for key in ['agent', 'actor', 'q']:
        if key in checkpoint:
            expert.load_state_dict(checkpoint[key])
            break

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    global_step = 0
    next_obs = torch.Tensor(envs.reset()).to(device)
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
                # TODO: there are several options
                # 1. use the expert action at the first epoch, then use the agent action
                # 2. use expert action with probability beta, then decay beta
                # 3. use agent action (the current implementation)
                # Stan, you can try these options and see which one works better
                if update == -1:
                    action = expert.get_eval_action(next_obs).detach()
                else:
                    action = agent.get_action(next_obs)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            result = collect_episode_info(info, result)
        collect_time += time.time() - tic

        tic = time.time()

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_expert_actions = expert.get_eval_action(b_obs).detach()

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.num_steps_per_collect)
        for epoch in range(args.update_epochs):
            mean_loss = 0.0
            np.random.shuffle(b_inds)
            for start in range(0, args.num_steps_per_collect, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Behavior Cloning
                pred_actions = agent.get_action(b_obs[mb_inds])
                loss = F.mse_loss(pred_actions, b_expert_actions[mb_inds])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_loss += loss.item()
            mean_loss /= (args.num_steps_per_collect // args.minibatch_size)
            print('epoch:', epoch, 'loss:', mean_loss)
            # TODO: Stan, please try and see which option works better
            # 1. set a threshold on the loss, if the loss is below the threshold, stop training
            # 2. train fixed number of epochs
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
            # TODO: add BC here
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