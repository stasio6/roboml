ALGO_NAME = 'iCEM-BC'
# State-to-Visual DAgger, plus SAC loss, the SAC uses AAC

import os
import argparse
import random
import time
import collections
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
    parser.add_argument("--num-experiments", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--population", type=int, default=200)
    parser.add_argument("--cem-iters", type=int, default=4)
    parser.add_argument("--cem-elites", type=int, default=20)
    parser.add_argument("--cem-population-decay", type=float, default=1.0) # TODO: Try doing that
    parser.add_argument("--cem-noise-beta", type=float, default=0.0) # TODO: Tune
    parser.add_argument("--cem-gaussian-bound", type=str, choices=['clip','none'], default='clip') # TODO: Maybe add more?
    parser.add_argument("--use-bc", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--step-limit", type=int)

    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--eval-freq", type=int, default=30_000)
    parser.add_argument("--num-envs", type=int, default=20)
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
    parser.add_argument("--expert-demo-num", type=int)
    parser.add_argument("--image-size", type=int, default=64, # we have not implemented memory optimization for replay buffer, so use small image for now
        help="the size of observation image, e.g. 64 means 64x64")


    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.script = __file__
    # fmt: on
    return args

import mani_skill2.envs
from mani_skill2.utils.common import flatten_state_dict, flatten_dict_space_keys
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.vector.vec_env import VecEnvObservationWrapper
from gym.core import Wrapper, ObservationWrapper
from gym import spaces
from gym.vector.async_vector_env import (
    AsyncVectorEnv, 
    AsyncState, 
    AlreadyPendingCallError,
)

def sample_action(args, env, sigma, mu):
    # Sample and simulation
    if args.cem_noise_beta > 0:
        # colored noise, the trick from iCEM
        assert mu.ndim == 2
        # mu has shape h,d: we need to swap d and h because temporal correlations are in last axis
        # noinspection PyUnresolvedReferences
        import colorednoise
        samples = colorednoise.powerlaw_psd_gaussian(
            args.cem_noise_beta, 
            size=(args.population, mu.shape[1], mu.shape[0]),
        ).transpose([0, 2, 1])
        samples = mu + sigma * samples
        samples = np.clip(samples, env.action_space.low, env.action_space.high)
    elif args.cem_gaussian_bound is None or args.cem_gaussian_bound == 'none':
        samples = mu + sigma * np.random.randn(args.population, *mu.shape)
    elif args.cem_gaussian_bound == 'clip':
        # This trick is from iCEM (works very well in cheetah and humanoid)
        # The reason why it works well is still unknown, the reason given by paper seems wrong
        samples = mu + sigma * np.random.randn(args.population, *mu.shape)
        samples = np.clip(samples, env.action_space.low, env.action_space.high)
    else:
        raise NotImplementedError()
    return samples

class SimulateActionsWrapper(Wrapper):
    def eval_action_sequences(self, state, action_sequences):
        scores = []
        for action_seq in action_sequences:
            score = 0
            self.env.set_state(state)
            for action in action_seq:
                _, rew, _, _ = self.env.step(action)
                score += rew
            scores.append(score)
        return np.array(scores)

def split_into_chunks(a, num_chunks):
    chunk_size = len(a) // num_chunks
    remainder = len(a) % num_chunks
    ret = []
    for i in range(remainder):
        ret.append(a[i*(chunk_size+1):(i+1)*(chunk_size+1)])
    k = remainder * (chunk_size + 1)
    for i in range(num_chunks - remainder):
        ret.append(a[k+i*chunk_size:k+(i+1)*chunk_size])
    return ret

def make_env(env_id, control_mode, seed, video_dir=None, video_trigger=None):
    def thunk():
        env = gym.make(env_id, reward_mode='dense', obs_mode='state', control_mode=control_mode)
        env = SimulateActionsWrapper(env) # for sim_envs

        if video_dir:
            env = RecordEpisode(env, output_dir=video_dir, save_trajectory=True, info_on_video=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.single_observation_space = env.observation_space
        env.single_action_space = env.action_space
        return env

    return thunk

def seed_env(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

LOG_STD_MAX = 2
LOG_STD_MIN = -5

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
            print(f"episodic_return={item['episode']['r']:.4f}, success={item['success']}")
            result['return'].append(item['episode']['r'])
            result['len'].append(item["episode"]["l"])
            result['success'].append(item['success'])
    return result

class AsyncVectorEnvMPC(AsyncVectorEnv):
    # def get_state(self): # this is not necessary
    #     return np.array(self.call('get_state'))

    # def set_state(self, state):
    #     # this can only set the same states to all envs
    #     self.call('set_state', state)

    def eval_action_sequences_async(self, state, action_sequences):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `eval_action_sequences_async` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )
        
        action_sequences_chunks = split_into_chunks(action_sequences, self.num_envs)

        for pipe, action_sequences in zip(self.parent_pipes, action_sequences_chunks):
            pipe.send(("_call", ("eval_action_sequences", (state, action_sequences), {})))
        self._state = AsyncState.WAITING_CALL
        # this should be changed to a new waiting state, but I just use CALL for now

    def eval_action_sequences(self, state, action_sequences):
        self.eval_action_sequences_async(state, action_sequences)
        rewards_list = self.call_wait()
        # this should be changed to its own wait method, but I just use call_wait for now
        return np.concatenate(rewards_list, axis=0)


import importlib.util
import sys

def import_file_as_module(path, module_name='tmp_module'):
    spec = importlib.util.spec_from_file_location(module_name, path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = foo
    spec.loader.exec_module(foo)
    return foo

def run_env(args, sim_envs, eval_env, bc_env, max_timesteps, seed, expert):
    
    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    global_env_step = 0
    eval_length = max_timesteps + 1
    result = defaultdict(list)
    collect_time = training_time = eval_time = 0
    obs = eval_env.reset()
    bc_env.reset()
    sim_envs.reset()

    start_time = time.time()
    for t in range(max_timesteps):
        state = eval_env.get_state()

        mu = np.tile((env.action_space.high + env.action_space.low) / 2, [args.horizon, 1])
        sigma = np.tile((env.action_space.high - env.action_space.low) / 4, [args.horizon, 1])

        # Fit a Gaussian Distribution by CEM
        population = args.population
        for i in range(args.cem_iters):
            if i == 0 and args.use_bc:
                bc_env.set_state(state)
                for act in range(args.horizon):
                    mu[act] = expert.get_eval_action(torch.Tensor(obs).to(device)).detach().cpu().numpy()
                    obs, _, _, _ = bc_env.step(mu[act])
                global_env_step += args.horizon + 1

            samples = sample_action(args, env, sigma, mu)
            rewards = sim_envs.eval_action_sequences(state, samples)
            global_env_step += args.population * (args.horizon + 1)
            
            # Get elite set (top-k samples)
            topk_idx = np.argpartition(-rewards, args.cem_elites, axis=0)[:args.cem_elites]
            elites = np.take(samples, topk_idx, axis=0)
            
            # Update Gaussian Distribution
            mu = elites.mean(axis=0)
            sigma = elites.std(axis=0)

            # Decay the population if necessary
            if args.cem_population_decay < 1:
                population = max(int(population * args.cem_population_decay), 2 * args.cem_elites)

            # Logging
            elite_rews = np.take(rewards, topk_idx, axis=0)
            logger.info(f'CEM iter {i+1}: elite mean reward = {elite_rews.mean():.4f}, action max std = {sigma.max():.3f}')

        # Execute the first action of mean action seqeuence
        obs, rew, done, info = eval_env.step(mu[0])
        global_env_step += 1
        logger.debug(f'Action Sacle Max: {abs(mu[0]).max():.4f}')
        logger.debug(str(mu[0]))
        logger.info(f'== At step {t}, reward = {rew:.4f}, success = {info["success"]}')
        if done:
            collect_episode_info([info], result)
            eval_length = t + 1
            break

    for k, v in result.items():
        logger.info(f"{k}: {np.mean(v):.4f}")
    logger.info(str(result))
    print("Num steps to solve:", eval_length)
    print("Global env steps:", global_env_step)
    
    return eval_length, global_env_step, time.time() - start_time

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
    with open(f'{log_path}/args.json', 'w+') as f:
        json.dump(vars(args), f, indent=4)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    gym.logger.set_level(gym.logger.INFO)
    import logging
    def setup_logger(name, save_dir, filename='log.log'):
        os.makedirs(save_dir, exist_ok=True)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if save_dir:
            log_path = os.path.join(save_dir, filename)
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        # To avoid conflict with Tensorflow
        # https://stackoverflow.com/questions/33662648/tensorflow-causes-logging-messages-to-double
        logger.propagate = False

        return logger
    logger = setup_logger(name=' ', save_dir=log_path)

    # env setup
    kwargs = {'context': 'forkserver'}
    sim_envs = AsyncVectorEnvMPC(
        [make_env(args.env_id, args.control_mode, args.seed+1) for i in range(args.num_envs)],
        **kwargs
    )
    eval_env = make_env(args.env_id, args.control_mode, args.seed+1, video_dir=log_path)()
    bc_env = make_env(args.env_id, args.control_mode, args.seed+1)()
    env = eval_env
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"
    assert args.population % args.num_envs == 0

    # expert setup
    from os.path import dirname as up
    args.expert_ckpt = 'checkpoints_bc/' + args.env_id + '/checkpoints/' + args.env_id + "_20.pt"
    #print(args.ex)
    expert_dir = up(up(args.expert_ckpt))
    import json
    with open(f'{expert_dir}/args.json', 'r') as f:
        expert_args = json.load(f)
    m = import_file_as_module(expert_args['script'])
    
    for key in ['Agent', 'Actor', 'QNetwork']:
        if hasattr(m, key):
            Expert = m.__dict__[key]
            break
    expert = Expert(eval_env).to(device)
    checkpoint = torch.load(args.expert_ckpt)
    for key in ['agent', 'actor', 'q']:
        if key in checkpoint:
            expert.load_state_dict(checkpoint[key])
            break

    avg_steps, avg_env_steps, avg_wall_time = (0, 0, 0)
    left = args.num_experiments
    seed = 0
    while left > 0:
        max_timesteps = bc_env.spec.max_episode_steps - args.horizon - 1 if args.step_limit is None else args.step_limit
        steps, env_steps, wall_time = run_env(args, sim_envs, eval_env, bc_env, max_timesteps, seed, expert)
        seed += 1
        writer.add_scalar("iCEM/steps", steps, seed)
        writer.add_scalar("iCEM/env_steps", env_steps, seed)
        writer.add_scalar("iCEM/wall_time", wall_time, seed)
        writer.add_scalar("iCEM/seed", seed, seed)
        writer.add_scalar("iCEM/success_seed", args.num_experiments - left, seed)
        print("Experiment done")
        print(steps, env_steps, wall_time)
        print("Currently:", seed, left)
        if steps > max_timesteps:
            continue
        avg_steps += steps
        avg_env_steps += env_steps
        avg_wall_time += wall_time
        left -= 1

    print("Avg steps:", avg_steps / args.num_experiments)
    print("Total env steps:", avg_env_steps)
    print("Total wall time:", avg_wall_time)

    sim_envs.close()
    eval_env.flush_trajectory()
    # eval_env.flush_video()
    # eval_env.reset() # to save the video
    eval_env.close()

