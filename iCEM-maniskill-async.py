ALGO_NAME = 'MPC-CEM'

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np

import datetime
from collections import defaultdict
from utils.logger import setup_logger

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="PickPanda_plus1_state-v0",
        help="the id of the environment")
    # parser.add_argument("--gamma", type=float, default=0.99,
    #     help="the discount factor gamma")
    parser.add_argument("--max-timesteps", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--population", type=int, default=300)
    parser.add_argument("--cem-iters", type=int, default=8)
    parser.add_argument("--cem-elites", type=int, default=10)
    parser.add_argument("--cem-momentum", type=float, default=0.1)
    parser.add_argument("--cem-population-decay", type=float, default=1.0)
    parser.add_argument("--cem-new-action-init", type=str, choices=['mean', 'repeat'], default='mean')
    parser.add_argument("--cem-noise-beta", type=float, default=0.0)

    # The following arguments are very important, try them when facing a new task
    parser.add_argument("--cem-gaussian-bound", type=str, choices=['clip','PETS','trunc','none'], default='clip')

    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--num-envs", type=int, default=32)
    
    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.script = __file__
    if args.cem_gaussian_bound == 'PETS' and args.cem_new_action_init == 'repeat':
        print('This combination is not recommended, try to use --cem-new-action-init mean')
        aa = input()
    # fmt: on
    return args

from gym import Wrapper

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

import mani_skill2.envs
from mani_skill2.utils.wrappers import ManiSkillActionWrapper, NormalizeActionWrapper, RenderInfoWrapper
gym.logger.set_level(gym.logger.INFO)

def make_env(env_id, seed, video_dir=None, video_trigger=None):
    def thunk():
        env = gym.make(env_id)
        env = ManiSkillActionWrapper(env)
        env = NormalizeActionWrapper(env)
        env = SimulateActionsWrapper(env) # for sim_envs

        if video_dir:
            env = RenderInfoWrapper(env) # this is required for RecordVideo wrapper to work
            env = gym.wrappers.RecordVideo(env, video_dir, video_trigger)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

from gym.vector.async_vector_env import (
    AsyncVectorEnv, 
    AsyncState, 
    AlreadyPendingCallError,
)

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


def collect_episode_info(info, result=None):
    if result is None:
        result = defaultdict(list)
    for item in info:
        if "episode" in item.keys():
            # logger.info(f"sim_count={i_sim}, episodic_return={item['episode']['r']:.4f}")
            result['return'].append(item['episode']['r'])
            result['len'].append(item["episode"]["l"])
            result['success'].append(item['success'])
    return result


if __name__ == "__main__":
    args = parse_args()

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    tag = '{:s}_{:d}'.format(now, args.seed)
    if args.exp_name: tag += '_' + args.exp_name
    log_name = os.path.join(args.env_id, ALGO_NAME, tag)
    log_path = os.path.join(args.output_dir, log_name)

    os.makedirs(log_path, exist_ok=True)
    logger = setup_logger(name=' ', save_dir=log_path)

    import json
    with open(f'{log_path}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    # env setup
    sim_envs = AsyncVectorEnvMPC(
        [make_env(args.env_id, args.seed + 1) for i in range(args.num_envs)]
    )
    eval_env = make_env(args.env_id, args.seed + 1, 
                        f'{log_path}/videos' if args.capture_video else None, 
                        lambda x: True,
    )()
    env = eval_env
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    max_timesteps = env.spec.max_episode_steps if args.max_timesteps is None else args.max_timesteps
    result = defaultdict(list)

    eval_env.reset()
    sim_envs.reset()

    for t in range(max_timesteps):
        state = eval_env.get_state()

        # Initialize mean and std of the action sequence of length H
        if t == 0:
            mu = np.tile((env.action_space.high + env.action_space.low) / 2, [args.horizon, 1])
        else:
            mu[:-1] = mu[1:] # shift initialization
            if args.cem_new_action_init == 'mean':
                # generally, this is a safer choice
                mu[-1] = (env.action_space.high + env.action_space.low) / 2
            elif args.cem_new_action_init == 'repeat':
                mu[-1] = mu[-2]
            else:
                raise NotImplementedError()
        sigma = np.tile((env.action_space.high - env.action_space.low) / 4, [args.horizon, 1])

        # Fit a Gaussian Distribution by CEM
        population = args.population
        for i in range(args.cem_iters):

            # Sample and simulation
            if args.cem_noise_beta > 0:
                # colored noise, the trick from iCEM
                assert mu.ndim == 2
                # mu has shape h,d: we need to swap d and h because temporal correlations are in last axis
                # noinspection PyUnresolvedReferences
                import colorednoise
                samples = colorednoise.powerlaw_psd_gaussian(
                    args.cem_noise_beta, 
                    size=(population, mu.shape[1], mu.shape[0]),
                ).transpose([0, 2, 1])
                samples = mu + sigma * samples
                samples = np.clip(samples, env.action_space.low, env.action_space.high)
            elif args.cem_gaussian_bound is None or args.cem_gaussian_bound == 'none':
                samples = mu + sigma * np.random.randn(population, *mu.shape)
            elif args.cem_gaussian_bound == 'clip':
                # This trick is from iCEM (works very well in cheetah and humanoid)
                # The reason why it works well is still unknown, the reason given by paper seems wrong
                samples = mu + sigma * np.random.randn(population, *mu.shape)
                samples = np.clip(samples, env.action_space.low, env.action_space.high)
            elif args.cem_gaussian_bound == 'trunc':
                # typical way to bound the actions in MPC-CEM
                lower = (env.action_space.low - mu) / (sigma + 1e-8)
                upper = (env.action_space.high - mu) / (sigma + 1e-8)

                from scipy.stats import truncnorm
                m = mu[None, :]
                s = sigma[None, :]
                samples = truncnorm.rvs(lower, upper, loc=m, scale=s,
                                        size=(population, *mu.shape))
            elif args.cem_gaussian_bound == 'PETS':
                # from PETS paper, truncation is set to 2\sigma
                # \sigma is adpated to be not larger than 0.5*b
                # where b is the minimum distance to the action bounds
                lb, ub = env.action_space.low, env.action_space.high
                lb_dist, ub_dist = mu - lb, ub - mu
                _std = np.maximum(1e-8, np.minimum(np.minimum(lb_dist / 2, ub_dist / 2), sigma))
                lower, upper = -2, 2

                from scipy.stats import truncnorm
                m = mu[None, :]
                s = _std[None, :]
                samples = truncnorm.rvs(lower, upper, loc=m, scale=s,
                                        size=(population, *mu.shape))
                
            else:
                raise NotImplementedError()
            rewards = sim_envs.eval_action_sequences(state, samples)
            
            # Get elite set (top-k samples)
            topk_idx = np.argpartition(-rewards, args.cem_elites, axis=0)[:args.cem_elites]
            elites = np.take(samples, topk_idx, axis=0)
            
            # Update Gaussian Distribution
            mu = args.cem_momentum * mu + (1 - args.cem_momentum) * elites.mean(axis=0)
            sigma = args.cem_momentum * sigma + (1 - args.cem_momentum) * elites.std(axis=0)

            # Decay the population if necessary
            if args.cem_population_decay < 1:
                population = max(int(population * args.cem_population_decay), 2 * args.cem_elites)

            # Logging
            elite_rews = np.take(rewards, topk_idx, axis=0)
            logger.info(f'CEM iter {i+1}: elite mean reward = {elite_rews.mean():.4f}, action max std = {sigma.max():.3f}')

        # Execute the first action of mean action seqeuence
        _, rew, done, info = eval_env.step(mu[0])
        logger.debug(f'Action Sacle Max: {abs(mu[0]).max():.4f}')
        logger.debug(str(mu[0]))
        logger.info(f'== At step {t}, reward = {rew:.4f}, success = {info["success"]}')
        if done:
            collect_episode_info([info], result)
            break
    
    sim_envs.close()
    eval_env.close()

    for k, v in result.items():
        logger.info(f"{k}: {np.mean(v):.4f}")
    logger.info(str(result))