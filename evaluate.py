import argparse
import os
import random
from distutils.util import strtobool

import gym
import numpy as np
import torch
from collections import defaultdict

def parse_args():
    # adapt from https://stackoverflow.com/a/61003775
    # Caveats:
    # 1. The setup overrides values in config files with values on the command line
    # 2. It only uses default values if options have not been set on the command line nor the settings file
    # 3. It does not check that the settings in the config file are valid

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--script", type=str, default=None)
    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument("--control-mode", type=str, default=None)
    parser.add_argument("-ep", "--num-eval-episodes", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("-st", "--save-trajectory", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("-ss", "--save-state", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--save-obj-id", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--render-human", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--success-only", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    args = parser.parse_args()

    if args.render_human:
        assert args.num_eval_envs == 1, "rendering human only works with 1 env"

    from os.path import dirname as up
    exp_dir = up(up(args.ckpt))

    import json
    with open(f'{exp_dir}/args.json', 'r') as f:
        tmp_args = argparse.Namespace()
        tmp_args.__dict__.update(json.load(f)) # add args of the checkpoint
        args = parser.parse_args(namespace=tmp_args) # override values by command line
    
    if args.log_path is None:
        import os.path as osp
        ckpt_name = osp.splitext(osp.basename(args.ckpt))[0]
        args.log_path = f'{exp_dir}/evaluation/{ckpt_name}'
    return args

import importlib.util
import sys

def import_file_as_module(path, module_name='tmp_module'):
    spec = importlib.util.spec_from_file_location(module_name, path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = foo
    spec.loader.exec_module(foo)
    return foo

class BaseTrajectorySaver(object):
    KEYS = [
        'observations',
        'states',
        'actions',
        # 'next_observations',
        # 'next_states',
        'rewards',
        'dones',
        'infos',
    ]
    def __init__(self, num_envs, save_dir, success_only):
        self.num_envs = num_envs
        self.save_dir = save_dir
        self.success_only = success_only
        self.traj = [
            {key: [] for key in self.KEYS} 
        for _ in range(num_envs)]
        self.data_to_save = []

    def add_transition(self, obs, act, next_obs, rew, done, info):
        for i in range(self.num_envs):
            self.traj[i]['observations'].append(obs[i])
            self.traj[i]['actions'].append(act[i])
            self.traj[i]['rewards'].append(rew[i])
            self.traj[i]['dones'].append(done[i])
            self.traj[i]['infos'].append(info[i])
            if done[i]:
                self.traj[i]['observations'].append(info[i]['terminal_observation'])
                if not self.success_only or info[i]['success']:
                    self.data_to_save.append(self.traj[i])
                self.traj[i] = {key: [] for key in self.KEYS}
    
    def add_state(self, s):
        for i in range(self.num_envs):
            self.traj[i]['states'].append(s[i])

    def add_obj_id(self, obj_id):
        assert self.num_envs == 1
        self.traj[0]['obj_id'] = obj_id

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        import pickle
        save_path = f'{self.save_dir}/{args.env_id}_trajectories_{len(self.data_to_save)}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(self.data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved {len(self.data_to_save)} trajectories to {save_path}')

    @property
    def num_traj(self):
        return len(self.data_to_save)


if __name__ == "__main__":
    args = parse_args()
    import time
    start_time = time.time()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Load Modules
    m = import_file_as_module(args.script)

    #############################################
    # Build Env
    #############################################
    if hasattr(m, 'make_env'):
        make_env = m.make_env
        VecEnv = gym.vector.SyncVectorEnv if args.save_trajectory and args.save_state or args.num_eval_envs == 1 else gym.vector.AsyncVectorEnv
        eval_envs = VecEnv(
            [make_env(env_id=args.env_id, seed=args.seed + i, 
                    control_mode=args.control_mode,
                    video_dir=f'{args.log_path}/videos' if args.capture_video and i == 0 else None,
                    # video_trigger=lambda x: x % (args.num_eval_episodes // args.num_eval_envs) == 0 ) 
                    # video_trigger=lambda x: True ) 
                    # link_id=args.link_id, obj_id=args.obj_id,
            )
            for i in range(args.num_eval_envs)]
        )
        if args.num_eval_envs == 1:
            env = eval_envs.envs[0]
    elif hasattr(m, 'make_vec_env'):
        make_vec_env = m.make_vec_env
        eval_envs = make_vec_env(args.env_id, args.num_eval_envs, args.seed+1000, args.control_mode,
                             video_dir=f'{args.log_path}/videos' if args.capture_video else None)
    else:
        raise NotImplementedError()
    envs = eval_envs

    #############################################
    # Build Agent
    #############################################
    # agent setup
    for key in ['Agent', 'Actor', 'QNetwork']:
        if hasattr(m, key):
            Agent = m.__dict__[key]
            break
    # envs.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(envs.single_observation_space.shape[0]+1,), dtype=np.float32)
    # envs.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(68,), dtype=np.float32)
    agent = Agent(envs).to(device)
    checkpoint = torch.load(args.ckpt)
    for key in ['agent', 'actor', 'q']:
        if key in checkpoint:
            agent.load_state_dict(checkpoint[key])
            break

    #############################################
    # Misc
    #############################################
    if hasattr(m, 'to_tensor'):
        to_tensor = m.to_tensor
        m.device = device
    else:
        to_tensor = lambda x: torch.tensor(x, device=device)

    #############################################
    # Evaluate
    #############################################
    if args.save_trajectory:
        saver = BaseTrajectorySaver(args.num_eval_envs, args.log_path, args.success_only)

    print('======= Evaluation Starts =========')
    
    global_ep_cnt = 0
    result = defaultdict(list)
    obs = eval_envs.reset()
    if args.save_trajectory:
        if args.save_state:
            saver.add_state([ env.get_state() for env in eval_envs.envs])
        if args.save_obj_id:
            saver.add_obj_id(env.unwrapped.model_id)
    # while len(result['return']) < args.num_eval_episodes:
    while (saver.num_traj if args.save_trajectory else len(result['return'])) < args.num_eval_episodes:
        if args.render_human:
            env.render()
        with torch.no_grad():
            obs = obs.astype(np.float32)
            action = agent.get_eval_action(to_tensor(obs)).cpu().numpy()
            # action = agent.get_eval_action(torch.Tensor([env.unwrapped._get_old_policy_input()]).to(device)).cpu().numpy()
            # action = agent.get_eval_action(torch.Tensor([env.unwrapped._get_v4_policy_input()]).to(device)).cpu().numpy()
        next_obs, rew, done, info = eval_envs.step(action)

        for item in info:
            if "episode" in item.keys():
                global_ep_cnt += 1
                if 'success' in item:
                    success = item['success']
                elif 'goal_achieved' in item:
                    success = item['goal_achieved']
                else:
                    success = 0
                print(f"{global_ep_cnt}: episodic_return={item['episode']['r']:.4f}, success={success}, steps={item['episode']['l']}")
                result['return'].append(item['episode']['r'])
                result['len'].append(item["episode"]["l"])
                result['success'].append(success)
        
        if args.save_trajectory:          
            saver.add_transition(obs, action, next_obs, rew, done, info)
            if args.save_state:
                saver.add_state([ env.get_state() for env in eval_envs.envs])
            if done[0]:
                if args.save_obj_id:
                    saver.add_obj_id(env.unwrapped.model_id) # must be after add_transition

        obs = next_obs
    
    print('======= Evaluation Ends =========')

    print(result)
    for k, v in result.items():
        print(f"{k}: {np.mean(v):.4f}")

    envs.close()
    if args.save_trajectory:
        saver.save()
    print(time.time() - start_time)
