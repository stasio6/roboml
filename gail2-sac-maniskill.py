ALGO_NAME = 'GAILv2-SAC'
# GAIL with 3 buffer: 
# replay buffer, success_buffer: expert + success agent traj, failure_buffer: failed agent traj

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import datetime
from collections import defaultdict

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
    parser.add_argument("--wandb-project-name", type=str, default="reward-learning",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="LiftCube-v2",
        help="the id of the environment")
    parser.add_argument("--demo-path", type=str, default='output/LiftCube-v2/SAC/221208-155359_1_try/evaluation/150016/trajectories_100.pkl',
        help="the path of demo H5 file")
    parser.add_argument("--total-timesteps", type=int, default=700_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=None,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.8,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=1024,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=4000,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--disc-lr", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--policy-frequency", type=int, default=1,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--disc-frequency", type=int, default=1,
        help="the frequency of training discriminator (delayed)")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    
    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--sync-venv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--num-steps-per-update", type=float, default=2)
    parser.add_argument("--training-freq", type=int, default=64)
    parser.add_argument("--log-freq", type=int, default=10000)
    parser.add_argument("--num-demo-traj", type=int, default=None)
    parser.add_argument("--reward-mode", type=str, required=True)
    parser.add_argument("--obs-norm", type=str, choices=['fixed','online',None], default=None)
    parser.add_argument("--save-freq", type=int, default=200_000)
    parser.add_argument("--value-always-bootstrap", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="in ManiSkill variable episode length setting, set to True if positive reawrd, False if negative reward.")
    parser.add_argument("--control-mode", type=str, default='pd_ee_delta_pos')

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
    assert args.obs_norm is None, 'currently only support no obs norm'
    # fmt: on
    return args

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode

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


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256),
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
    def __init__(self, env):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.action_scale = torch.FloatTensor((env.single_action_space.high - env.single_action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((env.single_action_space.high + env.single_action_space.low) / 2.0)

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, x):
        mean, log_std = self(x)
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

class Discriminator(nn.Module):
    def __init__(self, envs):
        super().__init__()
        state_shape = np.prod(envs.single_observation_space.shape)
        action_shape = np.prod(envs.single_action_space.shape)

        # self.net =  nn.Sequential(
        #     nn.Linear(state_shape + action_shape, 32),
        #     nn.Sigmoid(),
        #     nn.Linear(32, 1),
        # )
        # self.net =  nn.Sequential(
        #     nn.Linear(state_shape + action_shape, 64),
        #     nn.Sigmoid(),
        #     nn.Linear(64, 1),
        # )
        # self.net =  nn.Sequential(
        #     nn.Linear(state_shape + action_shape, 64),
        #     nn.Sigmoid(),
        #     nn.Linear(64, 64),
        #     nn.Sigmoid(),
        #     nn.Linear(64, 1),
        # )
        self.net = nn.Sequential(
            nn.Linear(state_shape + action_shape, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)

    def get_reward(self, s, a, mode='GAIL'):
        with torch.no_grad():
            logit = self(s, a) # D = sigmoid(logit)
            # The following results use: 1 - sigmoid(logit) = sigmoid(-logit)
            if mode == 'GAIL' or mode == 'positive': 
                # maximize E_{\pi} [-log(1 - D)].
                # equivalent to -log(1 - sigmoid(logit)) = -log(sigmoid(-logit))
                return -F.logsigmoid(-logit)
            elif mode == 'AIRL' or mode == 'identity':
                # maximize E_{\pi} [log(D) - log(1 - D)].
                # equivalent to log(sigmoid(logit)) - log(1 - sigmoid(logit)) = logit
                return logit
            elif mode == 'negative':
                # maximize E_{\pi} [log(D)].
                # equivalent to log(sigmoid(logit))
                return F.logsigmoid(logit)
            else:
                raise NotImplementedError()

class DiscriminatorBuffer(object):
    # can be optimized by create a buffer of size (n_traj, len_traj, dim)
    def __init__(self, buffer_size, obs_space, action_space, device):
        self.buffer_size = buffer_size
        self.observations = np.zeros((self.buffer_size,) + obs_space.shape, dtype=obs_space.dtype)
        self.actions = np.zeros((self.buffer_size, int(np.prod(action_space.shape))), dtype=action_space.dtype)
        self.device = device
        self.pos = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.buffer_size if self.full else self.pos

    def add(self, obs, actions):
        l = obs.shape[0]
        
        while self.pos + l >= self.buffer_size:
            self.full = True
            k = self.buffer_size - self.pos
            self.observations[self.pos:] = obs[:k]
            self.actions[self.pos:] = actions[:k]
            self.pos = 0
            obs = obs[k:]
            actions = actions[k:]
            l = obs.shape[0]
            
        self.observations[self.pos:self.pos+l] = obs.copy()
        self.actions[self.pos:self.pos+l] = actions.copy()
        self.pos = (self.pos + l) % self.buffer_size

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            observations=self.observations[idxs],
            actions=self.actions[idxs],
        )
        return {k: torch.tensor(v).to(self.device) for k,v in batch.items()}


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
    envs = VecEnv(
        [make_env(args.env_id, args.seed + i, args.control_mode) for i in range(args.num_envs)],
        **kwargs
    )
    eval_envs = VecEnv(
        [make_env(args.env_id, args.seed + 1000 + i, args.control_mode,
                f'{log_path}/videos' if args.capture_video and i == 0 else None) 
        for i in range(args.num_eval_envs)],
        **kwargs
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    max_action = float(envs.single_action_space.high[0])

    # demo dataset setup
    if 'pyrl' in args.demo_path:
        from utils.pyrl_data import load_demo_dataset
    else:
        from rookie.utils.data import load_demo_dataset
    demo_dataset = load_demo_dataset(args.demo_path, num_traj=args.num_demo_traj)
    demo_size = demo_dataset['actions'].shape[0]

    # discriminator setup
    disc = Discriminator(envs).to(device)
    disc_optimizer = optim.Adam(disc.parameters(), lr=args.disc_lr)

    # agent setup
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
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
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=True,
    )

    # GAILv2 specific
    sb = DiscriminatorBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
    )
    sb.add(obs=demo_dataset['observations'], actions=demo_dataset['actions'])
    fb = DiscriminatorBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
    )
    tmp_env = make_env(args.env_id, seed=0)()
    max_t = tmp_env.spec.max_episode_steps
    del tmp_env
    episode_obs = np.zeros((args.num_envs, max_t) + envs.single_observation_space.shape)
    episode_actions = np.zeros((args.num_envs, max_t) + envs.single_action_space.shape)
    step_in_episodes = np.zeros((args.num_envs,1,1), dtype=np.int32)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs = envs.reset()
    global_step = 0
    global_env_step = 0
    global_update = 0
    learning_has_started = False
    num_updates_per_training = int(args.training_freq // args.num_steps_per_update)
    disc_rew_sum = np.zeros(args.num_envs)
    result = defaultdict(list)
    collect_time = training_time = disc_time = eval_time = 0

    while global_step < args.total_timesteps:

        #############################################
        # Interact with environments
        #############################################
        tic = time.time()
        for local_step in range(args.training_freq // args.num_envs):
            global_step += 1 * args.num_envs

            # ALGO LOGIC: put action logic here
            if not learning_has_started:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = envs.step(actions)
            global_env_step += args.num_envs

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            result = collect_episode_info(infos, result)

            # GAILv2: record data for the current episode, add data to success buffer and failure buffer
            np.put_along_axis(episode_obs, step_in_episodes, values=obs[:, None, :], axis=1)
            np.put_along_axis(episode_actions, step_in_episodes, values=actions[:, None, :], axis=1)
            step_in_episodes += 1

            # GAIL-specific: record disc_return, just for plotting purpose
            disc_rewards = disc.get_reward(torch.Tensor(obs).to(device), torch.Tensor(actions).to(device), mode=args.reward_mode)
            disc_rew_sum += disc_rewards.flatten().cpu().numpy()

            # Prepare for the terminal observation
            real_next_obs = next_obs.copy()

            # Do everything for terminated episodes
            for i, d in enumerate(dones):
                if d:
                    # handle `terminal_observation`
                    real_next_obs[i] = infos[i]["terminal_observation"]

                    # record disc_return
                    result['GAIL/disc_return'].append(disc_rew_sum[i])
                    disc_rew_sum[i] = 0
                    
                    # add completed trajectory to disc buffer
                    l = step_in_episodes[i,0,0]
                    if infos[i]['success']:
                        sb.add(episode_obs[i, :l], episode_actions[i, :l])
                    else:
                        fb.add(episode_obs[i, :l], episode_actions[i, :l])
                    step_in_episodes[i] = 0

            # TRY NOT TO MODIFY: save data to reply buffer;
            rb.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
        collect_time += time.time() - tic

        # ALGO LOGIC: training.
        if global_step < args.learning_starts:
            continue

        learning_has_started = True
        for local_update in range(num_updates_per_training):
            global_update += 1
            data = rb.sample(args.batch_size)

            #############################################
            # Train discriminator
            #############################################
            tic = time.time()
            if global_update % args.disc_frequency == 0:
                success_data = sb.sample(args.batch_size)
                fail_data = fb.sample(args.batch_size)

                disc_obs = torch.cat([fail_data['observations'], success_data['observations']], dim=0)
                disc_actions = torch.cat([fail_data['actions'], success_data['actions']], dim=0)
                disc_labels = torch.cat([
                    torch.zeros((args.batch_size, 1), device=device), # fail label is 0
                    torch.ones((args.batch_size, 1), device=device), # success label is 1
                ], dim=0)

                logits = disc(disc_obs, disc_actions)
                disc_loss = F.binary_cross_entropy_with_logits(logits, disc_labels)
                
                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()

                pred = logits.detach() > 0
                disc_acc_demo = (pred & disc_labels.bool()).sum().item() / args.batch_size
                disc_acc_agent = (~pred & ~disc_labels.bool()).sum().item() / args.batch_size

                # print(f'Disc Acc: demo {disc_acc_demo*100:.2f}%, agent {disc_acc_agent*100:.2f}%')
            disc_time += time.time() - tic

            #############################################
            # Train agent
            #############################################
            # compute reward by discriminator
            tic = time.time()
            disc_rewards = disc.get_reward(data.observations, data.actions, mode=args.reward_mode)

            # update the value networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                if args.value_always_bootstrap:
                    next_q_value = disc_rewards.flatten() + args.gamma * (min_qf_next_target).view(-1)
                else:
                    next_q_value = disc_rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # timeouts have been hanlded by ReplayBuffer

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # update the policy network
            if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                pi, log_pi, _ = actor.get_action(data.observations)
                qf1_pi = qf1(data.observations, pi)
                qf2_pi = qf2(data.observations, pi)
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
                    tag = k if '/' in k else f"train/{k}"
                    writer.add_scalar(tag, np.mean(v), global_step)
                result = defaultdict(list)
            writer.add_scalar("GAIL/disc_loss", disc_loss.item(), global_step)
            writer.add_scalar("GAIL/acc_demo", disc_acc_demo, global_step)
            writer.add_scalar("GAIL/acc_agent", disc_acc_agent, global_step)
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
            writer.add_scalar("charts/disc_time", disc_time / tot_time, global_step)
            writer.add_scalar("charts/eval_time", eval_time / tot_time, global_step)
            writer.add_scalar("charts/collect_SPS", int(global_step / collect_time), global_step)
            writer.add_scalar("charts/training_SPS", int(global_step / training_time), global_step)
            writer.add_scalar("charts/disc_SPS", int(global_step / disc_time), global_step)
            writer.add_scalar("charts/env_steps", global_env_step, global_step)
            if args.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        # Evaluation
        tic = time.time()
        if (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            result = evaluate(args.num_eval_episodes, actor, eval_envs, device)
            for k, v in result.items():
                writer.add_scalar(f"eval/{k}", np.mean(v), global_step)
        eval_time += time.time() - tic

        # Checkpoint
        if args.save_freq and ( global_step >= args.total_timesteps or \
                (global_step - args.training_freq) // args.save_freq < global_step // args.save_freq):
            os.makedirs(f'{log_path}/checkpoints', exist_ok=True)
            torch.save({
                'actor': actor.state_dict(), # does NOT include action_scale
                'discriminator': disc.state_dict(),
                'qf1': qf1_target.state_dict(),
                'qf2': qf2_target.state_dict(),
            }, f'{log_path}/checkpoints/{global_step}.pt')

    envs.close()
    writer.close()