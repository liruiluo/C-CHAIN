import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import metaworld  # noqa: F401  # needed to register Meta-World envs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import copy


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentReLU(nn.Module):
    def __init__(self, envs, scale_up_ratio=1, sparsity_level=0, post_LN=False):
        super().__init__()
        self.scale_up_ratio = scale_up_ratio
        self.sparsity_level = sparsity_level
        self.post_LN = post_LN

        self.hidden_dim = 256 * scale_up_ratio

        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))

        if not self.post_LN:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(obs_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, 1), std=1.0),
            )
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(obs_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, act_dim), std=0.01),
            )
        else:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(obs_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_dim, elementwise_affine=False),
                layer_init(nn.Linear(self.hidden_dim, 1), std=1.0),
            )
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(obs_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_dim, elementwise_affine=False),
                layer_init(nn.Linear(self.hidden_dim, act_dim), std=0.01),
            )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        self.sparse_init()

    def sparse_init(self):
        for param in self.actor_mean.parameters():
            if param.ndim == 1:
                continue
            sparsity_mask = np.random.choice(
                [0, 1],
                size=param.shape,
                p=[self.sparsity_level, 1 - self.sparsity_level],
            )
            sparsity_mask = torch.as_tensor(sparsity_mask, dtype=param.dtype)
            param.data.mul_(sparsity_mask)
        for param in self.critic.parameters():
            if param.ndim == 1:
                continue
            sparsity_mask = np.random.choice(
                [0, 1],
                size=param.shape,
                p=[self.sparsity_level, 1 - self.sparsity_level],
            )
            sparsity_mask = torch.as_tensor(sparsity_mask, dtype=param.dtype)
            param.data.mul_(sparsity_mask)

    def get_value(self, x):
        return self.critic(x)

    def select_action(self, obs):
        with torch.no_grad():
            action_mean = self.actor_mean(obs)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            action = probs.sample()
        return action.cpu().numpy()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(x)
        return action, log_prob, entropy, value


def make_metaworld_env(env_name, idx, capture_video, run_name, max_episode_steps, base_seed):
    def thunk():
        seed = None if base_seed is None else base_seed + idx
        env = gym.make("Meta-World/goal_observable", env_name=env_name, seed=seed)
        # ensure tasks/goals are re-sampled across episodes
        env.unwrapped._freeze_rand_vec = False
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def eval_policy_metaworld(
    eval_agent,
    env_name,
    eval_episodes=10,
    device=None,
    max_episode_steps=500,
    seed=0,
):
    env = gym.make("Meta-World/goal_observable", env_name=env_name, seed=seed)
    # allow goal / random vector to change across episodes
    env.unwrapped._freeze_rand_vec = False
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = gym.wrappers.ClipAction(env)

    episode_returns, episode_lengths = [], []
    success_count = 0
    for _ in range(eval_episodes):
        obs, info = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        success_flag = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = eval_agent.select_action(obs_t)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            if "success" in info and float(info["success"]) == 1.0:
                success_flag = True
            total_r += float(reward)
            steps += 1
        episode_returns.append(total_r)
        episode_lengths.append(steps)
        if success_flag:
            success_count += 1

    env.close()
    success_rate = success_count / float(eval_episodes)
    return episode_returns, episode_lengths, success_rate


@dataclass
class Args:
    scale_up_ratio: int = 1
    sparsity_level: float = 0.0

    alg: str = "ppo_relu_metaworld"
    """algorithm name"""
    env_name: str = "hammer-v3-goal-observable"
    """Metaworld env_name, e.g. hammer-v3-goal-observable"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiment"""
    eval_freq_timesteps: int = 100_000
    """timestep interval to evaluate the policy"""
    seed: int = 1
    """random seed"""

    gpu_no: str = "0"
    """GPU index to use"""
    use_cluster: bool = False
    """set True when running on cluster with external GPU control"""
    track: bool = False
    """if toggled, track experiment with Weights and Biases"""

    capture_video: bool = False
    """whether to capture videos (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    learning_rate: float = 3e-4
    """initial Adam learning rate"""
    final_learning_rate: float = 1e-4
    """final Adam learning rate for linear schedule"""
    num_envs: int = 8
    """number of parallel environments"""
    num_steps: int = 2048
    """steps per rollout in each env"""
    anneal_lr: bool = True
    """toggle learning rate annealing (linear from learning_rate to final_learning_rate)"""
    gamma: float = 0.99
    """discount factor"""
    gae_lambda: float = 0.95
    """lambda for GAE"""
    num_minibatches: int = 64
    """number of minibatches"""
    update_epochs: int = 10
    """number of epochs per update"""
    norm_adv: bool = True
    """normalize advantages"""
    clip_coef: float = 0.2
    """PPO clip coefficient"""
    clip_vloss: bool = True
    """use value clipping"""
    ent_coef: float = 0.0
    """entropy coefficient"""
    vf_coef: float = 0.5
    """value loss coefficient"""
    max_grad_norm: float = 0.5
    """max grad norm"""
    target_kl: float | None = None
    """target KL threshold"""
    max_episode_steps: int = 500
    """max episode length for Metaworld env (default 500)"""

    # filled at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.alg}__{args.env_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            name=run_name,
            project="ppo_experiments",
            sync_tensorboard=True,
            config=vars(args),
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if not args.use_cluster:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script (no CPU fallback).")
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [
            make_metaworld_env(
                args.env_name,
                i,
                args.capture_video,
                run_name,
                args.max_episode_steps,
                args.seed,
            )
            for i in range(args.num_envs)
        ]
    )
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    eval_env_name = args.env_name

    agent = AgentReLU(envs, scale_up_ratio=args.scale_up_ratio, sparsity_level=args.sparsity_level).to(
        device
    )
    optimizer = optim.Adam(
        agent.parameters(),
        lr=args.learning_rate / np.sqrt(args.scale_up_ratio),
        eps=1e-5,
    )

    init_eval, init_horizon, init_success = eval_policy_metaworld(
        agent,
        eval_env_name,
        eval_episodes=10,
        device=device,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )
    print("---------------------------------------")
    print(
        f"T: 0, Evaluation over {len(init_eval)} episodes. "
        f"Scores: {np.mean(init_eval):.3f}, Horizons: {np.mean(init_horizon):.3f}"
    )
    print("---------------------------------------")
    writer.add_scalar("charts/Eval", np.mean(init_eval), 0)
    writer.add_scalar("charts/Eval_Horizon", np.mean(init_horizon), 0)
    writer.add_scalar("charts/SuccessRate", init_success, 0)

    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    life_step = 0
    start_time = time.time()

    eval_cnt = 0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            progress = (iteration - 1.0) / max(args.num_iterations - 1.0, 1.0)
            base_lr_now = args.learning_rate + progress * (
                args.final_learning_rate - args.learning_rate
            )
            lrnow = base_lr_now / np.sqrt(args.scale_up_ratio)
            optimizer.param_groups[0]["lr"] = lrnow
        # no need for frac here; kept for similarity with C-CHAIN

        # Rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            life_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ret = info["episode"]["r"]
                        length = info["episode"]["l"]
                        print(
                            f"global_step={global_step}, life_step={life_step}, episodic_return={ret}"
                        )
                        writer.add_scalar("charts/episodic_return", ret, life_step)
                        writer.add_scalar("charts/episodic_length", length, life_step)

        # Periodic evaluation
        if (global_step // args.eval_freq_timesteps) > eval_cnt:
            eval_cnt += 1
            cur_eval, cur_horizon, cur_success = eval_policy_metaworld(
                agent,
                eval_env_name,
                eval_episodes=10,
                device=device,
                max_episode_steps=args.max_episode_steps,
                seed=args.seed + eval_cnt,
            )
            print("---------------------------------------")
            print(
                f"T: {global_step}, life_step={life_step}, Evaluation over {len(cur_eval)} episodes. "
                f"Scores: {np.mean(cur_eval):.3f}, Horizons: {np.mean(cur_horizon):.3f}"
            )
            print("---------------------------------------")
            writer.add_scalar("charts/Eval", np.mean(cur_eval), eval_cnt)
            writer.add_scalar("charts/Eval_Horizon", np.mean(cur_horizon), eval_cnt)
            writer.add_scalar("charts/SuccessRate", cur_success, eval_cnt)

        # Compute advantages and returns
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[
                    t
                ]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        b_ref_inds = np.arange(args.batch_size)
        clipfracs = []

        pi_loss_list, v_loss_list, entropy_list = [], [], []

        # PPO update
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            np.random.shuffle(b_ref_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    )

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                pi_loss_list.append(pg_loss.item())
                v_loss_list.append(v_loss.item())
                entropy_list.append(entropy_loss.item())

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Diagnostics
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], life_step
        )
        writer.add_scalar(
            "losses/policy_loss", float(np.mean(pi_loss_list)), life_step
        )
        writer.add_scalar(
            "losses/value_loss", float(np.mean(v_loss_list)), life_step
        )
        writer.add_scalar(
            "losses/entropy", float(np.mean(entropy_list)), life_step
        )
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), life_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), life_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), life_step)
        writer.add_scalar("losses/explained_variance", explained_var, life_step)
        sps = int(life_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, life_step)

    envs.close()
    writer.close()
