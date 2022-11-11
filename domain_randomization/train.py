"""
This script uses domain randomization to effectively train a RL agent with a variety of possible masses
so to make the learned policy robust to changes in the overall distribution.
"""
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import gym
import argparse
import numpy as np
from tqdm import tqdm

import env.pendulum
import algos.TD3 as TD3
import algos.DDPG as DDPG
import utils.replay_buffer as replay


def parse_args():
    parser = argparse.ArgumentParser(description='Uniform Domain Randomization')
    # Args of Environment:
    parser.add_argument('--domain_type', default='source', type=str, 
        help='source / target')
    # parser.add_argument('--n_samples', default=5, type=int, 
    #     help='Number of training episodes')
    parser.add_argument('--max_timesteps', default=int(1e5), type=int,
        help='Number of total timesteps (default: 1e8)')
    parser.add_argument('--max_steps', default=200, type=int,
                    help='Maximum number of steps in each episode (default: 200)')
    parser.add_argument('--mass_low', default=0.5, type=float, 
        help='lower bound of mass to uniform distribution')
    parser.add_argument('--mass_high', default=1.5, type=float, 
        help='upper bound of mass to uniform distribution')
    parser.add_argument('--length_low', default=0.5, type=float, 
        help='lower bound of length to uniform distribution')
    parser.add_argument('--length_high', default=1.5, type=float, 
        help='upper bound of length to uniform distribution')
    parser.add_argument('--timesteps', default=100000, type=int, 
        help='Timesteps used to train the agent')
    # Args of Agents:
    parser.add_argument("--policy", default="TD3",
        help='Which algorithms? DDPG or TD3 (default: TD3)')
    parser.add_argument("--hidden_dim", default=256, type=int, 
        help='Number of nodes in hidden layers (default: 256)')
    parser.add_argument('--discount', default=0.99, type=float, metavar='G',
        help='discount factor, gamma (default: 0.99)')
    parser.add_argument('--lr', default=3e-4, type=float, metavar='G',
        help='learning rate, alpha (default: 1e-5)')
    parser.add_argument("--start_timesteps", default=int(1e4), type=int, 
        help='Number of steps for uniform-random action selection (default: 25e3)')
    # DDPG:
    parser.add_argument('--tau', default=0.005, type=float, metavar='G',
        help='Target network update rate (default: 0.005)')
    # TD3:
    parser.add_argument("--act_noise", default=0.1, type=float,
        help='Stddev for Gaussian exploration noise (default: 0.1)')
    parser.add_argument("--target_noise", default=0.2, type=float,
        help='Stddev for smoothing noise added to target policy (default: 0.2)')
    parser.add_argument("--noise_clip", default=0.5, type=float,
        help='Clipping range of target policy smoothing noise (default: 0.5)')
    parser.add_argument('--policy_update_freq', default=2, type=int, metavar='N',
        help='Frequency of “Delayed” policy updates (default: 2)')
    # Args of Replay buffer:
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
        help='Batch size of actor and critic networks (default: 256)')
    parser.add_argument('--replay_buffer_size', default=int(1e6), type=int, metavar='N',
        help='Maximum size of replay buffer (default: 1e6)')
    parser.add_argument('--seed', default=123, type=int, metavar='N',
        help='Random seed of Gym, PyTorch and Numpy (default: 123)')   
    parser.add_argument("--save_model", default=True, action="store_true",
        help='Save models and optimizer parameters (default: True)')
    parser.add_argument("--load_model", default=False, type=bool,
        help='Load trained models and optimizer parameters (default: False)')      
    return parser.parse_args()
args = parse_args()

def main():
    if args.domain_type.lower() == 'target': 
        env_id = "Pendulum-target-v2"
        env = gym.make(env_id)
    elif args.domain_type.lower() == 'source':
        env_id = "Pendulum-source-v2"
        env = gym.make(env_id)
        default_bounds = [args.mass_low, args.mass_high, args.length_low, args.length_high]
        env.set_parametric_dist(default_bounds)

    # Show information:
    print("-----------------------------------------")
    print(f"Env: {env_id}, Policy: {args.policy}, Seed: {args.seed}")
    print("-----------------------------------------")

    # Set seed for random number generators:
    if args.seed:
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        np.random.seed(args.seed)

    # Initialize policy:
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    min_act = env.action_space.low[0]
    max_act = env.action_space.high[0]
    avrg_act = (min_act+max_act)/2.0 
    scale_act = max_act-avrg_act # actor scaling

    kwargs = {
        "state_dim" : state_dim,
        "action_dim": action_dim,
        "hidden_dim": args.hidden_dim,
        "min_act": min_act,
        "max_act": max_act,
        "avrg_act": avrg_act,
        "scale_act" : scale_act,
        "discount": args.discount,
        "lr": args.lr,
        "tau": args.tau,
    }

    if args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)
    elif args.policy == "TD3":
        kwargs["target_noise"] = args.target_noise 
        kwargs["noise_clip"] = args.noise_clip 
        kwargs["policy_update_freq"] = args.policy_update_freq
        policy = TD3.TD3(**kwargs)
     
    # Set experience replay buffer:
    replay_buffer = replay.ReplayBuffer(state_dim, action_dim, args.replay_buffer_size)

    # Save models and optimizer parameters:
    file_name = f"{args.policy}_{env_id}"
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    max_total_reward = -1e9 # to save best models

    # Load trained models and optimizer parameters:
    if args.load_model == True:
        policy.load(f"./models/{file_name + '_best'}")

    # Setup loggers:
    if not os.path.exists("./results"):
        os.makedirs("./results")

    log_epi_path  = os.path.join("./results", "log_epi.txt") 
    log_step_path = os.path.join("./results", "log_step.txt")   
    log_epi  = open(log_epi_path, "w+")  # no. episode vs. Total reward vs. Each timesteps
    log_step = open(log_step_path, "w+") # Total timesteps vs. Total reward
    
    # Initialize environment:
    state, done = env.reset(), False
    i_episode = 0
    episode_reward = 0
    episode_timesteps = 0

    # iterations = tqdm(range(args.n_samples))
    iterations = tqdm(range(args.max_timesteps))
    
    for total_timesteps in iterations: 
        iterations.set_description(f"Total timesteps {total_timesteps}")

        episode_timesteps += 1

        # Select action randomly or from policy:
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample() 
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, avrg_act * args.act_noise, size=action_dim)
            ).clip(min_act, max_act)

        # changing the masses of the hopper
        if args.domain_type.lower() == 'source':
            env.set_random_parameters()

        # Perform action:
        next_state, reward, done, _ = env.step(action)
        
        # Episode termination:
        if episode_timesteps == args.max_steps:
            done = True
        done_bool = float(done) if episode_timesteps < args.max_steps else 0

        # Store a set of transitions in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data:
        if total_timesteps >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            print(f"Total timestpes: {total_timesteps+1}, #Episode: {i_episode+1}, timestpes: {episode_timesteps}, Reward: {episode_reward:.3f}")
                        
            # Save best model:
            if episode_reward > max_total_reward:
                print("#---------------- Best! ----------------#")
                best_model = 'Best Model!'
                max_total_reward = episode_reward
                policy.save(f"./models/{file_name + '_best'}")
            else:
                best_model = ''

            # Log data:
            log_epi.write('{}\t {}\t {}\t {}\n'.format(i_episode+1, episode_reward, episode_timesteps, best_model))
            log_epi.flush()
            if total_timesteps >= args.start_timesteps:
                log_step.write('{}\t {}\n'.format(total_timesteps+1, episode_reward))
                log_step.flush()

            # Reset environment:
            state, done = env.reset(), False
            i_episode += 1 
            episode_reward = 0
            episode_timesteps = 0
            if args.save_model: 
                policy.save(f"./models/{file_name}")

if __name__ == '__main__':
    main()