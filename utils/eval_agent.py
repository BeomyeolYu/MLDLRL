import torch
import gym
import numpy as np
from datetime import datetime

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_agent(policy, env_name, seed, eval_episodes=30, max_steps=2000):
    eval_env = gym.make(env_name)
    eval_env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        episode_timesteps = 0
        action = np.array([0.0, 0.0, 0.0, 0.0])

        is_TEST = True
        if is_TEST:
            action_list = []
            state_list  = []
            goal_list   = []
            
            H_action = np.array([0.0, 0.0, 0.0])
            L_action = np.array([0.0, 0.0, 0.0])
            H_goal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            L_goal = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

            TEST_list = []
            TEST = []

        while not done:
            episode_timesteps += 1

            # Actual quadrotor states
            x = np.array([state[0], state[1], state[2]])
            v = np.array([state[3], state[4], state[5]])
            R_vec = np.array([state[6],  state[7],  state[8],
                              state[9],  state[10], state[11],
                              state[12], state[13], state[14]])
            R = R_vec.reshape(3, 3, order='F')
            W = np.array([state[15], state[16], state[17]])

            # Compute theta
            theta_x = np.arctan2(state[1], state[0]) 

            # Imaginary quadrotor states
            _x = R_e3(-theta_x) @ x
            _v = R_e3(-theta_x) @ v
            _R = R_e3(-theta_x) @ R
            _R_vec = _R.reshape(1, 9, order='F')
            _state = np.concatenate((_x[0], _x[2], _v, _R_vec, W), axis=None)

            action_past = action
            # Select action according to policy
            action = policy.select_action(np.array(_state))

            # Perform action
            state, reward, done, _ = eval_env.step(action)

            # rewards
            C_R = 0.05
            Rd = np.eye(3)
            _Rd = R_e3(-theta_x) @ Rd
            reward -= C_R * (abs(_R[0][0]-_Rd[0][0]) + abs(_R[1][0]-_Rd[1][0]))  
            # = C_R * linalg.norm(R - Rd, 2)            
            if episode_timesteps == 1:
                C_A = 0.0 
            else:
                C_A = 0.03 # for smooth control
            reward -= C_A * (abs(action_past - action)).sum()
            reward = np.interp(reward, [0.0, 2.0], [0.0, 1.0])
            reward *= 0.1

            avg_reward += reward

            # Save data:
            if is_TEST:
                action_list.append(np.concatenate((action, H_action, L_action), axis=None))
                state_list.append(state)
                goal_list.append(np.concatenate((H_goal, L_goal), axis=None))
                if TEST == []:
                    TEST = np.array(np.zeros(4))
                TEST_list.append(TEST)

            if episode_timesteps == max_steps:
                done = True

        # Save data
        if is_TEST:
            min_len = min(len(action_list), len(state_list), len(goal_list))
            log_data = np.column_stack((action_list[-min_len:], state_list[-min_len:], goal_list[-min_len:], TEST_list[-min_len:]))
            header = "Actions and States\n"
            header += "actions, state[0], ..., state[n], goal[0], ..., goal[n]" 
            time_now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            np.savetxt('log_'+time_now+'.dat', log_data, header=header, fmt='%.10f') 

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} Episodes: {avg_reward:.3f} Error: {np.round(state[0:3]*3.0, 5)} ") 
    print("---------------------------------------")
    return avg_reward