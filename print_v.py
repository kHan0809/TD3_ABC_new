import copy

import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils
import TD3_BC
import time


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--env", default="hopper-medium-expert-v2")        # OpenAI gym environment name
	parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps",   default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--max_timesteps_Q", default=3e4, type=int, help="3e4")  # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=1, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--clutch", default=95e4, type=int)  # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=True)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha
	}

	# Initialize policy
	policy_v       = TD3_BC.TD3_BC(**kwargs)
	policy_v_prime = TD3_BC.TD3_BC(**kwargs)
	policy_v.load_v("v")
	policy_v_prime.load_v("v_prime")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(env.get_dataset())
	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1

	replay_buffer_dot = copy.deepcopy(replay_buffer)
	replay_buffer_dot.load_replay()


	# total  = 1311000
	# batch_size = 1000
	# id = np.arange(total)
	# acc, acc_prime = 0., 0.
	# for i in range(int(total/batch_size)):
	# 	if i != int(total/batch_size) - 1:
	# 		batch = id[i*batch_size:(i+1)*batch_size]
	# 	else:
	# 		batch = id[i * batch_size:]
	# 	state, action, next_state, reward, not_done, Return = replay_buffer_dot.get_sample_idx(batch)
	# 	adv = policy_v.test_adv2(state, Return)
	# 	adv_prime = policy_v_prime.test_adv2(state, Return)
	# 	diff = adv_prime - adv
	# 	w = np.clip(np.exp(adv/5)-1., 0., 0.5)
	# 	w_prime = np.clip(np.exp(adv_prime/20), 0., 0.5)
	# 	acc += np.sum(w>0)
	# 	acc_prime += np.sum(w_prime>0)
	# 	print("[steps] : %d | diff_w : %.2f" % (i + 1, np.sum(w_prime-w)))
	# print("Final : %d | %d | %d"%(acc, acc_prime, acc-acc_prime))

	# total  = 1990000
	# batch_size = 1000
	# id = np.arange(total)
	# acc = 0.
	# for i in range(int(total/batch_size)):
	# 	if i != int(total/batch_size) - 1:
	# 		batch = id[i*batch_size:(i+1)*batch_size]
	# 	else:
	# 		batch = id[i * batch_size:]
	# 	state, action, next_state, reward, not_done, Return = replay_buffer.get_sample_idx(batch)
	# 	adv = policy_v.test_adv2(state, Return)
	# 	adv_prime = policy_v_prime.test_adv2(state, Return)
	# 	diff = adv_prime - adv
	# 	w = np.clip(np.exp(adv/5)-1., 0., 0.5)
	# 	w_prime = np.clip(np.exp((adv_prime+10.6285) / 5) - 1., 0., 0.5)
	# 	w_diff = np.sum(w_prime - w)
	# 	acc += w_diff
	# 	print("[steps] : %d | diff_w : %.2f" % (i + 1, w_diff))
	# print("Final : ", acc)

	# total = 1990000
	# batch_size = 1000
	# id = np.arange(total)
	# for i in range(total):
	# 	state, action, next_state, reward, not_done, Return = replay_buffer.sample(1)
	# 	adv = policy_v.test_adv2(state, Return)
	# 	adv_prime = policy_v_prime.test_adv2(state, Return)
	# 	diff = adv_prime - adv
	# 	w = np.clip(np.exp(adv / 5) - 1., 0., 0.5)
	# 	w_prime = np.clip(np.exp(2.1257)*np.exp(adv_prime/ 5) - 1., 0., 0.5)
	# 	# w = np.exp(adv / 5)
	# 	# w_prime = np.exp(adv_prime / 5)
	# 	print("[steps] : %d | v : %.2f | v' : %.2f | diff : %.2f | w_diff : %.2f"%(i+1, adv, adv_prime, diff, w_prime-w))

	total = 1990000
	batch_size = 1000
	id = np.arange(total)
	for i in range(total):
		state, action, next_state, reward, not_done, Return = replay_buffer.sample(1)
		adv = policy_v.test_adv2(state, Return)
		adv_prime = policy_v_prime.test_adv2(state, Return)
		diff = adv_prime - adv
		w = np.clip(np.exp(adv / 5) - 1., 0., .5)
		w_prime = np.clip(np.exp(adv_prime/5), 0., .5)
		# w = np.exp(adv / 5)
		# w_prime = np.exp(adv_prime / 5)
		if w == 0 and w_prime > 0:
			print("[steps] : %d | v : %.2f | v' : %.2f" % (i + 1, w, w_prime))