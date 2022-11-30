import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils
import TD3_BC
import matplotlib.pyplot as plt
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
	args = parser.parse_args()

	envs = [
		"halfcheetah-random-v2",
		"hopper-random-v2",
		"walker2d-random-v2",
		"halfcheetah-medium-v2",
		"hopper-medium-v2",
		"walker2d-medium-v2",
		"halfcheetah-medium-expert-v2",
		"hopper-medium-expert-v2",
		"walker2d-medium-expert-v2",
		"halfcheetah-medium-replay-v2",
		"hopper-medium-replay-v2",
		"walker2d-medium-replay-v2",
	]
	p_dir = "./results/"
	ext = ".npy"

	for idx,env in enumerate(envs):
		try:
			file_name = f"{args.policy}_{env}_{args.seed}"
			data = np.load(p_dir + file_name + ext)
			plt.subplot(3,5,idx+1)
			plt.plot(data)
			plt.title(env)
			plt.xlabel("time steps")
		except:
			pass
	plt.show()





