import numpy as np
import torch
import copy
import pickle
class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def addReturn(self,dataset):
		Return_buffer = []
		tmp_return = 0
		for i in reversed(range(self.size)):
			if dataset['terminals'][i] or dataset["timeouts"][i] or i == (self.size - 1):
				tmp_return = dataset["rewards"][i]
				Return_buffer.append(tmp_return)
			else:
				tmp_return = dataset["rewards"][i] + 0.99 * tmp_return
				Return_buffer.append(tmp_return)

		Return_buffer = copy.deepcopy(np.flip(np.array(Return_buffer, dtype=np.float32)))
		self.Return = Return_buffer.reshape(-1,1)

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			torch.FloatTensor(self.Return[ind]).to(self.device),
		)

	def get_sample_idx(self, ind):
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			self.Return[ind],
		)

	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]
		self.addReturn(dataset)

	def change_replay(self,trained_value):
		tensor_state  = torch.FloatTensor(self.state).to(self.device)
		tensor_Return = torch.FloatTensor(self.Return).to(self.device)
		rv = tensor_Return - trained_value(tensor_state)
		idx = torch.where(rv > 0)[0].cpu().detach().numpy()
		print(self.state.shape,self.action.shape,self.next_state.shape,self.reward.shape,self.not_done.shape,self.Return.shape)
		self.state      = self.state[idx]
		self.action     = self.action[idx]
		self.next_state = self.next_state[idx]
		self.reward     = self.reward[idx]
		self.not_done   = self.not_done[idx]
		self.Return     = self.Return[idx]
		self.size      = self.state.shape[0]
		print(self.state.shape, self.action.shape, self.next_state.shape, self.reward.shape, self.not_done.shape,self.Return.shape)

	def save_replay(self):
		replay = dict(state=self.state,action=self.action,next_state=self.next_state,reward=self.reward,not_done=self.not_done,Return=self.Return,size=self.size)
		with open('D_prime.pickle', 'wb') as fw:
			pickle.dump(replay, fw)
	def load_replay(self):
		with open('D_prime.pickle', 'rb') as fr:
			replay_loaded = pickle.load(fr)
		self.state = replay_loaded["state"]
		self.action = replay_loaded["action"]
		self.next_state = replay_loaded["next_state"]
		self.reward = replay_loaded["reward"]
		self.not_done = replay_loaded["not_done"]
		self.Return = replay_loaded["Return"]
		self.size = replay_loaded["size"]


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std