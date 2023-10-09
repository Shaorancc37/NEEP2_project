import gym
import time
import sys
import numpy as np
import time
import torch
import torch.nn as nn
import math
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.distributions.categorical import Categorical
import multiprocessing as mp


class PPO:
    def __init__(self, actor, critic, problem, config_init, config_wandb, **hyperparameters):
        """
            初始化PPO

            Parameters:
                actor - actor model
                critic - critic model
                env - the environment
                hyperparameters - 超参数

            Returns:
                None
        """
        # # Make sure the environment is compatible with our code
        # assert (type(env.observation_space) == gym.spaces.Box)
        # assert (type(env.action_space) == gym.spaces.Box)

        # 初始化超参数
        self.env = None
        self._init_hyperparameters(hyperparameters)

        # 计算机的核数量
        self.num_cores = int(mp.cpu_count())
        print("该机器共有: " + str(self.num_cores) + " 核心可以并行")
        print("本次使用{}核心并行计算".format(self.multiprocess_num))
        # 构建问题
        self.problem = problem
        self.config_init = config_init
        self.config_wandb = config_wandb
        # self.obs_dim = env.observation_space.shape[0]
        # self.act_dim = env.action_space.shape[0]
        # print(self.obs_dim)
        # print(self.act_dim)

        # Initialize actor and critic networks
        # self.actor = actor(self.obs_dim, self.act_dim)  # ALG STEP 1
        # self.critic = critic(self.obs_dim, 1)

        # 定义actor和critic model
        self.actor = actor(input_size=self.input_size, output_dim=self.output_size, LSTM_hidden_size=self.hidden_size,
                           LSTM_num_layer=self.num_layer,
                           LSTM_time_step=self.time_step, device=self.device).to(self.device)
        self.critic = critic(input_size=self.input_size, output_dim=int(1), LSTM_hidden_size=self.hidden_size,
                             LSTM_num_layer=self.num_layer,
                             LSTM_time_step=self.time_step, device=self.device).to(self.device)
        # 初始化优化器
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # 初始化采样的协方差矩阵
        # 为了训练的稳定，使用的固定的协方差矩阵，事实上更好的效果是通过actor自学习一个协方差矩阵。
        self.cov_var = torch.full(size=(self.output_size,), fill_value=0.03)
        # print(self.cov_var)
        self.cov_mat = torch.diag(self.cov_var)
        # print(self.cov_mat)

        # 记录每个iteration的信息
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
        }

    def learn(self, total_timesteps):
        """
            PPO的主要训练过程

            Parameters:
                total_timesteps - 总的时间步，在这里并没有用，因为定义的是episode的长度。其他地方或者日后的多任务多维度用时间步来取平均

            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0
        i_so_far = 0
        while t_so_far < total_timesteps:  # ALG STEP 2
            # 收集一批的数据：一批obs， act， logprob， 处理后的奖励， 每批的长度
            start_time = time.time()
            batch_obs, batch_last_para, batch_acts, batch_log_probs, batch_rtgs, batch_lens, = self.rollout()  # ALG STEP 3
            # batch_obs = batch_obs.to(self.device)
            # batch_last_para = batch_last_para.to(self.device)
            # batch_acts = batch_acts.to(self.device)
            # batch_log_probs = batch_log_probs.to(self.device)
            # batch_rtgs = batch_rtgs.to(self.device)

            end_time = time.time()
            print("收集数据的时间：{}".format(end_time - start_time))

            # 计算总的时间步消耗和epoch
            t_so_far += np.sum(batch_lens)
            i_so_far += 1

            # Logging
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # 计算当前代的优势函数值，（R-V）
            start_time = time.time()
            V, _ = self.evaluate(batch_obs, batch_last_para, batch_acts)
            A_k = batch_rtgs - V.detach()  # ALG STEP 5

            # 为了减轻奖励不均的影响，标准化一下
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            end_time = time.time()
            print("计算优势的时间：{}".format(end_time - start_time))

            # 每个epoch更新self.n_updates_per_iteration轮
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # 计算 V_phi 和 pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_last_para, batch_acts)

                # 计算比率 pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                start_time = time.time()
                # loss
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # 更新actor
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # 更新critic
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                end_time = time.time()
                print("更新一次的时间：{}".format(end_time - start_time))

                # log
                self.logger['actor_losses'].append(actor_loss.detach())

            # 打印结果
            self._log_summary()

            # self.actor.to(self.device)
            # self.critic.to(self.device)

            # 保存模型
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def rollout_ori(self):
        """
            数据采集

            Parameters:
                None

            Return:
                batch_obs - 一批观测值. Shape: (number of timesteps, dimension of observation)
                batch_acts - 一批动作. Shape: (number of timesteps, dimension of action)
                batch_log_probs - 一批log——prob. Shape: (number of timesteps)
                batch_rtgs - 一批折扣后的奖励. Shape: (number of timesteps)
                batch_last_para - 返回的当代的四个参数
                batch_lens - 一批的长度. Shape: (number of episodes)
        """
        # 一批数据
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        batch_last_para = []

        t = 0  # 记录当前的episode

        # 每个batch，episode_per_batch个episode
        while t < self.episode_per_batch:
            ep_rews = []  # rewards collected per episode

            # 每个episode开始，重置环境
            self.env = OPTGAN_ENV(seed=t, problem=self.problem, config_init=self.config_init,
                                  config_wandb=self.config_wandb)
            (obs, last_para) = self.env.reset()
            # from numpy 转换成tensor
            # obs = torch.tensor(obs)
            # last_para = torch.tensor(last_para)
            done = False

            t += 1

            # 执行一次episode
            ep_t = 0
            while True:
                ep_t += 1

                # 收集观测值，两部分：obs（历史的最优集）和上代的参数
                batch_obs.append(obs)
                batch_last_para.append(last_para)

                # 计算action，并更新环境
                action, log_prob = self.get_action(obs, last_para)
                # print(action)
                (obs, last_para), rew, done, _ = self.env.step(action)

                # 记录奖励，动作，logprob
                ep_rews.append(rew)
                batch_acts.append(torch.tensor(action))
                batch_log_probs.append(log_prob)

                if done:
                    break

            # batch_obs, ep_rews, batch_acts, batch_log_probs, ep_t = self.one_eposide()
            # t += (ep_t + 1)

            ep_rews = (np.array(ep_rews) / (ep_t + 1)).tolist()  # 修正这一轮的奖励

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # print(len(batch_acts))
        # print(len(batch_obs))
        # print(type(batch_acts))
        # print(type(batch_obs))
        # print(batch_acts)
        # print(batch_obs)
        #

        batch_obs = torch.stack(batch_obs)
        batch_last_para = torch.stack(batch_last_para)

        batch_acts = torch.stack(batch_acts)
        batch_log_probs = torch.stack(batch_log_probs)

        # print(batch_obs.size())
        # print(batch_last_para.size())
        # print(batch_acts.size())
        # print(batch_log_probs.size())

        # batch_obs = np.array(batch_obs)
        # batch_acts = np.array(batch_acts)
        # batch_last_para = np.array(batch_last_para)
        # print(type(batch_acts))
        # print(type(batch_obs))
        # # print(batch_acts)
        # print(batch_obs.shape)
        #
        # batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        # batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        # batch_last_para = torch.tensor(batch_last_para, dtype=torch.float)
        # batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

        # Log
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_last_para, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def rollout_chuan(self):
        """
            数据采集

            Parameters:
                None

            Return:
                batch_obs - 一批观测值. Shape: (number of timesteps, dimension of observation)
                batch_acts - 一批动作. Shape: (number of timesteps, dimension of action)
                batch_log_probs - 一批log——prob. Shape: (number of timesteps)
                batch_rtgs - 一批折扣后的奖励. Shape: (number of timesteps)
                batch_last_para - 返回的当代的四个参数
                batch_lens - 一批的长度. Shape: (number of episodes)
        """
        # 一批数据
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        batch_last_para = []

        t = 0  # 记录当前的episode

        # 每个batch，episode_per_batch个episode
        while t < self.episode_per_batch:
            # print(t)
            epo_obs, epo_last_para, epo_acts, epo_log_probs, epo_rews, epo_t = self.one_eposide(t)
            # print(epo_obs)
            t += 1
            # print(type(epo_obs))
            # print(type(epo_last_para))
            # print(type(epo_acts))
            # print(type(epo_log_probs))
            # print(type(epo_rews))
            # print(type(epo_t))
            batch_obs.append(epo_obs)
            batch_last_para.append(epo_last_para)
            batch_acts.append(epo_acts)
            batch_log_probs.append(epo_log_probs)

            batch_rews.append(epo_rews)
            batch_lens.append(epo_t)

        batch_obs = torch.vstack(batch_obs)
        batch_last_para = torch.vstack(batch_last_para)
        batch_acts = torch.vstack(batch_acts)
        batch_log_probs = torch.hstack(batch_log_probs)

        # print(batch_obs.size())
        #
        # print(batch_last_para.size())
        # print(batch_acts.size())
        # print(batch_log_probs.size())

        batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

        # Log
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_last_para, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def rollout(self):
        """
            数据采集

            Parameters:
                None

            Return:
                batch_obs - 一批观测值. Shape: (number of timesteps, dimension of observation)
                batch_acts - 一批动作. Shape: (number of timesteps, dimension of action)
                batch_log_probs - 一批log——prob. Shape: (number of timesteps)
                batch_rtgs - 一批折扣后的奖励. Shape: (number of timesteps)
                batch_last_para - 返回的当代的四个参数
                batch_lens - 一批的长度. Shape: (number of episodes)
        """
        # 一批数据
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        batch_last_para = []

        t = 0  # 记录当前的episode
        pool = mp.Pool(self.multiprocess_num)

        # 每个batch，episode_per_batch个episode
        for ttt in np.arange(int(self.episode_per_batch / self.multiprocess_num)):
            results = []
            for t in range(self.multiprocess_num):
                results.append(pool.apply_async(self.one_eposide, args=([int(t + ttt * self.multiprocess_num)])))
            results = [p.get() for p in results]
            for result in results:
                batch_obs.append(result[0])
                batch_last_para.append(result[1])
                batch_acts.append(result[2])
                batch_log_probs.append(result[3])
                batch_rews.append(result[4])
                batch_lens.append(result[5])
        #
        #
        pool.close()

        batch_obs = torch.vstack(batch_obs)
        batch_last_para = torch.vstack(batch_last_para)
        batch_acts = torch.vstack(batch_acts)
        batch_log_probs = torch.hstack(batch_log_probs)

        # print(batch_obs.size())
        #
        # print(batch_last_para.size())
        # print(batch_acts.size())
        # print(batch_log_probs.size())

        batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

        # Log
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_last_para, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def one_eposide_ttt(self, t):
        print(t)
        return np.array([1, 2, 3, 4, 5, 6])

    def one_eposide(self, t):
        t = int(t)
        epo_obs = []
        epo_last_para = []
        epo_acts = []
        epo_log_probs = []
        epo_lens = []
        epo_rews = []  # rewards collected per episode

        # 每个episode开始，重置环境
        self.env = OPTGAN_ENV(seed=t, problem=self.problem, config_init=self.config_init,
                              config_wandb=self.config_wandb)
        (obs, last_para) = self.env.reset()
        # print(obs.size())
        done = False
        # 执行一次episode
        epo_t = 0
        while True:
            epo_t += 1

            # 收集观测值，两部分：obs（历史的最优集）和上代的参数
            epo_obs.append(obs)
            epo_last_para.append(last_para)

            # 计算action，并更新环境
            action, log_prob = self.get_action(obs, last_para)

            # action = torch.rand_like(torch.tensor(action))
            # action = np.clip(action, a_min=0, a_max=1)
            (obs, last_para), rew, done, _ = self.env.step(action)
            # print(type(obs))
            # print(type(last_para))
            # print(type(action))
            # print(type(log_prob))
            # print(type(rew))

            # 记录奖励，动作，logprob
            epo_rews.append(torch.tensor(rew))
            epo_acts.append(torch.tensor(action))
            epo_log_probs.append(log_prob)

            if done:
                break

        epo_rews = (np.array(epo_rews) / (epo_t + 1)).tolist()  # 修正这一轮的奖励
        # print("adadsadad:   ", epo_t)
        return torch.stack(epo_obs), torch.stack(epo_last_para), torch.stack(epo_acts), torch.stack(
            epo_log_probs), epo_rews, int(epo_t)

    def compute_rtgs(self, batch_rews):
        """
            计算折扣奖励

            Parameters:
                batch_rews - 一批奖励, Shape: (number of episodes, number of timesteps per episode)

            Return:
                batch_rtgs - 折扣奖励, Shape: (number of timesteps in batch)
        """
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):

            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs, last_para):
        """
            得到动作

            Parameters:         这两组参数共同构成当代的完整观测值
                obs - 历史最优集
                last_para - 上一代的参数

            Return:
                action -
                log_prob - 在分布中所选动作的对数概率
        """
        # 返回一个动作均值, 方差自定义
        # mean, covar = self.actor(obs)

        mean = self.actor(obs, last_para)
        # print("mean ", mean)
        # cov_mat = torch.diag(covar)
        # dist = MultivariateNormal(mean, cov_mat)

        # 构造动作采样分布M_N（mean, cov_mat）
        dist = MultivariateNormal(mean, self.cov_mat.to(self.device))

        # 动作采样并计算该动作的log概率
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # action = Categorical(logits=mean).sample()
        # log_prob = Categorical(logits=mean).log_prob(action)

        return action.cpu().detach().numpy(), log_prob.cpu().detach()

    def evaluate(self, batch_obs, batch_last_para, batch_acts):
        """
            通过critic评估当前的obs的V值用来（batch_rwt-V）和log prob

            Parameters:
                batch_obs -
                batch_last_para
                batch_acts -

            Return:
                V - critic计算的当前obs的V值
                log_probs - the log prob in given action
        """
        # 从critic中得到V值，用来(batch_rwt-V)
        V = self.critic(batch_obs, batch_last_para).squeeze()

        # 计算动作和log prob
        mean = self.actor(batch_obs, batch_last_para)
        # mean, covar = self.actor(batch_obs)
        # print(mean, covar)
        #
        # cov_mat = torch.diag(covar, )
        # print(cov_mat)
        # dist = MultivariateNormal(mean, cov_mat)
        dist = MultivariateNormal(mean, self.cov_mat.to(self.device))
        log_probs = dist.log_prob(batch_acts)

        # log_probs = Categorical(logits=mean).log_prob(batch_acts)
        return V, log_probs

    def _init_hyperparameters(self, hyperparameters):
        """
            初始化参数

            Parameters:
                hyperparameters - 超参数

            Return:
                None
        """
        # 超参数的定义
        # Algorithm hyperparameters
        self.multiprocess_num = 10  # 多进程并行的数量
        self.episode_per_batch = 3  # 不同于之前的按照timestep记，这次按照总的episode，每次收集这些数据
        self.timesteps_per_batch = 4800  # 每个batch的时间步数量
        self.max_timesteps_per_episode = 1600  # 每个episode最大的时间步
        self.n_updates_per_iteration = 5  # 一代更新actor/critic 几次
        self.lr = 0.005
        self.gamma = 0.95  # PPO的奖励折扣因子
        self.clip = 0.2  # PPO的clip大小

        self.render = True  # 是否渲染
        self.render_every_i = 10  # 多少代渲染一次
        self.save_freq = 10  # 多少代保存一次actor/critic模型
        self.seed = None

        # NN parameters
        self.device = "cpu"
        # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device{}".format(self.device))
        self.input_size = 11  # e.g. dimension + fitness
        self.hidden_size = 256  # LSTM hidden size
        self.output_size = 3  # e.g. optsize, popsize, lambda
        self.num_layer = 2  # LSTM layer
        self.time_step = 150  # e.g. observation num

        # 更新超参数
        for param, val in hyperparameters.items():
            # print(param, val)
            exec('self.' + param + ' = ' + str(val))

        # 设置种子
        if self.seed != None:
            # Check if our seed is valid first
            assert (type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):

        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.cpu().float().mean() for losses in self.logger['actor_losses']])

        #
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

    def train_on_parameter(self, param):
        result = 0
        for num in param:
            result += math.sqrt(num * math.tanh(num) / math.log2(num) / math.log10(num))
        result = [np.random.rand(2), np.random.rand(2), np.random.rand(2)]
        print("asda")
        return result

    def tsa(self):
        for i in np.arange(2):
            start_t = time.time()
            # 核心数量: cpu_count() 函数可以获得计算机的核心数量。
            num_cores = int(mp.cpu_count())
            print("本计算机总共有: " + str(num_cores) + " 核心")

            # 进程池: Pool() 函数创建了一个进程池类，用来管理多进程的生命周期和资源分配。
            #        这里进程池传入的参数是核心数量，意思是最多有多少个进程可以进行并行运算。
            pool = mp.Pool(8)

            param_dict = {'task1': list(range(10, 30000000)),
                          'task2': list(range(30000000, 60000000)),
                          'task3': list(range(60000000, 90000000)),
                          'task4': list(range(90000000, 120000000)),
                          'task5': list(range(120000000, 150000000)),
                          'task6': list(range(150000000, 180000000)),
                          'task7': list(range(180000000, 210000000)),
                          'task8': list(range(210000000, 240000000))}

            # 异步调度: apply_async() 是进程池的一个调度函数。第一个参数是计算函数.第二个参数是需要传入计算函数的参数，这里传入了计算函数名字和计算调参。
            #          异步的意义是在调度之后，虽然计算函数开始运行并且可能没有结束，异步调度都会返回一个临时结果，并且通过列表生成器临时保存在一个列表-results里。
            # for name, param in param_dict.items():
            #     print(name, param)
            results = [pool.apply_async(self.train_on_parameter, args=([param])) for name, param in param_dict.items()]
            # 调度结果: 如果检查列表 results 里的类，会发现 apply_async() 返回的是 ApplyResult，也就是 调度结果类。
            #          简单来说，就是一个用来等待异步结果生成完毕的容器。
            # 获取结果: 调度结果 ApplyResult 类可以调用函数 get(), 这是一个非异步函数，
            #          也就是说 get() 会等待计算函数处理完毕，并且返回结果。
            #          这里的结果就是计算函数的 return。
            results = [p.get() for p in results]
            for result in results:
                print(result[0], result[1], result[2])
            pool.close()

            end_t = time.time()
            elapsed_sec = (end_t - start_t)
            print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")

        # start = time.time()
        # for name, param in param_dict.items():
        #     train_on_parameter(name, param)
        # end = time.time()
        # print("单进程计算 共消耗: " + "{:.2f}".format(end - start) + " 秒")
