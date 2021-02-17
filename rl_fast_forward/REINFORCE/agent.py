from REINFORCE.policy import Policy
from REINFORCE.critic import Critic
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import socket
import math

from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_MODEL_FREQ = 50

class Agent:
    def __init__(self, state_size, action_size, lr=1e-2, critic_lr=1e-4, gamma=1.0, entropy_beta=1e-3):
        np.random.seed(123)
        torch.manual_seed(123)

        self.lr = lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.policy = Policy(state_size=state_size, action_size=action_size).to(device)
        self.critic = Critic(state_size=state_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.coptimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        # self.optimizer = optim.SGD(self.policy.parameters(), lr=self.lr)
        # self.coptimizer = optim.SGD(self.critic.parameters(), lr=self.critic_lr)

        self.creation_timestamp = dt.now().strftime('%Y%m%d_%H%M%S')

        self.entropy_beta = entropy_beta
        self.writer = None

    def train(self, envs, dataset_name, n_epochs, model_path=None):
        self.policy.train()
        exp_keys = np.array(list(envs.keys()))

        hostname = socket.gethostname()
        self.writer = SummaryWriter(log_dir='REINFORCE/logs/{}_reinforce_{}_{}_lr{}_gamma{}_eps{}/'.format(self.creation_timestamp, hostname, dataset_name.lower(), self.lr, self.gamma, n_epochs), filename_suffix='_{}'.format(self.creation_timestamp))

        for exp_name, env in envs.items():
            fig, ax = plt.subplots()
            plot_vec = env.dots_vec
            idxs = np.arange(len(plot_vec))

            ax.plot(idxs, plot_vec)
            self.writer.add_figure('Frames_Dots_{}/{}'.format(dataset_name, exp_name), fig)

        #### Running REINFORCE ####

        scores = {exp_key: [] for exp_key in envs.keys()}
        saved_log_probs = {exp_key: [] for exp_key in envs.keys()}
        # saved_probs = {exp_key: [] for exp_key in envs.keys()}
        saved_entropies = {exp_key: [] for exp_key in envs.keys()}
        rewards = {exp_key: [] for exp_key in envs.keys()}
        biases = {exp_key: 0. for exp_key in envs.keys()}
        avg_epoch_losses = []
        avg_critic_losses = []
        # _probs = np.ndarray((env.num_frames, n_epochs, 3), dtype=np.float32)
        for i_epoch in tqdm(range(n_epochs), desc='Epochs'):

            epoch_losses = []
            critic_losses = []

            # Shuffle the experiments at every new epoch
            np.random.shuffle(exp_keys)

            for exp_key in tqdm(exp_keys, desc='Epoch {}'.format(i_epoch)):  # Run an episode for each environment
                env = envs[exp_key]

                saved_log_probs[exp_key] = []
                # saved_probs[exp_key] = []
                saved_entropies[exp_key] = []
                rewards[exp_key] = []
                state = env.reset()

                saved_actions = []
                s_t_values = []
                saved_states = [state]

                done = False
                while not done:
                    
                    action, log_prob, entropy, probs = self.policy.act(state)

                    s_t_values.append(self.critic.criticize(state))

                    # saved_probs[exp_key].append(probs)
                    saved_log_probs[exp_key].append(log_prob)
                    saved_entropies[exp_key].append(entropy)
                    saved_actions.append(action)
                    state, reward, done, _ = env.step(action)
                    saved_states.append(state)

                    rewards[exp_key].append(reward)

                # scores_deque.append(sum(rewards[exp_key]))
                scores[exp_key].append(sum(rewards[exp_key]))

                discounts = [self.gamma**i for i in range(len(rewards[exp_key]))]
                R_t = np.zeros((len(rewards[exp_key]),), dtype=np.float32)
                R_t[-1] = discounts[-1] * rewards[exp_key][-1]  # We need to set the last reward

                for idx in range(2, len(rewards[exp_key])+1):
                    R_t[-idx] = discounts[-idx] * rewards[exp_key][-idx] + R_t[-idx+1]

                #R_t = (R_t - R_t.mean()) / (R_t.std() + 1e-12)
                R_n = sum(rewards[exp_key])

                H = torch.cat(saved_entropies[exp_key]).sum()  # Entropy Term
                # H = 0

                policy_loss = []
                critic_loss = []
                for idx, (log_prob, r_t) in enumerate(zip(saved_log_probs[exp_key], R_t)):
                    # Policy Gradients using (Rt)
                    # policy_loss.append(log_prob * r_t)

                    # Policy Gradients using (Rt - b)
                    # policy_loss.append(log_prob * (r_t - biases[exp_key]))

                    # Policy Gradients using  (Rt - v(s_t | theta_v)) Actor-Critic style
                    policy_loss.append(log_prob * (r_t - s_t_values[idx].item()) - self.entropy_beta*saved_entropies[exp_key][idx])

                
                J = torch.cat(policy_loss).sum()

                # Actor Loss
                policy_loss = -J

                # Critic Loss
                critic_loss = F.mse_loss(torch.cat(s_t_values), torch.tensor(R_t, dtype=torch.float32, device=device))  # Last state is an outlier, do not use it for loss minimization
                # critic_loss = F.smooth_l1_loss(torch.cat(s_t_values), torch.tensor(R_t, dtype=torch.float32, device=device)) # Last state is an outlier, do not use it for loss minimization

                # Apending losses for future optimization
                if not math.isnan(policy_loss.detach().cpu()):
                    epoch_losses.append(policy_loss.unsqueeze(0))
                critic_losses.append(critic_loss.unsqueeze(0))

                # biases[exp_key] = 0.9 * biases[exp_key] + 0.1 * np.mean(rewards[exp_key]) ### BIAS Term (Update via moving average as in https://github.com/KaiyangZhou/pytorch-vsumm-reinforce/blob/master/main.py)
                biases[exp_key] = np.mean(rewards[exp_key])  # BIAS Term (simple average)

                # Logging to Tensorboard (check the file location for inspection)
                self.writer.add_scalar('Rewards_{}/avg_reward_{}'.format(dataset_name, exp_key), np.mean(scores[exp_key]), i_epoch)
                self.writer.add_scalar('Rewards_{}/curr_reward_{}'.format(dataset_name, exp_key), R_n, i_epoch)
                self.writer.add_scalar('Entropy_{}/curr_entropy_{}'.format(dataset_name, exp_key), H, i_epoch)
                self.writer.add_scalar('Losses_{}/curr_loss_J_{}'.format(dataset_name, exp_key), J.detach().cpu(), i_epoch)
                self.writer.add_scalar('Losses_{}/curr_actor_loss_{}'.format(dataset_name, exp_key), policy_loss.detach().cpu(), i_epoch)
                self.writer.add_scalar('Losses_{}/curr_critic_loss_{}'.format(dataset_name, exp_key), critic_loss.detach().cpu(), i_epoch)

                self.write_selected_frames_image_to_log(env, i_epoch, 'Training_{}'.format(dataset_name), exp_key)

            epoch_losses = torch.cat(epoch_losses).mean()
            critic_losses = torch.cat(critic_losses).mean()
            avg_epoch_losses.append(epoch_losses.detach().cpu())
            avg_critic_losses.append(critic_losses.detach().cpu())

            self.writer.add_scalar('Rewards_{}/_overall_avg_reward'.format(dataset_name), np.mean([scores[exp_key] for exp_key in exp_keys]), i_epoch)
            
            self.optimizer.zero_grad()
            epoch_losses.backward()  # Computes the derivative of loss with respect to theta (dLoss/dTheta)
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0) # <--- Uncomment for grad-clipping
            self.optimizer.step()  # Updates the theta parameters (e.g., theta = theta -lr * dLoss/dTheta in SGD)

            self.coptimizer.zero_grad()
            critic_losses.backward()  # Computes the derivative of loss with respect to theta (dLoss/dTheta)
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0) # <--- Uncomment for grad-clipping
            self.coptimizer.step()  # Updates the theta parameters (e.g., theta = theta -lr * dLoss/dTheta in SGD)

            if i_epoch % SAVE_MODEL_FREQ == SAVE_MODEL_FREQ-1:
                self.save_model(model_path.split('.')[0] + '_epoch{}.pth'.format(i_epoch+1))

        self.save_model(model_path.split('.')[0] + '_epoch{}.pth'.format(i_epoch+1))

    def test(self, env, dataset_name, exp_name):
        self.policy.eval()

        if self.writer is None:
            hostname = socket.gethostname()
            self.writer = SummaryWriter(log_dir='REINFORCE/logs/{}_reinforce_{}_{}_test/'.format(self.creation_timestamp, hostname, dataset_name.lower()), filename_suffix='_{}'.format(self.creation_timestamp))

        fig, ax = plt.subplots()
        plot_vec = env.dots_vec
        idxs = np.arange(len(plot_vec))

        ax.plot(idxs, plot_vec)
        self.writer.add_figure('Frames_Dots_{}/{}'.format(dataset_name, exp_name), fig)

        rewards = []
        state = env.reset()

        done = False
        idx = 0
        with torch.no_grad():
            while not done:
                action, probs = self.policy.argmax_action(state)
                # print(idx, probs.detach().cpu().numpy())
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                idx += 1

        discounts = [self.gamma**i for i in range(len(rewards)+1)]
        
        R = sum([a*b for a, b in zip(discounts, rewards)])

        print('Reward: {:.3f}\nDiscounted Reward: {:.3f}\nNum selected frames: {}'.format(sum(rewards), R, len(env.selected_frames)))

    def write_reward_function_surface_to_log(self, env):

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(0, 30, 3)
        Y = np.arange(0, 30, 3)
        X, Y = np.meshgrid(X, Y)
        Z = env.gaussian(X, env.desired_skip, env.SIGMA, env.LAMBDA_MULTIPLIER) + env.exponential_decay(Y)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        self.writer.add_figure('Reward Function Surface', fig)
        mesh = np.expand_dims(np.array([X, Y, Z]).reshape(3, -1).transpose(1, 0), axis=0)
        colors = np.repeat([[[0, 0, 255]]], mesh.shape[1], axis=1)
        self.writer.add_mesh('Reward Function Surface', vertices=mesh, colors=colors)

    def write_selected_frames_image_to_log(self, env, i_episode, prefix, suffix=None):
        skips = np.array(env.selected_frames[1:]) - np.array(env.selected_frames[:-1])
        fig, ax = plt.subplots()
        ax.scatter(env.selected_frames[:-1], skips)
        if suffix:
            self.writer.add_figure('{}/{}'.format(prefix, suffix), fig, i_episode)
        else:
            self.writer.add_figure('{}'.format(prefix), fig, i_episode)

    def save_model(self, model_path):
        print('[{}] Saving model...'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S')))
        torch.save(self.policy.state_dict(), model_path)
        print('[{}] Done!'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S')))

    def load_model(self, model_path):
        print('[{}] Loading model...'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.policy.load_state_dict(torch.load(model_path))
        print('[{}] Done!'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S')))
