import datetime
import random
import time
from itertools import count

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from agents import TDAgent, RandomAgent, evaluate_agents
from agents import hr_obs
from gym_backgammon.envs.backgammon import WHITE, BLACK

torch.set_default_tensor_type('torch.DoubleTensor')

verbose = False

class BaseModel(nn.Module):
    def __init__(self, lr, lamda, seed=123):
        super(BaseModel, self).__init__()
        self.lr = lr
        self.lamda = lamda  # trace-decay parameter
        self.start_episode = 0

        self.optimizer = None

        torch.manual_seed(seed)
        random.seed(seed)

    def update_weights(self, p, p_next):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def init_weights(self):
        raise NotImplementedError

    def checkpoint(self, checkpoint_path, step, name_experiment):
        path = checkpoint_path + "/{}_{}_{}.tar".format(name_experiment, datetime.datetime.now().strftime('%Y%m%d_%H%M_%S_%f'), step + 1)
        torch.save({'step': step + 1, 'model_state_dict': self.state_dict()}, path)
        print("\nCheckpoint saved: {}".format(path))


    # ** here!
    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        self.start_episode = checkpoint['step']

        self.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train_batch(self, batch_obs_updates, batch_r_updates, randomize_batch):
        if verbose:
            print(f"as in train_batch, size of obs_up: ", len(batch_obs_updates), ", size of r_upd: ", len(batch_r_updates))

        if randomize_batch:
            batch_tot = batch_obs_updates + batch_r_updates
            batch_size = len(batch_tot)
            if verbose:
                print(f"as r1, {batch_size=}")

                for t in range (batch_size):
                    (o_in, o_out) = batch_tot[t]
                    out_is_r = isinstance(o_out, (float, int))

                    if out_is_r:
                        print(f"{t=} ", hr_obs(o_in, "input_obs= "), f" {o_out=}, {out_is_r=}")
                    else:
                        print(f"{t=} ", hr_obs(o_in, "input_obs= "), hr_obs(o_out, "output_obs= "), f" {out_is_r=}")

                print("\n\n\n")
                    
            for t in range (batch_size):
                rand_samp = random.randrange(batch_size)
                (o_in, o_out) = batch_tot[rand_samp]
                out_is_r = isinstance(o_out, (float, int))
                if verbose:
                    print(f"{rand_samp=} {out_is_r=}")
                if out_is_r:
                    reward = o_out
                    t_in = self(o_in)
                    loss = self.update_weights(t_in, reward)
                    if verbose:
                        print(f"as: {reward=}", hr_obs(o_in, "o_in="))
                        print(f"as r3: {loss=}")
                    t_in = None
                else:
                    t_in = self(o_in)
                    t_out = self(o_out)
                    loss = self.update_weights(t_in, t_out)
                    if verbose:
                        print("as batch, ", hr_obs(o_in, "input_obs= "), hr_obs(o_out, "output_obs= "))
                        print(f"as r4: {loss=}")
                    t_in = None
                    t_out = None
            return
        
        for (o_in, o_out) in batch_obs_updates:
            #print("as batch, ", hr_obs(o_in, "input_obs= "), hr_obs(o_out, "output_obs= "))
            t_in = self(o_in)
            t_out = self(o_out)
            #print(f"as: {t_in=} {t_out=}")
            loss = self.update_weights(t_in, t_out)
            #print(f"as: {loss=}")

        t_in = None
        t_out = None

        for (o_in, reward) in batch_r_updates:
            #print(f"as: {reward=}", hr_obs(o_in, "o_in="))
            t_in = self(o_in)
            loss = self.update_weights(t_in, reward)
            #print(f"as: {loss=}")
            t_in = None

            
    def train_agent(self, env, n_episodes, save_path=None, save_step=0, name_experiment=''):

        cuda_avail = torch.cuda.is_available()
        print(f"{cuda_avail=}")
        
        batch_train = True
        randomize_batch = True
        batch_train_num_games = 8  # number of games to wait until training
        batch_obs_updates = []
        batch_r_updates = []
                
        start_episode = self.start_episode
        n_episodes += start_episode

        wins = {WHITE: 0, BLACK: 0}
        network = self

        agents = {WHITE: TDAgent(WHITE, net=network), BLACK: TDAgent(BLACK, net=network)}

        durations = []
        steps = 0
        start_training = time.time()
        
        for episode in range(start_episode, n_episodes):

            if batch_train and (episode % batch_train_num_games == 0):
                if verbose:
                    print(f"as bt {episode=} {batch_train_num_games=}")

                self.train_batch(batch_obs_updates, batch_r_updates, randomize_batch)
                
                batch_obs_updates = []
                batch_r_updates = []
            
            agent_color, first_roll, observation = env.reset()
            agent = agents[agent_color]

            t = time.time()

            for i in count():
                
                if first_roll:
                    roll = first_roll
                    first_roll = None
                else:
                    roll = agent.roll_dice()

                if not batch_train:
                    p = self(observation)

                actions = env.get_valid_actions(roll)
                action = agent.choose_best_action(actions, env)
                observation_next, reward, done, winner = env.step(action)

                if not batch_train:
                    p_next = self(observation_next)

                if done:
                    #print (f"as: done with game {winner=}")
                    if winner is not None:
                        if not batch_train:
                            loss = self.update_weights(p, reward)
                        else:
                            batch_r_updates.append((observation, reward))

                        wins[agent.color] += 1

                    tot = sum(wins.values())
                    tot = tot if tot > 0 else 1

                    print("Game={:<6d} | Winner={} | after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(episode + 1, winner, i,
                        agents[WHITE].name, wins[WHITE], (wins[WHITE] / tot) * 100,
                        agents[BLACK].name, wins[BLACK], (wins[BLACK] / tot) * 100, time.time() - t))

                    durations.append(time.time() - t)
                    steps += i
                    
                    break
                else:
                    if not batch_train:
                        loss = self.update_weights(p, p_next)
                    else:                        
                        batch_obs_updates.append((observation, observation_next))


                # training over, clear p, and p_next
                p = None
                p_next = None

                agent_color = env.get_opponent_agent()
                agent = agents[agent_color]

                observation = observation_next

            if save_path and save_step > 0 and episode > 0 and (episode + 1) % save_step == 0:
                self.checkpoint(checkpoint_path=save_path, step=episode, name_experiment=name_experiment)
                agents_to_evaluate = {WHITE: TDAgent(WHITE, net=network), BLACK: RandomAgent(BLACK)}
                evaluate_agents(agents_to_evaluate, env, n_episodes=20)
                print()

        # finished all episodes one final batch training
        if batch_train:
            self.train_batch(batch_obs_updates, batch_r_updates, randomize_batch)
            batch_obs_updates = []
            batch_r_updates = []


        print("\nAverage duration per game: {} seconds".format(round(sum(durations) / n_episodes, 3)))
        print("Average game length: {} plays | Total Duration: {}".format(round(steps / n_episodes, 2), datetime.timedelta(seconds=int(time.time() - start_training))))

        if save_path:
            self.checkpoint(checkpoint_path=save_path, step=n_episodes - 1, name_experiment=name_experiment)

            with open('{}/comments.txt'.format(save_path), 'a') as file:
                file.write("Average duration per game: {} seconds".format(round(sum(durations) / n_episodes, 3)))
                file.write("\nAverage game length: {} plays | Total Duration: {}".format(round(steps / n_episodes, 2), datetime.timedelta(seconds=int(time.time() - start_training))))

        env.close()


class TDGammonCNN(BaseModel):
    def __init__(self, lr, seed=123, output_units=1):
        super(TDGammonCNN, self).__init__(lr, seed=seed, lamda=0.7)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),  # CHANNEL it was 3
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.hidden = nn.Sequential(
            nn.Linear(64 * 8 * 8, 80),
            nn.Sigmoid()
        )

        self.output = nn.Sequential(
            nn.Linear(80, output_units),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def init_weights(self):
        pass

    def forward(self, x):
        # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
        x = np.dot(x[..., :3], [0.2989, 0.5870, 0.1140])
        x = x[np.newaxis, :]
        x = torch.from_numpy(np.array(x))
        x = x.unsqueeze(0)
        x = x.type(torch.DoubleTensor)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64 * 8 * 8)
        x = x.reshape(-1)
        x = self.hidden(x)
        x = self.output(x)
        return x

    def update_weights(self, p, p_next):

        if isinstance(p_next, int):
            p_next = torch.tensor([p_next], dtype=torch.float64)

        loss = self.loss_fn(p_next, p)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


class TDGammon(BaseModel):
    def __init__(self, hidden_units, lr, lamda, init_weights, seed=123, input_units=198, output_units=1):
        super(TDGammon, self).__init__(lr, lamda, seed=seed)

        self.hidden = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.Sigmoid()
        )

        # self.hidden2 = nn.Sequential(
        #     nn.Linear(hidden_units, hidden_units),
        #     nn.Sigmoid()
        # )

        # self.hidden3 = nn.Sequential(
        #     nn.Linear(hidden_units, hidden_units),
        #     nn.Sigmoid()
        # )

        self.output = nn.Sequential(
            nn.Linear(hidden_units, output_units),
            nn.Sigmoid()
        )

        if init_weights:
            self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            nn.init.zeros_(p)

    def forward(self, x):
        x = torch.from_numpy(np.array(x))
        x = self.hidden(x)
        # x = self.hidden2(x)
        # x = self.hidden3(x)
        x = self.output(x)
        return x

    def update_weights(self, p, p_next):
        # reset the gradients
        self.zero_grad()

        # compute the derivative of p w.r.t. the parameters
        p.backward()

        with torch.no_grad():

            td_error = p_next - p

            # get the parameters of the model
            parameters = list(self.parameters())

            for i, weights in enumerate(parameters):
                # w <- w + alpha * td_error * (grad w w.r.t P_t)
                new_weights = weights + self.lr * td_error * weights.grad
                weights.copy_(new_weights)

        return td_error
