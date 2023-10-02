import random
import time
from itertools import count
from random import randint, choice

import numpy as np
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS

random.seed(0)


# AGENT ============================================================================================


class Agent:
    def __init__(self, color):
        self.color = color
        self.name = 'Agent({})'.format(COLORS[color])

    def roll_dice(self):
        return (-randint(1, 6), -randint(1, 6)) if self.color == WHITE else (randint(1, 6), randint(1, 6))

    def choose_best_action(self, actions, env):
        raise NotImplementedError


# RANDOM AGENT =======================================================================================


class RandomAgent(Agent):
    def __init__(self, color):
        super().__init__(color)
        self.name = 'RandomAgent({})'.format(COLORS[color])

    def choose_best_action(self, actions, env):
        return choice(list(actions)) if actions else None


# HUMAN AGENT =======================================================================================


class HumanAgent(Agent):
    def __init__(self, color):
        super().__init__(color)
        self.name = 'HumanAgent({})'.format(COLORS[color])

    def choose_best_action(self, actions=None, env=None):
        pass


# TD-GAMMON AGENT =====================================================================================


class TDAgent(Agent):
    def __init__(self, color, net):
        super().__init__(color)
        self.net = net
        self.name = 'TDAgent({})'.format(COLORS[color])

    def choose_best_action(self, actions, env):
        best_action = None

        if actions:
            values = [0.0] * len(actions)
            tmp_counter = env.counter
            env.counter = 0
            state = env.game.save_state()

            # Iterate over all the legal moves and pick the best action
            for i, action in enumerate(actions):
                observation, reward, done, info = env.step(action)
                values[i] = self.net(observation)

                # restore the board and other variables (undo the action)
                env.game.restore_state(state)

            # practical-issues-in-temporal-difference-learning, pag.3
            # ... the network's output P_t is an estimate of White's probability of winning from board position x_t.
            # ... the move which is selected at each time step is the move which maximizes P_t when White is to play and minimizes P_t when Black is to play.

            #print ("as: values = " + str(values))
            #print ("as: values type = " + str(type(values)))

            dvalues = [v.detach() for v in values]

            #print ("as: dvalues = " + str(dvalues))
            #print ("as: dvalues type = " + str(type(dvalues)))
            
            
            #best_action_index = int(np.argmax(values)) if self.color == WHITE else int(np.argmin(values))
            # as modified below to use dvalus
            best_action_index = int(np.argmax(dvalues)) if self.color == WHITE else int(np.argmin(dvalues))
            best_action = list(actions)[best_action_index]
            #print ("as : best action index: " + str(best_action_index))
            #print ("as : best action : " + str(best_action))
            env.counter = tmp_counter

        return best_action


# TD-GAMMON AGENT (play against gnubg) ================================================================


class TDAgentGNU(TDAgent):

    def __init__(self, color, net, gnubg_interface):
        super().__init__(color, net)
        self.gnubg_interface = gnubg_interface

    def roll_dice(self):
        gnubg = self.gnubg_interface.send_command("roll")
        return self.handle_opponent_move(gnubg)

    def choose_best_action(self, actions, env):
        best_action = None

        if actions:
            game = env.game
            values = [0.0] * len(actions)
            state = game.save_state()

            for i, action in enumerate(actions):
                game.execute_play(self.color, action)
                opponent = game.get_opponent(self.color)
                observation = game.get_board_features(opponent) if env.model_type == 'nn' else env.render(mode='state_pixels')
                values[i] = self.net(observation)

                
                game.restore_state(state)

            best_action_index = int(np.argmax(values)) if self.color == WHITE else int(np.argmin(values))
            best_action = list(actions)[best_action_index]

        return best_action

    def handle_opponent_move(self, gnubg):
        # Once I roll the dice, 2 possible situations can happen:
        # 1) I can move (the value gnubg.roll is not None)
        # 2) I cannot move, so my opponent rolls the dice and makes its move, and eventually ask for doubling, so I have to roll the dice again

        # One way to distinguish between the above cases, is to check the color of the player that performs the last move in gnubg:
        # - if the player's color is the same as the TD Agent, it means I can send the 'move' command (no other moves have been performed after the 'roll' command) - case 1);
        # - if the player's color is not the same as the TD Agent, this means that the last move performed after the 'roll' is not of the TD agent - case 2)
        previous_agent = gnubg.agent
        if previous_agent == self.color:  # case 1)
            return gnubg
        else:  # case 2)
            while previous_agent != self.color and gnubg.winner is None:
                # check if my opponent asks for doubling
                if gnubg.double:
                    # default action if the opponent asks for doubling is 'take'
                    gnubg = self.gnubg_interface.send_command("take")
                else:
                    gnubg = self.gnubg_interface.send_command("roll")
                previous_agent = gnubg.agent
            return gnubg


def evaluate_agents(agents, env, n_episodes):
    wins = {WHITE: 0, BLACK: 0}

    for episode in range(n_episodes):

        agent_color, first_roll, observation = env.reset()
        agent = agents[agent_color]

        print ("gedol repo, agent first color: ", agent_color)
        
        t = time.time()

        as_move_strs = []
        as_move_pstrs = []
        
        for i in count():

            if first_roll:
                roll = first_roll
                first_roll = None
            else:
                roll = agent.roll_dice()

            actions = env.get_valid_actions(roll)
            action = agent.choose_best_action(actions, env)
            observation_next, reward, done, winner = env.step(action)

            #print ("observation: ", observation)
            #print ("agent color: ", agent_color)
            #print ("agent: ", agent)
            #print ("roll: ", roll)
            #print ("action: ", action) # action is tuple, immuttable
            saction = []
            if action is None:
                saction.append([None, None])
            else:
                for p in action:
                    # hack
                    increasing = True
                    if p[0] == "bar":
                        if p[1] > 17:
                            increasing = False
                    else:        
                        if (p[0] > p[1]):
                            increasing = False
                    #print ("increasing", increasing)
                    new_start = "bar"
                    new_end = ""
                    if (p[0] != "bar"):
                        if increasing:
                            new_start = 24 - p[0]
                            new_end = 24 - p[1]
                        else:
                            new_start = 1 + p[0]
                            new_end = 1 + p[1]
                    else:
                        if increasing:
                            new_end = 24 - p[1]
                        else:
                            new_end = 1 + p[1]
                    saction.append([new_start, new_end])

            #print ("saction: ", saction) # saction is list, muttable
            sroll = None
            if (roll[0] < 0):
                sroll = str(-roll[0]) + str(-roll[1])
            else:
                sroll = str(roll[0]) + str(roll[1])
            #print ("sroll: ", sroll)
            as_move_pstr = sroll + ": "
            if saction is not None:
                for sa in saction:
                    as_move_pstr += str(sa[0]) + "/" + str(sa[1]) + " "
            #print ("as_move_pstr", as_move_pstr)
            as_move_pstrs.append(as_move_pstr)
            if (len(as_move_pstrs) == 2):
                as_move_strs.append (as_move_pstrs[0] + "\t" + as_move_pstrs[1])
                as_move_pstrs = []

            if done:
                #print("done!")
                if (len(as_move_pstrs) == 1):
                    as_move_strs.append(as_move_pstrs[0])                    
                elif (len(as_move_pstrs) > 1):
                    raise Exception ("as: e1")
                #print("as_move_strs: ", as_move_strs)

                as_move_count = 0
                as_final_game_str = "as new game:\n"
                for as_ms in as_move_strs:
                    as_final_game_str += (str(as_move_count) + ") " + as_ms) + "\n"
                    as_move_count += 1

                print (as_final_game_str)
                
                if winner is not None:
                    wins[agent.color] += 1
                tot = wins[WHITE] + wins[BLACK]
                tot = tot if tot > 0 else 1

                print("EVAL => Game={:<6d} | Winner={} | after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(episode + 1, winner, i,
                    agents[WHITE].name, wins[WHITE], (wins[WHITE] / tot) * 100,
                    agents[BLACK].name, wins[BLACK], (wins[BLACK] / tot) * 100, time.time() - t))
                break

            agent_color = env.get_opponent_agent()
            agent = agents[agent_color]

            observation = observation_next
    return wins
