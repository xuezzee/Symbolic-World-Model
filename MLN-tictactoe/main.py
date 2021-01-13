import torch
import numpy as np
import argparse
from agent import Agent
from agent_by_myself import MyMLNAgent
# from ACagent import Agent
from TicTacToe import TicTacTeo

def run():
    board = {"win":0, "lose":0, "draw":0}
    args = get_args()
    agent = Agent(args)
    agent.load('./model2/')
    env = TicTacTeo()
    examples = []
    world = []
    for ep in range(args.epoches):
        print("|ep:%d ----------------------------------------------------------------------------|"%ep)
        env.reset()
        state = env.state
        vec_state = env.state2vector(state)
        atom_state = list(env.state2atoms(state))
        done = False
        while not done:
            # print("atom_state",type(atom_state[0].predicate.name))
            act, atom_act, info = agent.choose_action(atom_state, env.get_valid())
            # act = (2,1)
            # atom_act = "Place(2,1)"
            reward, done, placed = env.next_step3(act)
            print("done:", done)
            print("reward:", reward)
            state_ = env.state
            print("action:", act)
            print(state_)
            vec_state_ = env.state2vector(state_)
            atom_state_ = env.state2atoms(state_)
            if placed:
                agent.learn(vec_state, reward, vec_state_, atom_act, info)
            else:
                atom_act = "!" + atom_act
                print("atom_act:", atom_act)
                agent.learn(vec_state, reward, vec_state_, atom_act)
            vec_state = vec_state_
            atom_state = atom_state_
            if done:
                if reward == 1:
                    board["win"] = board["win"] + 1
                elif reward == -1:
                    board["lose"] = board["lose"] + 1
                else:
                    board["draw"] = board["draw"] + 1
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!\nboard:",board)

def run2():
    board = {"win":0, "lose":0, "draw":0}
    args = get_args()
    agent = Agent(args)
    # agent.load('./model2/')
    env = TicTacTeo()
    examples = []
    for ep in range(args.epoches):
        print("|ep:%d ----------------------------------------------------------------------------|"%ep)
        env.reset()
        state = env.state
        vec_state = env.state2vector(state)
        atom_state = list(env.state2atoms(state))
        done = False
        world = []
        while not done:
            print(state)
            # print("atom_state",type(atom_state[0].predicate.name))
            act, atom_act = agent.choose_action(atom_state, env.get_valid())
            # act = (2,1)
            # atom_act = "Place(2,1)"
            reward, done, placed = env.next_step3(act)
            state = env.state
            vec_state_ = env.state2vector(state)
            atom_state_ = env.state2atoms(state)
            world.append(agent.get_world(atom_act))
            agent.learn(world, [vec_state, reward, vec_state_])
            vec_state = vec_state_
            atom_state = atom_state_
            if done:
                if reward == 1:
                    # examples = examples + world
                    agent.learn(world, [vec_state, reward, vec_state_], 0.1)
                    board["win"] = board["win"] + 1
                elif reward == -1:
                    # examples = examples + world
                    agent.learn(world, [vec_state, reward, vec_state_], -0.1)
                    board["lose"] = board["lose"] + 1
                else:
                    # examples = examples + world
                    agent.learn(world, [vec_state, reward, vec_state_], 0.01)
                    board["draw"] = board["draw"] + 1
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!\nboard:",board)

def run3():
    board = {"win":0, "lose":0, "draw":0}
    args = get_args()
    agent = MyMLNAgent()
    # agent.load('./model2/')
    env = TicTacTeo()
    examples = []
    for ep in range(args.epoches):
        print("|ep:%d ----------------------------------------------------------------------------|"%ep)
        env.reset()
        state = env.state
        vec_state = env.state2vector(state)
        atom_state = list(env.state2atoms(state))
        done = False
        world = []
        while not done:
            print(state)
            # print("atom_state",type(atom_state[0].predicate.name))
            exe_act, atom_act, idx_act = agent.choose_action(atom_state)
            # act = (2,1)
            # atom_act = "Place(2,1)"
            reward, done, placed = env.next_step3(exe_act)
            state = env.state
            vec_state_ = env.state2vector(state)
            atom_state_ = env.state2atoms(state)
            world.append(agent.get_world(atom_act))
            # agent.learn(world, [vec_state, reward, vec_state_])
            agent.learn(idx_act)
            vec_state = vec_state_
            atom_state = atom_state_
            if done:
                if reward == 1:
                    # examples = examples + world
                    for w in world:
                        agent.learn(w, [vec_state, reward, vec_state_], 0.1)
                    board["win"] = board["win"] + 1
                elif reward == -1:
                    # examples = examples + world
                    for w in world:
                        agent.learn(w, [vec_state, reward, vec_state_], -0.1)
                    board["lose"] = board["lose"] + 1
                else:
                    # examples = examples + world
                    for w in world:
                        agent.learn(w, [vec_state, reward, vec_state_], 0.01)
                    board["draw"] = board["draw"] + 1
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!\nboard:",board)

def run_AC():
    board = {"win":0, "lose":0, "draw":0}
    ACTION = {0:(0, 0), 1:(0, 1), 2:(0, 2),
              3:(1, 0), 4:(1, 1), 5:(1, 2),
              6:(2, 0), 7:(2, 1), 8:(2, 2)}
    args = get_args()
    agent = Agent(args)
    agent.load('./model2/')
    env = TicTacTeo()
    for ep in range(args.epoches):
        print("|ep:%d ----------------------------------------------------------------------------|"%ep)
        env.reset()
        state = env.state
        vec_state = env.state2vector(state)
        done = False
        while not done:
            # print("atom_state",type(atom_state[0].predicate.name))
            act = agent.choose_action(vec_state)
            # act = (2,1)
            # atom_act = "Place(2,1)"
            reward, done = env.next_step2(ACTION[act])
            state_ = env.state
            # print("action:", act)
            print(state_)
            # print("done:", done)
            # print("reward:", reward)
            vec_state_ = env.state2vector(state_)
            print("value", agent.critic(vec_state_))
            trans = {'s':vec_state, 'r':reward, 's_':vec_state_, 'a':act}
            # if reward == 1 or reward == -1:
            #     print()
            # agent.learn(trans)
            vec_state = vec_state_
            if done:
                if reward == 1:
                    board["win"] = board["win"] + 1
                elif reward == -1:
                    board["lose"] = board["lose"] + 1
                else:
                    board["draw"] = board["draw"] + 1
        print("board:",board)
        print("ratio", board['win']/(board['lose'] + 1e-9))
        if ep % 100 == 0:
            agent.save(path='./model2/')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoches", default=100000)
    parser.add_argument("--dim_s", default=9)
    parser.add_argument("--dim_a", default=9)
    parser.add_argument("--dim_h", default=128)

    parser = parser.parse_args()
    parser.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return parser

if __name__ == '__main__':
    # run()
    # run2()
    run3()
    # run_AC()