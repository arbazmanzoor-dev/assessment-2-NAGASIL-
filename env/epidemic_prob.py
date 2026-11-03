import EoN
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import numpy as np
import copy
from gym import spaces

# to calculate the integrate of exponential decay
from scipy.integrate import quad
import math


class EpidemicEnv:
    def __init__(self, N, G, H, J, IC, return_statuses, starttime, timestep, stage_num, endtime):
        # definitions of how the epidemic model works
        self.N = N
        self.G = G
        self.H = H
        self.J = J
        self.orig_IC = copy.deepcopy(IC)
        self.IC = IC
        self.return_statuses = return_statuses
        self.timestep = timestep
        self.currtime = 0
        # count of news 2*N, count of state change history 2*N, number of followers N
        # self.observation_space = 5 * self.N
        self.observation_space = np.zeros((5*self.N))
        self.action_space = spaces.Discrete(self.N)
        self.stage_num = stage_num
        self.stage = 0
        self.starttime = starttime
        self.endtime = endtime
        self.reward_range = (0, float("inf"))
        self.metadata = {"render.modes": []}
        
        followers = np.zeros((self.N))
        for u, v in self.G.out_edges():
            followers[v] += 1
        self.followers = followers
        
        # start time after currtime, let fake news propagate till starttime
        # if self.starttime > self.currtime: 
            # graph_hist, IC = EoN.Gillespie_simple_contagion(self.G, self.H, self.J, self.IC, self.return_statuses, tmin=self.currtime, tmax = self.starttime, return_full_data=True)
            
            # self.currtime = self.starttime
            # self.IC = IC
        
    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        
    def reset(self):
        # during simulation, G, H, J will not change, just reset state of node and currtime
        self.IC = copy.deepcopy(self.orig_IC)
        self.currtime = 0
        self.stage = 0
        
        # TODO more currtime to starttime
        # start time after currtime, let fake news propagate till starttime
        if self.starttime > self.currtime: 
            graph_hist, IC = EoN.Gillespie_simple_contagion(self.G, self.H, self.J, self.IC, self.return_statuses, tmin=self.currtime, tmax = self.starttime, return_full_data=True)
            
            self.currtime = self.starttime
            self.IC = IC
            
            return_state = np.zeros((4 * self.N))
            
            # calculate the integral of intensity of every user from self.currtime to self.currtime+self.timestep
            result_state = graph_hist.get_statuses(time=self.currtime)
            
            nodes = list(self.G.nodes)
            for node in nodes:
                if result_state[node] == "R":
                    return_state[node+3*self.N] = 1
                elif result_state[node] == "I":
                    return_state[node+self.N] = 1
                    
            for time_slide in range(self.currtime-self.starttime, self.currtime):
                
                curr_traj = graph_hist.get_statuses(time=time_slide)
                for i in range(self.N):
                    # c_k^f d_k^f
                    if curr_traj[i] == "F":
                        return_state[i] += 1
                        #return_state[i+self.N] += 1
                    # c_k^M d_k^M
                    if curr_traj[i] == "M":
                        return_state[i+2*self.N] += 1
                        #return_state[i+3*self.N] += 1
                            

            return np.concatenate((return_state, self.followers))
        
        else:
            return np.concatenate((np.zeros((4*self.N)), self.followers))
        

    def step(self, action):
        # action is done by modifing self.IC
        self.stage += 1
        
        # DEPRECATED, now will set a user to R without considering the probablity
        # # with the prob to change state info recovered
        # if random.random() < 0.8:
            # # print(action)
            # self.IC[action] = "R"
        
        # NEW SETTING
        self.IC[action] = "R"
        
        
        #t, self.IC, S, EI, ER, I, R = EoN.Gillespie_simple_contagion(G, H, J, self.IC, return_statuses, tmin=self.currtime, tmax = self.currtime + self.timestep)
        
        
        graph_hist, IC = EoN.Gillespie_simple_contagion(self.G, self.H, self.J, self.IC, self.return_statuses, tmin=self.currtime, tmax = self.currtime + self.timestep, return_full_data=True)
        
        # Let's just ignore the FXXKING timestep and make it simple
        # prepare the state to return
        return_state = np.zeros((4 * self.N))
        # prev_traj = None
        for time_slide in range(self.currtime, self.currtime + self.timestep):
            
            curr_traj = graph_hist.get_statuses(time=time_slide)
            for i in range(self.N):
                # c_k^f d_k^f
                if curr_traj[i] == "F":
                    return_state[i] += 1
                    #return_state[i+self.N] += 1
                # c_k^M d_k^M
                if curr_traj[i] == "M":
                    return_state[i+2*self.N] += 1
                    #return_state[i+3*self.N] += 1
        
        final_state = graph_hist.get_statuses(time=self.currtime + self.timestep)
        for i in range(self.N):
            # number of infected users at final stage 
            if final_state[i] == "I":
                return_state[i+self.N] = 1
            elif final_state[i] == "R":
                return_state[i+3*self.N] = 1
                    
            # if prev_traj is None:
                # for i in range(self.N):
                    # # c_k^f d_k^f
                    # if curr_traj[i] == "F":
                        # return_state[i] += 1
                        # return_state[i+self.N] += 1
                    # # c_k^M d_k^M
                    # if curr_traj[i] == "M":
                        # return_state[i+2*self.N] += 1
                        # return_state[i+3*self.N] += 1
                    
            # else:
                # for i in range(self.N):
                    # # c_k^f d_k^f
                    # if curr_traj[i] == "F":
                        # return_state[i] += 1
                        # if curr_traj[i] != prev_traj[i]:
                            # return_state[i+self.N] += 1
                    # # c_k^M d_k^M
                    # if curr_traj[i] == "M":
                        # return_state[i+2*self.N] += 1
                        # if curr_traj[i] != prev_traj[i]:
                            # return_state[i+3*self.N] += 1
            
            # prev_traj = curr_traj
                
            
        
        return_state = np.concatenate((return_state, self.followers))
        
        # prepare the reward to return
        return_reward = 0
        isEnd = False
        if self.stage_num <= self.stage:
            graph_hist, IC = EoN.Gillespie_simple_contagion(self.G, self.H, self.J, IC, self.return_statuses, tmin=self.currtime + self.timestep, tmax = self.endtime, return_full_data=True)
            
            isEnd = True
            final_state = graph_hist.get_statuses(time=self.endtime)
            for i in range(self.N):
                # number of infected users at final stage 
                if final_state[i] == "I":
                    return_reward += 1
                    
            # return_reward = -np.log(float(return_reward)/float(self.N))
            return_reward = self.N - return_reward
            # for i in range(self.N):
                # if final_state[i] == "I":
                    # return_reward += 1
            # if return_reward == 0:
                # return_reward = 1e3
            # else:
                # return_reward = -np.log(float(return_reward)/float(self.N))
            self.currtime = self.endtime
            self.IC = IC
        else:
            # prepare for next step
            self.currtime += self.timestep
            self.IC = IC
        
        # plt.semilogy(t, S, label = 'Susceptible')
        # plt.semilogy(t, EI, label = 'Exposed Infected')
        # plt.semilogy(t, ER, label = 'Exposed Recovered')
        # plt.semilogy(t, I, label = 'Infected')
        # plt.semilogy(t, R, label = 'Recovered')
        # plt.legend()

        # plt.show()

        
        return return_state, return_reward, isEnd, {}
        
        

class EpidemicEnvDecay:
    def __init__(self, N, G, H, J, IC, return_statuses, starttime, timestep, stage_num, endtime):
        # definitions of how the epidemic model works
        self.N = N
        self.G = G
        self.H = H
        self.J = J
        self.orig_IC = copy.deepcopy(IC)
        self.IC = IC
        self.return_statuses = return_statuses
        self.timestep = timestep
        self.currtime = 0
        # count of news 2*N, count of state change history 2*N, number of followers N
        # self.observation_space = 5 * self.N
        self.observation_space = np.zeros((5*self.N))
        self.action_space = spaces.Discrete(self.N)
        self.stage_num = stage_num
        self.stage = 0
        self.starttime = starttime
        self.endtime = endtime
        self.reward_range = (0, float("inf"))
        self.metadata = {"render.modes": []}
        
        self.state_d_f = np.zeros((self.N))
        self.state_d_m = np.zeros((self.N))
        followers = np.zeros((self.N))
        for u, v in self.G.out_edges():
            followers[v] += 1
        self.followers = followers
        
        # start time after currtime, let fake news propagate till starttime
        # if self.starttime > self.currtime: 
            # graph_hist, IC = EoN.Gillespie_simple_contagion_decay_fast(self.G, self.H, self.J, self.IC, self.return_statuses, tmin=self.currtime, tmax = self.starttime, return_full_data=True)
            
            # self.currtime = self.starttime
            # self.IC = IC
        
    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        
    def logistic(self, x, x_a):
        # Here, since the output is a prob, we let L = 1 and k=1
        return 1 / (1 + math.exp(x_a-x))
        
    def reset(self):
        # during simulation, G, H, J will not change, just reset state of node and currtime

        # DEPRECATED
        # self.IC = copy.deepcopy(self.orig_IC)
        
        # New setting, every episode will reset infected users
        for i in range(self.N):
            self.IC[i] = 'S'
        for i in random.choices(range(self.N), k=20):
            self.IC[i] = 'I'
            
        for node in self.G.nodes():
            self.G.nodes[node]['state'] = self.IC[node]
            if self.IC[node] == 'I':
                self.G.nodes[node]['lsctime'] = [0.]
                self.G.nodes[node]['lscstate'] = ['I']
            else:
                self.G.nodes[node]['lsctime'] = []
                self.G.nodes[node]['lscstate'] = []
        self.currtime = 0
        self.stage = 0
        
        self.state_d_f = np.zeros((self.N))
        self.state_d_m = np.zeros((self.N))
        
        # start time after currtime, let fake news propagate till starttime
        if self.starttime > self.currtime: 
            graph_hist, IC = EoN.Gillespie_simple_contagion_decay_fast(self.G, self.H, self.J, self.IC, self.return_statuses, tmin=self.currtime, tmax = self.starttime, return_full_data=True)
            
            self.currtime = self.starttime
            self.IC = IC
            
            return_state = np.zeros((4 * self.N))
            
            # calculate the integral of intensity of every user from self.currtime to self.currtime+self.timestep
            result_state = graph_hist.get_statuses(time=self.currtime)
            
            nodes = list(self.G.nodes)
            for node in nodes:
                node_lsctime = self.G.nodes[node]['lsctime']
                node_lscstate = self.G.nodes[node]['lscstate']
                
                if result_state[node] == "R":
                    return_state[node+3*self.N] = 1
                elif result_state[node] == "I":
                    return_state[node+self.N] = 1
                
                # calculate the integral of intensities as state
                for i in range(len(node_lsctime)):
                    if node_lsctime[i] > self.currtime:
                        # i is the first one or the last one
                        if i == 0 or i == len(node_lsctime) - 1:
                            # and is the last one
                            int_start_time = node_lsctime[i]
                            if i == len(node_lsctime) - 1:
                                int_end_time = self.currtime+self.timestep
                            # not the last one
                            else:
                                int_end_time = node_lsctime[i+1]
                            
                        # else
                        else:
                            # last one bigger than currtime
                            if node_lsctime[i-1] > self.currtime:
                                int_start_time = node_lsctime[i-1]
                            else:
                                int_start_time = self.currtime
                            if node_lsctime[i+1] < self.currtime+self.timestep:
                                int_end_time = node_lsctime[i+1]
                            else:
                                int_end_time = self.currtime+self.timestep
                        
                        
                        int_intensity = self.getnewsprob(i, int_start_time, int_end_time)
                        if node_lscstate[i] == 'R':
                            # return_state[node+2*self.N] += int_intensity
                            self.state_d_m[node] += int_intensity

                        elif node_lscstate[i] == 'I':
                            # return_state[node] += int_intensity
                            self.state_d_f[node] += int_intensity
            
                return_state[node+2*self.N] = self.state_d_m[node]
                return_state[node] = self.state_d_f[node]
             

            return np.concatenate((return_state, self.followers))
        
        else:
            return np.concatenate((np.zeros((4*self.N)), self.followers))

    def getnewsprob(self, user, starttime, endtime):
        # print(self.G.nodes[user]['intensity_weight'] * quad(lambda x:  math.exp(-x) , 0, endtime-starttime)[0])
        return self.G.nodes[user]['intensity_weight'] * quad(lambda x:  math.exp(-x) , 0, endtime-starttime)[0]

    def step(self, action, withreward=False):
        # action is done by modifing self.IC
        self.stage += 1
        
        # DEPRECATED, now will set a user to R without considering the probablity
        # # with the prob to change state info recovered
        # if random.random() < 0.8:
            # # print(action)
            # self.IC[action] = "R"
        
        # NEW SETTING
        # DEPRECATED AGAIN
        # self.IC[action] = 'R'
        # if self.G.nodes[action]["state"] != 'R':
            # self.G.nodes[action]["state"] = 'R'
            # self.G.nodes[action]["lsctime"].append(self.currtime)
            # self.G.nodes[action]["lscstate"].append('R')
        
        prob_action = self.logistic(1.5 , self.G.nodes[action]['logistic_weight'])
        if random.random() < prob_action:
            self.IC[action] = 'R'
            if self.G.nodes[action]["state"] != 'R':
                self.G.nodes[action]["state"] = 'R'
                self.G.nodes[action]["lsctime"].append(self.currtime)
                self.G.nodes[action]["lscstate"].append('R')
        
        
        #t, self.IC, S, EI, ER, I, R = EoN.Gillespie_simple_contagion(G, H, J, self.IC, return_statuses, tmin=self.currtime, tmax = self.currtime + self.timestep)
        
        
        graph_hist, IC = EoN.Gillespie_simple_contagion_decay_fast(self.G, self.H, self.J, self.IC, self.return_statuses, tmin=self.currtime, tmax = self.currtime + self.timestep, return_full_data=True)
        
        # print(graph_hist)
        
        # Let's just ignore the FXXKING timestep and make it simple
        # prepare the state to return
        return_state = np.zeros((4 * self.N))
        
        # calculate the integral of intensity of every user from self.currtime to self.currtime+self.timestep
        nodes = list(self.G.nodes)
        # print("nodes is:",nodes)
        for node in nodes:
            node_lsctime = self.G.nodes[node]['lsctime']
            node_lscstate = self.G.nodes[node]['lscstate']
            # calculate the integral of intensities as state
            for i in range(len(node_lsctime)):
                # print(node_lsctime[i], self.currtime)
                if node_lsctime[i] > self.currtime:
                    # i is the first one or the last one
                    if i == 0 or i == len(node_lsctime) - 1:
                        # and is the last one
                        int_start_time = node_lsctime[i]
                        if i == len(node_lsctime) - 1:
                            int_end_time = self.currtime+self.timestep
                        # not the last one
                        else:
                            int_end_time = node_lsctime[i+1]
                        
                    # else
                    else:
                        # last one bigger than currtime
                        if node_lsctime[i-1] > self.currtime:
                            int_start_time = node_lsctime[i-1]
                        else:
                            int_start_time = self.currtime
                        if node_lsctime[i+1] < self.currtime+self.timestep:
                            int_end_time = node_lsctime[i+1]
                        else:
                            int_end_time = self.currtime+self.timestep
                    
                    
                    int_intensity = self.getnewsprob(node, int_start_time, int_end_time)
                    # print("add intensity")
                    # print(int_start_time,int_end_time)
                    # print(int_intensity)
                    if node_lscstate[i] == 'R':
                        # return_state[node+2*self.N] += int_intensity
                        self.state_d_m[node] += int_intensity
                        # return_state[node+3*self.N] += 1
                    elif node_lscstate[i] == 'I':
                        # return_state[node] += int_intensity
                        self.state_d_f[node] += int_intensity
                        # return_state[node+self.N] += 1
                        
            return_state[node+2*self.N] = self.state_d_m[node]
            return_state[node] = self.state_d_f[node]

        return_state = np.concatenate((return_state, self.followers))
        
        
        step_reward = 0
        final_state = graph_hist.get_statuses(time=self.currtime + self.timestep)
        for i in range(self.N):
            if final_state[i] == "R":
                return_state[i+3*self.N] = 1
            elif final_state[i] == "I":
                return_state[i+self.N] = 1
            
            if final_state[i] == "I":
                step_reward += 1
        # print("step_reward:", step_reward)
        step_reward = -np.log(float(step_reward)/float(self.N))
        
        
        # prepare the reward to return
        return_reward = 0
        isEnd = False
        if self.stage_num <= self.stage:
            graph_hist, IC = EoN.Gillespie_simple_contagion_decay_fast(self.G, self.H, self.J, IC, self.return_statuses, tmin=self.currtime + self.timestep, tmax = self.endtime, return_full_data=True)
            
            isEnd = True
            final_state = graph_hist.get_statuses(time=self.endtime)
            for i in range(self.N):
                # if final_state[i] == "R" or final_state[i] == "S":
                    # return_reward += 1
                if final_state[i] == "I":
                    return_reward += 1
            
            if return_reward == 0:
                return_reward = 10.
            else:
                return_reward = -np.log(float(return_reward)/float(self.N))
            # for i in range(self.N):
                # if final_state[i] == "I":
                    # return_reward += 1
            # if return_reward == 0:
                # return_reward = 1e3
            # else:
                # return_reward = -np.log(float(return_reward)/float(self.N))
            self.currtime = self.endtime
            self.IC = IC
        else:
            # prepare for next step
            self.currtime += self.timestep
            self.IC = IC
        
        # plt.semilogy(t, S, label = 'Susceptible')
        # plt.semilogy(t, EI, label = 'Exposed Infected')
        # plt.semilogy(t, ER, label = 'Exposed Recovered')
        # plt.semilogy(t, I, label = 'Infected')
        # plt.semilogy(t, R, label = 'Recovered')
        # plt.legend()

        # plt.show()
        
        if withreward:
            return return_state, return_reward, isEnd, {}, step_reward
        else:
            return return_state, return_reward, isEnd, {}
        
        
    def steps(self, actions):
        # action is done by modifing self.IC
        # self.stage += 1
        
        # DEPRECATED, now will set a user to R without considering the probablity
        # # with the prob to change state info recovered
        # if random.random() < 0.8:
            # # print(action)
            # self.IC[action] = "R"
        
        # DEPRECATED AGAIN
        # NEW SETTING
        # for action in actions:
            # self.IC[action] = 'R'
            # if self.G.nodes[action]["state"] != 'R':
                # self.G.nodes[action]["state"] = 'R'
                # self.G.nodes[action]["lsctime"].append(self.currtime)
                # self.G.nodes[action]["lscstate"].append('R')
                
        # NEW NEW SETTING
        for action in actions:      
            prob_action = self.logistic(1.5, self.G.nodes[action]['logistic_weight'])
            if random.random() < prob_action:
                self.IC[action] = 'R'
                if self.G.nodes[action]["state"] != 'R':
                    self.G.nodes[action]["state"] = 'R'
                    self.G.nodes[action]["lsctime"].append(self.currtime)
                    self.G.nodes[action]["lscstate"].append('R')
        
        
        #t, self.IC, S, EI, ER, I, R = EoN.Gillespie_simple_contagion(G, H, J, self.IC, return_statuses, tmin=self.currtime, tmax = self.currtime + self.timestep)
        
        
        graph_hist, IC = EoN.Gillespie_simple_contagion_decay_fast(self.G, self.H, self.J, self.IC, self.return_statuses, tmin=self.currtime, tmax = self.endtime, return_full_data=True)
        
        # print(graph_hist)
        
        # Let's just ignore the FXXKING timestep and make it simple
        # prepare the state to return
        return_state = np.zeros((4 * self.N))
        
        # calculate the integral of intensity of every user from self.currtime to self.currtime+self.timestep
        nodes = list(self.G.nodes)
        for node in nodes:
            node_lsctime = self.G.nodes[node]['lsctime']
            node_lscstate = self.G.nodes[node]['lscstate']
            # calculate the integral of intensities as state
            for i in range(len(node_lsctime)):
                if node_lsctime[i] > self.currtime:
                    # i is the first one or the last one
                    if i == 0 or i == len(node_lsctime) - 1:
                        # and is the last one
                        int_start_time = node_lsctime[i]
                        if i == len(node_lsctime) - 1:
                            int_end_time = self.currtime+self.timestep
                        # not the last one
                        else:
                            int_end_time = node_lsctime[i+1]
                        
                    # else
                    else:
                        # last one bigger than currtime
                        if node_lsctime[i-1] > self.currtime:
                            int_start_time = node_lsctime[i-1]
                        else:
                            int_start_time = self.currtime
                        if node_lsctime[i+1] < self.currtime+self.timestep:
                            int_end_time = node_lsctime[i+1]
                        else:
                            int_end_time = self.currtime+self.timestep
                    
                    
                    int_intensity = self.getnewsprob(i, int_start_time, int_end_time)
                    if node_lscstate[i] == 'R':
                        # return_state[node+2*self.N] += int_intensity
                        self.state_d_m[node] += int_intensity
                        # return_state[node+3*self.N] += 1
                    elif node_lscstate[i] == 'I':
                        # return_state[node] += int_intensity
                        self.state_d_f[node] += int_intensity
                        # return_state[node+self.N] += 1
                        
            return_state[node+2*self.N] = self.state_d_m[node]
            return_state[node] = self.state_d_f[node]

        return_state = np.concatenate((return_state, self.followers))
        
        
        
        # prepare the reward to return
        return_reward = 0
            
        isEnd = True
        final_state = graph_hist.get_statuses(time=self.endtime)
        for i in range(self.N):
            # if final_state[i] == "R" or final_state[i] == "S":
                # return_reward += 1
            if final_state[i] == "I":
                return_reward += 1


        if return_reward == 0:
            return_reward = 10.
        else:
            return_reward = -np.log(float(return_reward)/float(self.N))
        # for i in range(self.N):
            # if final_state[i] == "I":
                # return_reward += 1
        # if return_reward == 0:
            # return_reward = 1e3
        # else:
            # return_reward = -np.log(float(return_reward)/float(self.N))
        self.currtime = self.endtime
        self.IC = IC
        
        # plt.semilogy(t, S, label = 'Susceptible')
        # plt.semilogy(t, EI, label = 'Exposed Infected')
        # plt.semilogy(t, ER, label = 'Exposed Recovered')
        # plt.semilogy(t, I, label = 'Infected')
        # plt.semilogy(t, R, label = 'Recovered')
        # plt.legend()

        # plt.show()
        


        return return_state, return_reward, isEnd, {}
        

