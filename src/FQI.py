import gymnasium as gym
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
from pickle import dump, load
from gymnasium.wrappers import TimeLimit
from fast_env import FastHIVPatient

env_fixed = TimeLimit(
    env=FastHIVPatient(domain_randomization=False), max_episode_steps=200
) 
env_rand = TimeLimit(
    env=FastHIVPatient(domain_randomization=True), max_episode_steps=200
) 

class HIV_FQI:
    def __init__(self,augment_state=False,gamma=0.98,epsilon=0.1):
        self.num_actions = 4
        self.Q = None
        self.augment_state = augment_state
        self.gamma = gamma
        self.epsilon = epsilon
        self.it = 0

    # this function collects samples from env
    # it can operate both on and off policy, as indicated by the bool on_policy
    def collect_samples(self, env, num_samples, on_policy=False, disable_tqdm=False):
        s, _ = env.reset()
        #dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        I = []
        for it in tqdm(range(num_samples), disable=disable_tqdm):
            a = np.random.randint(4)
            if on_policy:
                a = self.epsilon_greedy(self.augment_one_state(s,self.it/200))
            s2, r, done, trunc, _ = env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(trunc)
            I.append(self.it/200)
            if done or trunc:
                s, _ = env.reset()
                self.it = 0
            else:
                s = s2
                self.it += 1
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        I = np.array(I)
        
        return S, A, R, S2, D, I
    
    def epsilon_greedy(self,s):
        if self.Q is not None:
            if np.random.rand() > self.epsilon:
                return self.greedy_action(s)
        return np.random.randint(4)
    
    def greedy_action(self, s):
        Q2 = np.zeros((self.num_actions))
        for a in range(self.num_actions):
            A = a*np.ones((1))
            SA = np.append(s,A)
            Q2[a] = self.Q.predict(SA.reshape((1,-1)))
        return np.argmax(Q2)

    def save(self, path):
        with open(path, "wb") as file:
            dump(self.Q, file, protocol=5)

    def load(self, path=None):
        with open(path, "rb") as file:
            self.Q = load(file)

    # augment the state with potentially useful variables
    def augment_one_state(self,s,it):
        aug = np.zeros((7))
        aug[:6] = s
        aug[-1] = it/200
        return aug
    
    # augments many states at the same time with potentially useful variables
    def augment_many_states(self, states, its):
        return np.append(states,its.reshape((-1,1)),axis=1)
    
    # monte carlo evaluation
    # different from the one provided by the professor bc it uses the FastHIVPatient class
    # credits to Clément for the code for the FastHIVPatient class
    # credits to Antani for the idea of implementing this alternative evaluation method
    def mc_eval(self, env, num_episodes=1):
        cum_rewards = []
        it = 0
        for _ in range(num_episodes):
            s,_ = env.reset()
            done = False
            trunc = False
            cum_reward = 0
            while not done and not trunc:
                if self.augment_state:
                    s = self.augment_one_state(s,it/200)
                a = self.greedy_action(s)
                s2,r,done,trunc,_ = env.step(a)
                cum_reward += r
                s = s2
                it += 1
            cum_rewards.append(cum_reward)
        return np.mean(cum_rewards)
    
    # expands the data on arrays S_old, A_old, R_old, S2_old, D_old, I_old by addind that in S_, A_, R_, S2_, D_, I_
    # if the resulting arrays have length greater than max_data then only the first 5000 and last max_data - 5000 samples are kept
    def expand_data(self, S_old, A_old, R_old, S2_old, D_old, I_old, S_, A_, R_, S2_, D_, I_, max_data=50000):
        # Concatenate and trim for S
        S = np.concatenate((S_old, S_))
        if S.shape[0] > max_data:
            S = np.concatenate((S[0:5000], S[-(max_data - 5000):]))

        # Concatenate and trim for A
        A = np.concatenate((A_old, A_))
        if A.shape[0] > max_data:
            A = np.concatenate((A[0:5000], A[-(max_data - 5000):]))

        # Concatenate and trim for R
        R = np.concatenate((R_old, R_))
        if R.shape[0] > max_data:
            R = np.concatenate((R[0:5000], R[-(max_data - 5000):]))

        # Concatenate and trim for S2
        S2 = np.concatenate((S2_old, S2_))
        if S2.shape[0] > max_data:
            S2 = np.concatenate((S2[0:5000], S2[-(max_data - 5000):]))

        # Concatenate and trim for D
        D = np.concatenate((D_old, D_))
        if D.shape[0] > max_data:
            D = np.concatenate((D[0:5000], D[-(max_data - 5000):]))

        # Concatenate and trim for I
        I = np.concatenate((I_old, I_))
        if I.shape[0] > max_data:
            I = np.concatenate((I[0:5000], I[-(max_data - 5000):]))
                        
        return S, A, R, S2, D, I 

    # trains the agent
    # checks every new model for performance and saves those who surpass certain thresholds on the fixed environment
    def train(self, num_epochs=10, iterations_per_epoch=200, path=None):
        fixed_env = TimeLimit(
            env=FastHIVPatient(domain_randomization=False), max_episode_steps=200
        ) 
        rand_env = TimeLimit(
            env=FastHIVPatient(domain_randomization=True), max_episode_steps=200
        ) 

        fixed_highscore = 0
        print("collecting samples")
        S, A, R, S2, D, I = self.collect_samples(fixed_env,num_samples=10000,on_policy=False)
        if self.augment_state:
            S = self.augment_many_states(S,I)
            S2 = self.augment_many_states(S2,I)
        if path is not None:
            self.load(path)
            # collect new data on policy
            S_, A_, R_, S2_, D_, I_ = self.collect_samples(rand_env,num_samples=15000,on_policy=True)
            S, A, R, S2, D, I = self.expand_data(S, A, R, S2, D, I, S_, A_, R_, S2_, D_, I_)
            if self.augment_state:
                S = self.augment_many_states(S,I)
                S2 = self.augment_many_states(S2,I)
        for epoch in range(num_epochs):
            print(S.shape, A.shape, R.shape, S2.shape, D.shape, I.shape)
            SA = np.append(S, A, axis=1)
            print(f"Starting epoch {epoch + 1}...")
            for i in tqdm(range(iterations_per_epoch)):
                if self.Q is None:
                    values = R.copy()
                else:
                    num_samples = S.shape[0]
                    Q2 = np.zeros((num_samples,self.num_actions))
                    for a2 in range(self.num_actions):
                        A2 = a2*np.ones((S.shape[0],1))
                        S2A2 = np.append(S2,A2,axis=1)
                        Q2[:,a2] = self.Q.predict(S2A2)
                    max_Q2 = np.max(Q2,axis=1)
                    values = R.flatten() + self.gamma*(1-D)*max_Q2

                Q = xgb.XGBRegressor(n_estimators=50)
                Q.fit(SA, values)
                self.Q = Q

                # evaluate agent in fixed env
                fixed_score = self.mc_eval(fixed_env)
                if fixed_score > fixed_highscore and fixed_score > 1e10:                
                    print(f"{fixed_score:e}")
                    fixed_highscore = fixed_score
                    print("\033[32mTHIS WAS A RECORD!\033[0m")
                    if self.augment_state:                        
                        self.save(f"src/models/best_FQI.pkl")
                    else:
                        self.save(f"src/models/best_FQI.pkl")
                    if fixed_score > 3e10:                    
                        # evaluate agent in rand env
                        rand_score = self.mc_eval(rand_env, num_episodes=50)
                        print(f"{rand_score:e}")
                elif fixed_score > 3e10:
                    print("Evaluating agent in rand env")
                    rand_score = self.mc_eval(rand_env, num_episodes=50)
                    print(f"{rand_score:e}")                    
                    # self.save(f"src/models/best_FQI.pkl")

            # collect new data on policy
            S_, A_, R_, S2_, D_, I_ = self.collect_samples(rand_env,num_samples=5000,on_policy=True)
            if self.augment_state:
                S_ = self.augment_many_states(S_,I_)
                S2_ = self.augment_many_states(S2_,I_)
            S, A, R, S2, D, I = self.expand_data(S, A, R, S2, D, I, S_, A_, R_, S2_, D_, I_)
            print(S.shape,A.shape,R.shape,S2.shape)
    


    
