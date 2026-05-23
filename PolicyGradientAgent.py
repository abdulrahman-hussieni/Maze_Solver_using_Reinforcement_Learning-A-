import random
import time
import pickle
import os
import math


class PolicyGradientAgent:
    """
    REINFORCE Algorithm - Monte Carlo Policy Gradient Method.

    Uses a softmax (Boltzmann) parameterized policy:
        pi(a|s) = exp(theta[s,a]) / sum_a'(exp(theta[s,a']))

    Update rule (after each episode):
        theta[s,a] += alpha * gamma^t * G_t * grad_log_pi(a|s)

    where G_t is the discounted return from time step t.
    """

    def __init__(self, env, alpha=0.01, gamma=0.99):
        assert alpha > 0, "alpha non valido"
        assert 0 <= gamma < 1, "gamma non valido"

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.actions = [i for i in range(0, self.env.get_action_space().n)]
        self.num_actions = len(self.actions)

        # Policy parameters: theta[state][action] -> preference value
        # Stored as dict: key = (tuple(state), action), value = float
        self.theta = {}

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------
    def normalize_state(self, state):
        return tuple(state[:2])

    def get_theta(self, state, action):
        return self.theta.get((self.normalize_state(state), action), 0.0)

    def set_theta(self, state, action, value):
        self.theta[(self.normalize_state(state), action)] = value

    # ------------------------------------------------------------------
    # Softmax policy
    # ------------------------------------------------------------------

    def compute_softmax_probs(self, state):
        """
        pi(a|s) = exp(theta(s,a)) / sum_a'(exp(theta(s,a')))
        Uses numerical stability trick: subtract max before exp.
        """
        preferences = [self.get_theta(state, a) for a in self.actions]
        max_pref = max(preferences)
        exp_prefs = [math.exp(p - max_pref) for p in preferences]
        total = sum(exp_prefs)
        probs = [e / total for e in exp_prefs]
        return probs

    def choose_action(self, state):
        probs = self.compute_softmax_probs(state)
        action = random.choices(self.actions, weights=probs, k=1)[0]
        return action

    # ------------------------------------------------------------------
    # REINFORCE update
    # ------------------------------------------------------------------

    def learn(self, episode):
        """
        REINFORCE Monte Carlo update.

        episode: list of (state, action, reward) tuples from one full episode.

        For each time step t:
            G_t = sum_{k=t}^{T} gamma^(k-t) * r_k
            For each action a:
                grad_log_pi(a|s_t) = I(a == a_t) - pi(a|s_t)
            theta[s_t, a] += alpha * gamma^t * G_t * grad_log_pi(a|s_t)
        """
        T = len(episode)

        for t in range(T):
            state, action_taken, _ = episode[t]

            # Compute discounted return G_t
            G_t = 0.0
            for k in range(t, T):
                _, _, reward = episode[k]
                G_t += (self.gamma ** (k - t)) * reward

            # Compute current policy probabilities at state
            probs = self.compute_softmax_probs(state)

            # Update theta for every action
            discount = self.gamma ** t
            for idx, a in enumerate(self.actions):
                # Gradient of log pi: 1 - pi(a|s) if a == a_taken, else -pi(a|s)
                if a == action_taken:
                    grad = 1.0 - probs[idx]
                else:
                    grad = -probs[idx]

                old_theta = self.get_theta(state, a)
                new_theta = old_theta + self.alpha * discount * G_t * grad
                self.set_theta(state, a, new_theta)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def start_training(self, num_of_episodes=100, time_between_step=0,
                       time_between_episode=0, save_theta=False,
                       theta_file_name="policy_gradient_theta"):
        assert num_of_episodes > 0, "number_of_episodes non valido"

        self.env.initialize_env()
        results = []

        for episode_idx in range(num_of_episodes):
            print(f"Started episode {episode_idx}")
            time.sleep(time_between_episode)

            state, reward, done, info = self.env.reset()

            episode_buffer = []   # stores (state, action, reward)
            rewards = [reward]
            infos = [info]

            while not done:
                action = self.choose_action(state)
                new_state, reward, done, info = self.env.step(action)

                episode_buffer.append((state, action, reward))
                rewards.append(reward)
                infos.append(info)

                state = new_state
                time.sleep(time_between_step)

            # Full episode collected -> run REINFORCE update
            self.learn(episode_buffer)
            results.append([rewards, infos])

        if save_theta:
            self.save_theta(theta_file_name)

        return results

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_theta(self, file_name):
        if not file_name.endswith(".pkl"):
            file_name = file_name + ".pkl"
        with open(file_name, "wb") as f:
            pickle.dump(self.theta, f, pickle.HIGHEST_PROTOCOL)
        print("\nPolicy Gradient theta salvato.\n")

    def load_stored_theta(self, file_name):
        if os.path.isfile(file_name):
            with open(file_name, "rb") as f:
                self.theta = pickle.load(f)
            print("\nPolicy Gradient theta caricato.\n")
        else:
            print("\nFile .pkl non trovato\n")

    # ------------------------------------------------------------------
    # Getters / Setters
    # ------------------------------------------------------------------

    def get_theta_table(self):
        return self.theta
