from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def default_array():
    return np.zeros(2)  # Hit or Stick

class BlackjackAgent:
    def __init__(self, env, learning_rate, discount_factor, epsilon):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # 使用两个Q表
        self.q_table1 = defaultdict(default_array)
        self.q_table2 = defaultdict(default_array)
        self.visit_counts = defaultdict(default_array)

    def get_state_key(self, s):
        return tuple(s) if isinstance(s, (list, np.ndarray)) else s

    def get_action(self, s):
        s_key = self.get_state_key(s)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # 使用两个Q表的平均值
            q_tab = (self.q_table1[s_key] + self.q_table2[s_key]) / 2
            return np.argmax(q_tab)

    def update_q_table(self, s, a, r, next_s, done):
        s_key = self.get_state_key(s)
        next_s_key = self.get_state_key(next_s)

        self.visit_counts[s_key][a] += 1

        # 随机选择一个Q表进行更新
        if np.random.random() < 0.5:
            # 使用Q2选择动作，用Q1更新
            next_action = np.argmax(self.q_table2[next_s_key])
            next_q = self.q_table1[next_s_key][next_action] if not done else 0
            current_q = self.q_table1[s_key][a]
            td_error = r + self.discount_factor * next_q - current_q
            self.q_table1[s_key][a] += self.learning_rate * td_error
        else:
            # 使用Q1选择动作，用Q2更新
            next_action = np.argmax(self.q_table1[next_s_key])
            next_q = self.q_table2[next_s_key][next_action] if not done else 0
            current_q = self.q_table2[s_key][a]
            td_error = r + self.discount_factor * next_q - current_q
            self.q_table2[s_key][a] += self.learning_rate * td_error

    def plot_policy(self):
        """Plot policy heatmaps"""
        player_sum = np.arange(21, 3, -1)
        dealer_card = np.arange(1, 11)

        policy_matrix_no_ace = np.zeros((len(player_sum), len(dealer_card)))
        policy_matrix_ace = np.zeros((len(player_sum), len(dealer_card)))

        for i, player in enumerate(player_sum):
            for j, dealer in enumerate(dealer_card):
                state_no_ace = (player, dealer, 0)
                state_ace = (player, dealer, 1)
                # 使用两个Q表的平均值
                q_no_ace = (self.q_table1[state_no_ace] + self.q_table2[state_no_ace]) / 2
                q_ace = (self.q_table1[state_ace] + self.q_table2[state_ace]) / 2
                policy_matrix_no_ace[i, j] = np.argmax(q_no_ace)
                policy_matrix_ace[i, j] = np.argmax(q_ace)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        # No ace
        im1 = ax1.imshow(policy_matrix_no_ace, cmap='RdYlBu')
        ax1.set_title('Policy (No Usable Ace)')
        ax1.set_xlabel('Dealer Card')
        ax1.set_ylabel('Player Sum')
        ax1.set_xticks(np.arange(len(dealer_card)))
        ax1.set_yticks(np.arange(len(player_sum)))
        ax1.set_xticklabels(dealer_card)
        ax1.set_yticklabels(player_sum)

        # With Ace
        im2 = ax2.imshow(policy_matrix_ace, cmap='RdYlBu')
        ax2.set_title('Policy (With Usable Ace)')
        ax2.set_xlabel('Dealer Card')
        ax2.set_ylabel('Player Sum')
        ax2.set_xticks(np.arange(len(dealer_card)))
        ax2.set_yticks(np.arange(len(player_sum)))
        ax2.set_xticklabels(dealer_card)
        ax2.set_yticklabels(player_sum)

        # Add colorbars
        plt.colorbar(im1, ax=ax1, label='Action (0: Stick, 1:Hit)')
        plt.colorbar(im2, ax=ax2, label='Action (0: Stick, 1:Hit)')

        plt.tight_layout()

        plt.savefig(f'policy_heatmap.png', dpi=300)
        plt.show()