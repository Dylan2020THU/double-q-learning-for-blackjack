Double Q-Learning Blackjack (Gym)

This project trains a Blackjack agent with Double Q-Learning using OpenAI Gym's `Blackjack-v1` environment.
The agent uses two Q-tables to reduce overestimation bias and learns a policy for:
- `0` = Stick
- `1` = Hit

------------------------------------------------------------
1) Project Files
------------------------------------------------------------
- `q_table_agent.py`
  Contains the `BlackjackAgent` class and Double Q-Learning update logic.

- `q_table_training.py`
  Runs training, plots average reward, generates policy heatmap, and includes optional test play.

- `q_table_test.py`
  Evaluates a saved Q-table file (`q_table_*.pkl`) over many episodes.

- `check_q_table.py`
  Loads and prints the latest saved Q-table file.

- `requirements.txt`
  Python dependencies.

------------------------------------------------------------
2) Requirements
------------------------------------------------------------
- Python 3.9+ (recommended)
- Packages from `requirements.txt`

Install dependencies:

pip install -r requirements.txt

------------------------------------------------------------
3) How To Run
------------------------------------------------------------
Train the agent:

python q_table_training.py

During training, the script:
1. Trains for 100,000 episodes (default)
2. Prints average reward every 10,000 episodes
3. Saves a reward curve image:
   - `training_reward_<timestamp>.png`
4. Saves a policy heatmap:
   - `policy_heatmap.png`
5. Prompts whether to run a quick test

------------------------------------------------------------
4) Notes
------------------------------------------------------------
- `q_table_test.py` and `check_q_table.py` expect files like `q_table_*.pkl`.
- Current `q_table_training.py` does not save a Q-table `.pkl` yet.
  If you want to use those scripts, add Q-table serialization in training first.

------------------------------------------------------------
5) Example Output
------------------------------------------------------------
- Training reward curve image
- Policy heatmap for:
  - no usable ace
  - usable ace

------------------------------------------------------------
6) References
------------------------------------------------------------
- NIPS 2010: Double Q-Learning
- AAAI 2016: Deep Reinforcement Learning with Double Q-Learning

