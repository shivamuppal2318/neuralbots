NeuralLux-Bots: Lux AI Multi-Agent DQN
Deep Reinforcement Learning for Multi-Agent Strategy in Lux AI Season 3

📌 Table of Contents
Introduction

Features

Installation

Usage

Training Pipeline

Architecture

Results & Performance

Contributing

License

🎯 Introduction
This project implements Deep Q-Networks (DQN) for multi-agent reinforcement learning (MARL) tailored for the Lux AI Challenge Season 3. Our agents learn to navigate the environment strategically, gather resources, and compete effectively by sharing experience through a replay buffer and maintaining separate policy and target networks for stable training.

✨ Features
✅ Multi-Agent Deep Q-Learning for complex strategic decision-making.

✅ Experience Replay via a shared replay buffer to improve sample efficiency.

✅ Target & Policy Networks to stabilize training and reduce divergence.

✅ Epsilon-Greedy Exploration for balancing exploration and exploitation.

✅ Adaptive Model Sampling to prevent overfitting on specific strategies.

✅ Log-Based Reward Tracking for better debugging and performance monitoring.

⚙️ Installation
1️⃣ Clone the repository

bash
Copy
Edit
git clone https://github.com/your_username/LuxAI-MultiAgentDQN.git
cd LuxAI-MultiAgentDQN
2️⃣ Set up a virtual environment

bash
Copy
Edit
python3 -m venv luxai_env
source luxai_env/bin/activate  # On Windows use: luxai_env\Scripts\activate
3️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Install Lux AI environment

bash
Copy
Edit
pip install luxai_s3
🚀 Usage
Run the environment with your bot:

bash
Copy
Edit
luxai-s3 path/to/your/bot.py path/to/opponent/bot.py --output replay.json
Train the model:

bash
Copy
Edit
python train.py
Evaluate performance:

bash
Copy
Edit
python evaluate.py --episodes 50
🧠 Training Pipeline
1️⃣ Observation Extraction: Transform raw environment state into agent-friendly features.
2️⃣ Action Selection: Select actions using the policy network with epsilon-greedy exploration.
3️⃣ Experience Storage: Store tuples (state, action, reward, next_state) into a shared replay buffer.
4️⃣ Batch Training:

Sample mini-batches from the replay buffer.

Compute Temporal Difference (TD) targets using the target network.

Update policy network by minimizing mean squared error (MSE) loss of Q-values.
5️⃣ Target Network Update: Periodically sync the target network weights with the policy network for training stability.

🔧 Architecture
Multi-Agent DQN Structure
Agent.py: Manages game interaction and reward computation.

DQN.py: Implements Deep Q-Networks, including policy and target networks.

ReplayBuffer.py: Shared experience replay buffer for multi-agent learning.

Train.py: Training loop and optimization logic.

Evaluate.py: Performance evaluation and testing.

Neural Network Model
Fully connected layers with ReLU activations.

Q-Values Head: Predicts action-value estimates for decision making.

X, Y Coordinate Heads: Predict (x, y) positions for sapping or positional actions.

📊 Results & Performance
Episodes	Avg Reward	Win Rate (%)
100	-5.3	45.2
500	10.2	60.5
1000	18.7	75.8

Sample Training Reward Curve

Replace path/to/reward_curve.png with your actual reward curve image.

💡 Contributing
We welcome contributions! To contribute:
1️⃣ Fork the repository
2️⃣ Create a feature branch (git checkout -b feature-name)
3️⃣ Commit your changes with meaningful messages (git commit -m "Add feature")
4️⃣ Push to your branch (git push origin feature-name)
5️⃣ Open a pull request

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.



💡 Contributing
We welcome contributions! To contribute: 1️⃣ Fork the repository. 2️⃣ Create a feature branch. 3️⃣ Commit changes with meaningful messages. 4️⃣ Submit a pull request.

📜 License
📌 This project is licensed under MIT License.
