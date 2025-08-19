Deep Space Explorer – Reinforcement Learning for Autonomous Navigation
📌 Project Overview

This project implements a Reinforcement Learning (RL) agent for autonomous spacecraft navigation in a simulated space environment. The agent learns to optimize fuel usage, avoid obstacles, and reach its goal by interacting with a custom-built environment.

We use the Advantage Actor-Critic (A2C) algorithm with PPO enhancements to train the agent efficiently, enabling adaptive decision-making in dynamic and uncertain space conditions.

⚡ Key Features

✅ Custom OpenAI Gym Environment with dynamic obstacles, gravity wells, fuel stations, and space physics

✅ Advantage Actor-Critic (A2C) + PPO implementation in PyTorch

✅ Reward Engineering for trajectory optimization, obstacle avoidance, and fuel efficiency

✅ Difficulty Levels (Easy, Medium, Hard) with increasing environment complexity

✅ Training Visualization using Matplotlib & Seaborn for performance tracking

✅ Interactive Testing with real-time trajectory rendering

🛠️ Tech Stack

Programming Language: Python 3.8+

Libraries: PyTorch, NumPy, OpenAI Gym, Matplotlib, Seaborn

RL Algorithm: Advantage Actor-Critic (A2C) with PPO optimization

Environment Design: Custom Gym environment with space physics simulation

📊 Results

🚀 Agent successfully learned collision-free trajectories under different difficulty levels

🔋 Improved fuel efficiency by ~40% through hyperparameter tuning

🛰️ Achieved adaptive navigation in dynamic environments with moving obstacles and gravitational effects


🚀 How to Run

Clone the repo:

git clone https://github.com/manisha-31/Deep-Space-Explorer-RL.git
cd Deep-Space-Explorer-RL


Install dependencies:

pip install -r requirements.txt

Test the trained agent:

python RL.py

📌 Future Improvements

Multi-agent RL for swarm robotics in space exploration

Improved simulation-to-reality transfer for real spacecraft navigation

Integration with computer vision modules for terrain mapping

📖 References

Sutton & Barto, Reinforcement Learning: An Introduction

OpenAI Gym Documentation

NASA Technical Reports on Autonomous Navigation with RL

🔥 This project demonstrates how Reinforcement Learning can power autonomous space missions, reducing reliance on Earth-based control and enabling adaptive decision-making in complex environments.
