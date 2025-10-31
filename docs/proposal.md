# Deep Dive Proposal: Q-Learning

## Project Overview

### Project Goals

For this project, I plan to use the Q-Learning algorithm to solve a robotic path-planning problem; specifically, the navigation of a warehouse. This requires three steps: understanding Q-Learning in the context of other machine learning methods; implementing the Q-Learning algorithm to find the optimal path through a toy model of a warehouse; and analyzing the algorithm's performance given varying circumstances.

### What Is Q-Learning?

Q-Learning is a Reinforcement Learning (RL) algorithm used to train an agent to make optimal choices in an environment without modelling it mathematically.

All types of RL algorithms are concerned with agents acting in environments to maximize some reward function. What makes Q-Learning unique is that it does not attempt to model the dynamics of the environment. This means that the Q-Learning algorithm has no idea how taking an action in one state leads to another. Instead, it only knows the cost associated with taking that action in that state. This way, the algorithm  learns a policy that maximizes the reward function by trial-and-error.

### Motivation

I chose this project to give myself an opportunity to explore RL, which is an important class of algorithms in robotic planning. While Q-Learning isn't the most cutting-edge RL algorithm available, it is simple to understand and implement, while still being an applicable and popular algorithm choice. Through this project, I hope to gain experience working with RL algorithms and analyzing their performance in the context of robotic decision-making.

## Deliverables

As a final deliverable for this project, I plan to write a final report on my learning that includes the following components:

### Summary of Q-Learning

My report will begin with a discussion of what Q-Learning is, what makes it unique in the field of machine learning, and how it compares to other RL algorithms. I will create a block of pseudocode describing the algorithm and an explanation of what each line does. I will also articulate the significance of Q-Learning in the field of mobile robotics and path planning.

In doing so, I will gain a strong understanding of what applications Q-Learning is best--suited for in comparison to other RL methods (i.e. policy gradient method, Model Predictive Control). I also hope to gain a general understanding of how other classes of machine learning (supervised and unsupervised) are distinct from RL.

### Codebase of Q-Learning Implementation

I will implement the Q-Learning algorithm to train a robotic agent to find the optimal path to a destination in a model warehouse environment.

I envision a warehouse world model in which a robotic agent has a starting position and a goal position in 2D space. I plan to model the world as a weighted, undirected graph, which a robot will traverse to achieve some goal. The world may contain impassible obstacles (such as shelves) and/or areas of discouraged travel (such as highly-trafficked zones). These areas will incur a cost if the robot traverses them, which I will encode in the weights of the world graph's edges.

The robotic agent will have a set of discrete actions it can take in the world: up, down, left, and right. When the robot takes an action, it will incur the cost associated with its new position. The Q-Learning algorithm will associate that cost with the given state-action pair, and eventually teach the agent the optimal path through the warehouse.

In addition to the world and the robotic agent, I plan to have a class dedicated to visualizing each step of the learning process. The visualizer will create animated plots of the agent as it explores the world model, as well as benchmarking plots that summarize how quickly the algorithm can find the optimal path given variations on the same environment (i.e. more or less obstacles).

### Performance Analysis

I will analyze the success of the Q-Learning algorithm given variations on the warehouse environment.  Given the plots created by my visualizer class, I will be able to explain why the algorithm performs better or worse in different scenarios. I will propose possible improvements to the algorithm, and/or speculate on how different methods of RL might compare to the performance of my algorithm implementation. This analysis will be in the form of a final report, alongside my initial summary of the algorithm.

## Rubric

I propose the following grading scheme, in which I receive up to 8 points for completing the project:

A: 7-8 pts | B: 5-6pts | C: 3-4pts | D = 1-2pts | F: 0pts

- Q-Learning summary: 1 point
- Warehouse model implementation: 1 point
- Q-Learning implementation: 3 points
- Plots and analysis: 2 points
- Unit testing: 1 point

## Timeline

I will commit to the following timeline for my project, given that I have seven days to complete it.

### Day 1: Q-Learning Math and Project Proposal

I will start by reading about the Q-Learning algorithm and its underlying math. I will contextualize what I learn by spending some time on alternate RL algorithms, as well as a quick tour of non-RL classes of maching learning (supervised and unsupervised learning). I will demonstrate my learning by writing the Q-Learning Summary component of my final report, including pseudocode to describe the algorithm.

In addition to this, I will write the project proposal (this document) for my project. I will plan two days of contingency to be safe, and scope my project accordingly given the amount of time I have left. I will also choose a few simple stretch goals in the event that things go well.

### Day 2: Implement the Environment Model

I will implement the underlying graph that models my warehouse environment. I will create constructor methods to make it easy to create new environments; for example, one method will take an entire set of weights, while another will take an obstacle density and generate a graph internally. I will also create a few example environments with different features, such as discouraged regions, in preparation for benchmarking.

### Day 3: Implement Agent and Q-Learning Algorithm

I will implement the robotic agent, which utilizes Q-Learning to find the optimal path through the warehouse. The agent will have a given set of actions it can take in the world, as well as a starting position. The agent will log the Q-value of every state-action pair it explores in a Q-table. After repeatedly exploring the warehouse environment, the agent will eventually find the optimal path from its starting location to its goal.

### Day 4: Write Unit Tests and Visualizer

I plan to develop unit tests alongside my classes, but I will set a day aside to catch up on any missing unit tests that will be helpful for verifying my implementation.

I will also write my visualizer class, which will provide digestible insights into how well the algorithm is working. I will use this to sanity-check my implementation, and make sure that any underperformance is still within reasonable expectations for the Q-Learning algorithm.

### Day 5: Contingency Day For Debugging

I will set this day aside as a contingency day, in the event that the unit tests and visualization code I developed reveal any errors in my implementation.

### Day 6: Write Final Report

I will write the final report for this project, which primarily includes discussion of the plots produced by my visualizer. I will also go through my codebase and make sure that all functions are properly documented. If there are any errors in my implementation that I have not had the chance to fix, I will explain them here.

This is a tentative list of plots I would like to include in my final report:
- Obstacle density vs length of optimized path
- Obstacle density vs time to train model
- Trial # vs length of path
- Comparison of Dijkstra's shortest path to RL shortest path

### Day 7: Contingency Day and Stretch Goals

I will set this day aside as a contingency day to catch up on any existing bugs or missing parts of the final report. If I have no extra work to do, I will explore one of the following stretch goals:
- Further analysis of Q-Learning success with different types of environmental models
- Integrate stochasticity into actions taken; update algorithm learning rate
- Comparison of Q-Learning method to Dijkstra's algorithm or A* algorithm, in terms of quality and speed
- Speculation on how other RL or ML algorithms would perform given the same problem statement
