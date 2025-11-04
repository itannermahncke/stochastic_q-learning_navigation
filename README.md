# Q-Learning In Stochastic Environments

> Ivy Mahncke\
> DSA Fall 2025 Midterm Project

## Project Overview

This is my midterm project for Data Structures & Algorithms, Fall 2025. In this project, I was challenged to study and implement an algorithm of my choice. I chose to explore the Q-Learning algorithm, a type of model-free Reinforcement Learning well-suited for stochastic environments. To demonstrate my learning, I trained a robotic agent to navigate a warehouse under various levels of uncertainty. I also experimentally determined an optimal value for its learning rate parameter.

I implore you to check out the [final report I wrote about the project,](docs/report.md) where I cover everything I did in a lot more detail!

My [initial project proposal document](docs/proposal.md) can also be found in this repository.

To see the assignment description and expectations, feel free to do so on [our class website.](https://olindsa2025.github.io/assignments/assignment_06.html)

## File Structure

This project contains the following files:

```bash
├── src
│   ├── main/kotlin
│   │   ├── Agent.kt
│   │   ├── WarehouseEnv.kt
│   │   ├── Visualizer.kt
│   │   ├── Utility.kt
│   │   ├── Main.kt
│   ├── test/kotlin
│   │   ├── AgentTest.kt
│   │   ├── WarehouseEnvTest.kt
├── docs
│   ├── gifs
│   ├── images
│   ├── proposal.md
│   ├── report.md
├── .gitignore
├── pom.xml
├── README.md
```

`src` houses the source code for this project. Here's a quick descriptor of the contents:
- `Agent.kt`: my implementation of the Q-Learning algorithm (accompanied by unit tests in `AgentTest.kt`)- `Agent.kt`: my implementation of the Q-Learning algorithm (accompanied by unit tests in `AgentTest.kt`)
- `WarehouseEnv.kt`: my warehouse environment model, capable of being deterministic or stochastic (accompanied by unit tests in `WarehouseEnvTest.kt`)
- `Visualizer.kt`: code for animating Q-Learning episodes and plotting information about training sessions
- `Utility.kt`: useful data classes and enums that are used across the project
- `Main.kt`: the file where it all comes together!

`docs` houses all documentation of my project. The two important files in here are:
- `proposal.md`: my project proposal! It contains learning goals, planned deliverables, and a project timeline.
- `report.md`: my final report! It contains my background research, methodology, and results, as well as ideas for future work.
