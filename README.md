# Introduction

This assignment involves finding an optimal policy for a given MDP.  Specifically, the implementation will focus on iterative techniques such as value iteration, policy iteration, and their respective variants. Additionally, the analysis will extend to the examination of the impact of various components defining an MDP on the optimal policy. 

# Domain Description

Consider an autonomous car in the grid world environment. This car comes with a hybrid engine, allowing it to run on petrol or electricity. It is capable of moving in four directions: top, down, right, and left. Due to construction work, holes have been dug up in the streets of the grid world. Therefore, the autonomous car needs to find an optimal policy (the action it needs to take in every grid cell) to avoid falling into the holes and drive safely to its destination.

### Environment

The environment is represented as a 2D grid world with discrete grid cells. Specific cells are designated as 'H,' indicating the presence of a hole. The garage is denoted by 'S,' where the vehicle is initially parked, and the destination is marked as 'G.' Cells that are neither holes nor the goal are labelled as 'F,' indicating free cells.

### Transition Model

- The car can move in four directions: top ('t'), down ('d'), right ('r'), and left ('l'). The intended state change for each action is:-

<pre>
              right (x,y) = (x+1, y)
              left (x,y) = (x-1, y)
              top (x,y) = (x, y+1)
              down (x,y) = (x, y-1)
</pre>

- Because of rain, the roads in the grid world become slippery, introducing uncertainty to the effects of actions. In other words, the transitions are stochastic. Any action will result in the intended cell with a probability $p$  and randomly in one of the other three neighbours with probability $(1-p)$. Transitions outside the grid boundary are not allowed.  That is, transitions resulting in a position outside the grid boundary will not change the state.
- It is important to note that moving into a hole or reaching the goal is considered a *terminal state* in this context.

### Reward Model

- The car gets a reward after taking an action and reaching a state.
- For each action that does not lead to a hole or the goal, the agent is granted a `living_reward`. In terminal states, where the agent encounters a hole or reaches the goal, it receives the respective rewards, namely `hole_reward` and `goal_reward`.  Formally, the reward model $R(s, a, s^\prime)$  is defined as

$$
R(s, a, s^\prime) = \begin{cases}
\mathtt{living\_reward}, \quad \text{if } s^\prime = \text{F} \\
\mathtt{hole\_reward}, \quad \text{if } s^\prime = \text{H} \\
\mathtt{goal\_reward}, \quad \text{if } s^\prime = \text{G} \\
\end{cases}
$$

- The discount factor, $\gamma$, represents the autonomous car's degree of patience or persistence in accumulating delayed rewards.

## Part A: Solving for optimal policy

Your task is to use iterative methods to solve the given MDP and find the optimal policy $\pi^*$. Your implementation should be general enough to be able to work for any given grid.  You are provided with two maps: `small_map` and `large_map` to test your implementation. Assume the following default values for the parameters of the transition and reward model in your implementation. 

- The transition probability $p$  is 0.8, meaning that 80% of the time, actions lead to the intended state, while there's a 20% chance of reaching a random neighbouring cell.
- The `living_reward` and `hole_reward` are 0. The `goal_reward` is 1.
- The discount factor, $\gamma$, is 0.9.
    
  ### Value Iteration
    
    Implement a vanilla value iteration. As discussed in the lecture, threshold the max-norm distance between consequent value updates to determine convergence. Experiment with different epsilon. Experiment with varying epsilon values and report the epsilon for which the final value estimate yields a sensible optimal policy.

  ### Asynchronous Updates

    Implement value iteration with asynchronous updates, where the value function is updated in-place within each iteration.  Note that the order in which cells are iterated over in the grid influences the convergence rate in the case of asynchronous updates. Implement the following sweeping strategies.

- **Row-major sweep**: Cells are updated row by row, traversing each row before moving to the next.
- **Prioritized sweep:**  Use the magnitude of Bellman error to guide state selection. That is, the states whose successors change the most are prioritized.  The bellman error is defined as
    
$$
\Big| \max_a \mathbb{E}[R_{t+1} + \gamma v(S_{t+1}) | S_t =s] - v(s)\Big |
$$
    

&emsp;&emsp;Compare and report the convergence speed, measured in the number of iterations, for both synchronous and asynchronous versions of value iteration. Asses whether the difference becomes notable for larger grids.

### Policy Iteration

Implement policy iteration, assuming a random initial policy estimate. Report the criterion used to determine the convergence of the policy iteration. 

For all the algorithms, report the following information for both maps. 

1. Report the wall time taken for convergence of both the algorithms and the number of iterations (count of policy iteration cycles for policy iteration and value iteration cycles for value iteration) till convergence in the default settings. Which algorithm do you think is scalable for larger environments?
2. Plot the estimated value of the starting state for every iteration. You should report two plots, one for each map. Each plot should contain four line graphs (one for each algorithm - vanilla value iteration, row-major sweep, prioritized sweep, policy iteration), showing the evolution of the value of the starting state. 
3. Generate a heat map of the values obtained after the convergence. Draw the optimal policy as arrows (denoting the direction of movement) over the grid cell. Which algorithm results in a better optimal policy?
