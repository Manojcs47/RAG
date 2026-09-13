# Hugging Face Deep RL Course — Unit 1: Introduction to Deep Reinforcement Learning

Source: https://huggingface.co/learn/deep-rl-course/unit1/introduction


---

<!-- introduction -->

# Introduction to Deep Reinforcement Learning [[introduction-to-deep-reinforcement-learning]]

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/thumbnail.jpg" alt="Unit 1 thumbnail" width="100%">


Welcome to the most fascinating topic in Artificial Intelligence: **Deep Reinforcement Learning.**

Deep RL is a type of Machine Learning where an agent learns **how to behave** in an environment **by performing actions** and **seeing the results.**

In this first unit, **you'll learn the foundations of Deep Reinforcement Learning.**


Then, you'll **train your Deep Reinforcement Learning agent, a lunar lander to land correctly on the Moon** using <a href="https://stable-baselines3.readthedocs.io/en/master/"> Stable-Baselines3 </a>, a Deep Reinforcement Learning library.


<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/lunarLander.gif" alt="LunarLander">

And finally, you'll **upload this trained agent to the Hugging Face Hub 🤗, a free, open platform where people can share ML models, datasets, and demos.**

It's essential **to master these elements** before diving into implementing Deep Reinforcement Learning agents. The goal of this chapter is to give you solid foundations.


After this unit, in a bonus unit, you'll be **able to train Huggy the Dog 🐶 to fetch the stick and play with him 🤗**.

<video src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit0/huggy.mp4" type="video/mp4" controls autoplay loop mute />

So let's get started! 🚀

---

<!-- what-is-rl -->

# What is Reinforcement Learning? [[what-is-reinforcement-learning]]

To understand Reinforcement Learning, let’s start with the big picture.

## The big picture [[the-big-picture]]

The idea behind Reinforcement Learning is that an agent (an AI) will learn from the environment by **interacting with it** (through trial and error) and **receiving rewards** (negative or positive) as feedback for performing actions.

Learning from interactions with the environment **comes from our natural experiences.**

For instance, imagine putting your little brother in front of a video game he never played, giving him a controller, and leaving him alone.


<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/Illustration_1.jpg" alt="Illustration_1" width="100%">

Your brother will interact with the environment (the video game) by pressing the right button (action). He got a coin, that’s a +1 reward. It’s positive, he just understood that in this game **he must get the coins.**

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/Illustration_2.jpg" alt="Illustration_2" width="100%">

But then, **he presses the right button again** and he touches an enemy. He just died, so that's a -1 reward.


<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/Illustration_3.jpg" alt="Illustration_3" width="100%">

By interacting with his environment through trial and error, your little brother understands that **he needs to get coins in this environment but avoid the enemies.**

**Without any supervision**, the child will get better and better at playing the game.

That’s how humans and animals learn, **through interaction.** Reinforcement Learning is just a **computational approach of learning from actions.**


### A formal definition [[a-formal-definition]]

We can now make a formal definition:

<Tip>
Reinforcement learning is a framework for solving control tasks (also called decision problems) by building agents that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.
</Tip>

But how does Reinforcement Learning work?

---

<!-- rl-framework -->

# The Reinforcement Learning Framework [[the-reinforcement-learning-framework]]

## The RL Process [[the-rl-process]]

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/RL_process.jpg" alt="The RL process" width="100%">
<figcaption>The RL Process: a loop of state, action, reward and next state</figcaption>
<figcaption>Source: <a href="http://incompleteideas.net/book/RLbook2020.pdf">Reinforcement Learning: An Introduction, Richard Sutton and Andrew G. Barto</a></figcaption>
</figure>

To understand the RL process, let’s imagine an agent learning to play a platform game:

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/RL_process_game.jpg" alt="The RL process" width="100%">

- Our Agent receives **state  \\(S_0\\)** from the **Environment** — we receive the first frame of our game (Environment).
- Based on that **state \\(S_0\\),** the Agent takes **action \\(A_0\\)** — our Agent will move to the right.
- The environment goes to a **new** **state \\(S_1\\)** — new frame.
- The environment gives some **reward \\(R_1\\)** to the Agent — we’re not dead *(Positive Reward +1)*.

This RL loop outputs a sequence of **state, action, reward and next state.**

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/sars.jpg" alt="State, Action, Reward, Next State" width="100%">

The agent's goal is to _maximize_ its cumulative reward, **called the expected return.**

## The reward hypothesis: the central idea of Reinforcement Learning [[reward-hypothesis]]

⇒ Why is the goal of the agent to maximize the expected return?

Because RL is based on the **reward hypothesis**, which is that all goals can be described as the **maximization of the expected return** (expected cumulative reward).

That’s why in Reinforcement Learning, **to have the best behavior,** we aim to learn to take actions that **maximize the expected cumulative reward.**


## Markov Property [[markov-property]]

In papers, you’ll see that the RL process is called a **Markov Decision Process** (MDP).

We’ll talk again about the Markov Property in the following units. But if you need to remember something today about it, it's this: the Markov Property implies that our agent needs **only the current state to decide** what action to take and **not the history of all the states and actions** they took before.

## Observations/States Space [[obs-space]]

Observations/States are the **information our agent gets from the environment.** In the case of a video game, it can be a frame (a screenshot). In the case of the trading agent, it can be the value of a certain stock, etc.

There is a differentiation to make between *observation* and *state*, however:

- *State s*: is **a complete description of the state of the world** (there is no hidden information). In a fully observed environment.


<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/chess.jpg" alt="Chess">
<figcaption>In chess game, we receive a state from the environment since we have access to the whole check board information.</figcaption>
</figure>

In a chess game, we have access to the whole board information, so we receive a state from the environment. In other words, the environment is fully observed.

- *Observation o*: is a **partial description of the state.** In a partially observed environment.

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/mario.jpg" alt="Mario">
<figcaption>In Super Mario Bros, we only see the part of the level close to the player, so we receive an observation.</figcaption>
</figure>

In Super Mario Bros, we only see the part of the level close to the player, so we receive an observation.

In Super Mario Bros, we are in a partially observed environment. We receive an observation **since we only see a part of the level.**

<Tip>
In this course, we use the term "state" to denote both state and observation, but we will make the distinction in implementations.
</Tip>

To recap:
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/obs_space_recap.jpg" alt="Obs space recap" width="100%">


## Action Space [[action-space]]

The Action space is the set of **all possible actions in an environment.**

The actions can come from a *discrete* or *continuous space*:

- *Discrete space*: the number of possible actions is **finite**.

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/mario.jpg" alt="Mario">
<figcaption>In Super Mario Bros, we have only 4 possible actions: left, right, up (jumping) and down (crouching).</figcaption>

</figure>

Again, in Super Mario Bros, we have a finite set of actions since we have only 4 directions.

- *Continuous space*: the number of possible actions is **infinite**.

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/self_driving_car.jpg" alt="Self Driving Car">
<figcaption>A Self Driving Car agent has an infinite number of possible actions since it can turn left 20°, 21,1°, 21,2°, honk, turn right 20°…
</figcaption>
</figure>

To recap:
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/action_space.jpg" alt="Action space recap" width="100%">

Taking this information into consideration is crucial because it will **have importance when choosing the RL algorithm in the future.**

## Rewards and the discounting [[rewards]]

The reward is fundamental in RL because it’s **the only feedback** for the agent. Thanks to it, our agent knows **if the action taken was good or not.**

The cumulative reward at each time step **t** can be written as:

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/rewards_1.jpg" alt="Rewards">
<figcaption>The cumulative reward equals the sum of all rewards in the sequence.
</figcaption>
</figure>

Which is equivalent to:

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/rewards_2.jpg" alt="Rewards">
<figcaption>The cumulative reward = rt+1 (rt+k+1 = rt+0+1 = rt+1)+ rt+2 (rt+k+1 = rt+1+1 = rt+2) + ...
</figcaption>
</figure>

However, in reality, **we can’t just add them like that.** The rewards that come sooner (at the beginning of the game) **are more likely to happen** since they are more predictable than the long-term future reward.

Let’s say your agent is this tiny mouse that can move one tile each time step, and your opponent is the cat (that can move too). The mouse's goal is **to eat the maximum amount of cheese before being eaten by the cat.**

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/rewards_3.jpg" alt="Rewards" width="100%">

As we can see in the diagram, **it’s more probable to eat the cheese near us than the cheese close to the cat** (the closer we are to the cat, the more dangerous it is).

Consequently, **the reward near the cat, even if it is bigger (more cheese), will be more discounted** since we’re not really sure we’ll be able to eat it.

To discount the rewards, we proceed like this:

1. We define a discount rate called gamma. **It must be between 0 and 1.** Most of the time between **0.95 and 0.99**.
- The larger the gamma, the smaller the discount. This means our agent **cares more about the long-term reward.**
- On the other hand, the smaller the gamma, the bigger the discount. This means our **agent cares more about the short term reward (the nearest cheese).**

2. Then, each reward will be discounted by gamma to the exponent of the time step. As the time step increases, the cat gets closer to us, **so the future reward is less and less likely to happen.**

Our discounted expected cumulative reward is:
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/rewards_4.jpg" alt="Rewards" width="100%">

---

<!-- two-methods -->

# Two main approaches for solving RL problems [[two-methods]]

<Tip>
Now that we learned the RL framework, how do we solve the RL problem?
</Tip>

In other words, how do we build an RL agent that can **select the actions that maximize its expected cumulative reward?**

## The Policy π: the agent’s brain [[policy]]

The Policy **π** is the **brain of our Agent**, it’s the function that tells us what **action to take given the state we are in.** So it **defines the agent’s behavior** at a given time.

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/policy_1.jpg" alt="Policy" />
<figcaption>Think of policy as the brain of our agent, the function that will tell us the action to take given a state</figcaption>
</figure>

This Policy **is the function we want to learn**, our goal is to find the optimal policy π\*, the policy that **maximizes expected return** when the agent acts according to it. We find this π\* **through training.**

There are two approaches to train our agent to find this optimal policy π\*:

- **Directly,** by teaching the agent to learn which **action to take,** given the current state: **Policy-Based Methods.**
- Indirectly, **teach the agent to learn which state is more valuable** and then take the action that **leads to the more valuable states**: Value-Based Methods.

## Policy-Based Methods [[policy-based]]

In Policy-Based methods, **we learn a policy function directly.**

This function will define a mapping from each state to the best corresponding action. Alternatively, it could define **a probability distribution over the set of possible actions at that state.**

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/policy_2.jpg" alt="Policy" />
<figcaption>As we can see here, the policy (deterministic) <b>directly indicates the action to take for each step.</b></figcaption>
</figure>


We have two types of policies:


- *Deterministic*: a policy at a given state **will always return the same action.**

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/policy_3.jpg" alt="Policy"/>
<figcaption>action = policy(state)</figcaption>
</figure>

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/policy_4.jpg" alt="Policy" width="100%"/>

- *Stochastic*: outputs **a probability distribution over actions.**

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/policy_5.jpg" alt="Policy"/>
<figcaption>policy(actions | state) = probability distribution over the set of actions given the current state</figcaption>
</figure>

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/policy-based.png" alt="Policy Based"/>
<figcaption>Given an initial state, our stochastic policy will output probability distributions over the possible actions at that state.</figcaption>
</figure>


If we recap:

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/pbm_1.jpg" alt="Pbm recap" width="100%" />
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/pbm_2.jpg" alt="Pbm recap" width="100%" />


## Value-based methods [[value-based]]

In value-based methods, instead of learning a policy function, we **learn a value function** that maps a state to the expected value **of being at that state.**

The value of a state is the **expected discounted return** the agent can get if it **starts in that state, and then acts according to our policy.**

“Act according to our policy” just means that our policy is **“going to the state with the highest value”.**

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/value_1.jpg" alt="Value based RL" width="100%" />

Here we see that our value function **defined values for each possible state.**

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/value_2.jpg" alt="Value based RL"/>
<figcaption>Thanks to our value function, at each step our policy will select the state with the biggest value defined by the value function: -7, then -6, then -5 (and so on) to attain the goal.</figcaption>
</figure>

Thanks to our value function, at each step our policy will select the state with the biggest value defined by the value function: -7, then -6, then -5 (and so on) to attain the goal.

If we recap:

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/vbm_1.jpg" alt="Vbm recap" width="100%" />
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/vbm_2.jpg" alt="Vbm recap" width="100%" />

---

<!-- deep-rl -->

# The “Deep” in Reinforcement Learning [[deep-rl]]

<Tip>
What we've talked about so far is Reinforcement Learning. But where does the "Deep" come into play?
</Tip>

Deep Reinforcement Learning introduces **deep neural networks to solve Reinforcement Learning problems** — hence the name “deep”.

For instance, in the next unit, we’ll learn about two value-based algorithms: Q-Learning (classic Reinforcement Learning) and then Deep Q-Learning.

You’ll see the difference is that, in the first approach, **we use a traditional algorithm** to create a Q table that helps us find what action to take for each state.

In the second approach, **we will use a Neural Network** (to approximate the Q value).

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/deep.jpg" alt="Value based RL"/>
<figcaption>Schema inspired by the Q learning notebook by Udacity
</figcaption>
</figure>

If you are not familiar with Deep Learning you should definitely watch [the FastAI Practical Deep Learning for Coders](https://course.fast.ai) (Free).

---

<!-- exp-exp-tradeoff -->

# The Exploration/Exploitation trade-off [[exp-exp-tradeoff]]

Finally, before looking at the different methods to solve Reinforcement Learning problems, we must cover one more very important topic: *the exploration/exploitation trade-off.*

- *Exploration* is exploring the environment by trying random actions in order to **find more information about the environment.**
- *Exploitation* is **exploiting known information to maximize the reward.**

Remember, the goal of our RL agent is to maximize the expected cumulative reward. However, **we can fall into a common trap**.

Let’s take an example:

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/exp_1.jpg" alt="Exploration" width="100%">

In this game, our mouse can have an **infinite amount of small cheese** (+1 each). But at the top of the maze, there is a gigantic sum of cheese (+1000).

However, if we only focus on exploitation, our agent will never reach the gigantic sum of cheese. Instead, it will only exploit **the nearest source of rewards,** even if this source is small (exploitation).

But if our agent does a little bit of exploration, it can **discover the big reward** (the pile of big cheese).

This is what we call the exploration/exploitation trade-off. We need to balance how much we **explore the environment** and how much we **exploit what we know about the environment.**

Therefore, we must **define a rule that helps to handle this trade-off**. We’ll see the different ways to handle it in the future units.

If it’s still confusing, **think of a real problem: the choice of picking a restaurant:**


<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/exp_2.jpg" alt="Exploration">
<figcaption>Source: <a href="https://inst.eecs.berkeley.edu/~cs188/sp20/assets/lecture/lec15_6up.pdf"> Berkley AI Course</a>
</figcaption>
</figure>

- *Exploitation*: You go to the same one that you know is good every day and **take the risk to miss another better restaurant.**
- *Exploration*: Try restaurants you never went to before, with the risk of having a bad experience **but the probable opportunity of a fantastic experience.**

To recap:
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/expexpltradeoff.jpg" alt="Exploration Exploitation Tradeoff" width="100%">

---

<!-- tasks -->

# Type of tasks [[tasks]]

A task is an **instance** of a Reinforcement Learning problem. We can have two types of tasks: **episodic** and **continuing**.

## Episodic task [[episodic-task]]

In this case, we have a starting point and an ending point **(a terminal state). This creates an episode**: a list of States, Actions, Rewards, and new States.

For instance, think about Super Mario Bros: an episode begins at the launch of a new Mario Level and ends **when you’re killed or you reached the end of the level.**

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/mario.jpg" alt="Mario">
<figcaption>Beginning of a new episode.
</figcaption>
</figure>


## Continuing tasks [[continuing-tasks]]

These are tasks that continue forever (**no terminal state**). In this case, the agent must **learn how to choose the best actions and simultaneously interact with the environment.**

For instance, an agent that does automated stock trading. For this task, there is no starting point and terminal state. **The agent keeps running until we decide to stop it.**

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/stock.jpg" alt="Stock Market" width="100%">

To recap:
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/tasks.jpg" alt="Tasks recap" width="100%">

---

<!-- summary -->

# Summary [[summary]]

That was a lot of information! Let's summarize:

- Reinforcement Learning is a computational approach of learning from actions. We build an agent that learns from the environment **by interacting with it through trial and error** and receiving rewards (negative or positive) as feedback.

- The goal of any RL agent is to maximize its expected cumulative reward (also called expected return) because RL is based on the **reward hypothesis**, which is that **all goals can be described as the maximization of the expected cumulative reward.**

- The RL process is a loop that outputs a sequence of **state, action, reward and next state.**

- To calculate the expected cumulative reward (expected return), we discount the rewards: the rewards that come sooner (at the beginning of the game) **are more probable to happen since they are more predictable than the long term future reward.**

- To solve an RL problem, you want to **find an optimal policy**. The policy is the “brain” of your agent, which will tell us **what action to take given a state.** The optimal policy is the one which **gives you the actions that maximize the expected return.**

- There are two ways to find your optimal policy:
    1. By training your policy directly: **policy-based methods.**
    2. By training a value function that tells us the expected return the agent will get at each state and use this function to define our policy: **value-based methods.**

- Finally, we speak about Deep RL because we introduce **deep neural networks to estimate the action to take (policy-based) or to estimate the value of a state (value-based)** hence the name “deep”.

---

<!-- glossary -->

# Glossary [[glossary]]

This is a community-created glossary. Contributions are welcome!

### Agent

An agent learns to **make decisions by trial and error, with rewards and punishments from the surroundings**.

### Environment

An environment is a simulated world **where an agent can learn by interacting with it**.

### Markov Property

It implies that the action taken by our agent is **conditional solely on the present state and independent of the past states and actions**.

### Observations/State

- **State**:  Complete description of the state of the world.
- **Observation**: Partial description of the state of the environment/world.

### Actions

- **Discrete Actions**: Finite number of actions, such as left, right, up, and down.
- **Continuous Actions**: Infinite possibility of actions; for example, in the case of self-driving cars, the driving scenario has an infinite possibility of actions occurring.

### Rewards and Discounting

- **Rewards**: Fundamental factor in RL. Tells the agent whether the action taken is good/bad.
- RL algorithms are focused on maximizing the **cumulative reward**.
- **Reward Hypothesis**: RL problems can be formulated as a maximisation of (cumulative) return.
- **Discounting** is performed because rewards obtained at the start are more likely to happen as they are more predictable than long-term rewards.

### Tasks

- **Episodic**: Has a starting point and an ending point.
- **Continuous**: Has a starting point but no ending point.

### Exploration v/s Exploitation Trade-Off

- **Exploration**: It's all about exploring the environment by trying random actions and receiving feedback/returns/rewards from the environment.
- **Exploitation**: It's about exploiting what we know about the environment to gain maximum rewards.
- **Exploration-Exploitation Trade-Off**: It balances how much we want to **explore** the environment and how much we want to **exploit** what we know about the environment.

### Policy

- **Policy**: It is called the agent's brain. It tells us what action to take, given the state.
- **Optimal Policy**: Policy that **maximizes** the **expected return** when an agent acts according to it. It is learned through *training*.

### Policy-based Methods:

- An approach to solving RL problems.
- In this method, the Policy is learned directly. 
- Will map each state to the best corresponding action at that state. Or a probability distribution over the set of possible actions at that state.

### Value-based Methods:

- Another approach to solving RL problems.
- Here, instead of training a policy, we train a **value function** that maps each state to the expected value of being in that state.

Contributions are welcome 🤗

If you want to improve the course, you can [open a Pull Request.](https://github.com/huggingface/deep-rl-class/pulls)

This glossary was made possible thanks to:

- [@lucifermorningstar1305](https://github.com/lucifermorningstar1305)
- [@daspartho](https://github.com/daspartho)
- [@misza222](https://github.com/misza222)


---

<!-- conclusion -->

# Conclusion [[conclusion]]

Congrats on finishing this unit! **That was the biggest one**, and there was a lot of information. And congrats on finishing the tutorial. You’ve just trained your first Deep RL agents and shared them with the community! 🥳

It's **normal if you still feel confused by some of these elements**. This was the same for me and for all people who studied RL.

**Take time to really grasp the material** before continuing. It’s important to master these elements and have a solid foundation before entering the fun part.

Naturally, during the course, we’re going to use and explain these terms again, but it’s better to understand them before diving into the next units.

In the next (bonus) unit, we’re going to reinforce what we just learned by **training Huggy the Dog to fetch a stick**.

You will then be able to play with him 🤗.

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/notebooks/unit-bonus1/huggy.jpg" alt="Huggy"/>

Finally, we would love **to hear what you think of the course and how we can improve it**. If you have some feedback then, please 👉  [fill this form](https://forms.gle/BzKXWzLAGZESGNaE9)

### Keep Learning, stay awesome 🤗


