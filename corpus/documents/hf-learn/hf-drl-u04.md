# Hugging Face Deep RL Course — Unit 4: Policy Gradient with PyTorch

Source: https://huggingface.co/learn/deep-rl-course/unit4/introduction


---

<!-- introduction -->

# Introduction [[introduction]]

  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/thumbnail.png" alt="thumbnail"/>

In the last unit, we learned about Deep Q-Learning. In this value-based deep reinforcement learning algorithm, we **used a deep neural network to approximate the different Q-values for each possible action at a state.**

Since the beginning of the course, we have only studied value-based methods, **where we estimate a value function as an intermediate step towards finding an optimal policy.**

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/link-value-policy.jpg" alt="Link value policy" />

In value-based methods, the policy ** \(π\) only exists because of the action value estimates since the policy is just a function** (for instance, greedy-policy) that will select the action with the highest value given a state.

With policy-based methods, we want to optimize the policy directly **without having an intermediate step of learning a value function.**

So today, **we'll learn about policy-based methods and study a subset of these methods called policy gradient**. Then we'll implement our first policy gradient algorithm called Monte Carlo **Reinforce** from scratch using PyTorch.
Then, we'll test its robustness using the CartPole-v1 and PixelCopter environments.

You'll then be able to iterate and improve this implementation for more advanced environments.

<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/envs.gif" alt="Environments"/>
</figure>

Let's get started!

---

<!-- what-are-policy-based-methods -->

# What are the policy-based methods?

The main goal of Reinforcement learning is to **find the optimal policy \\(\pi^{*}\\) that will maximize the expected cumulative reward**.
Because Reinforcement Learning is based on the *reward hypothesis*: **all goals can be described as the maximization of the expected cumulative reward.**

For instance, in a soccer game (where you're going to train the agents in two units), the goal is to win the game. We can describe this goal in reinforcement learning as
**maximizing the number of goals scored** (when the ball crosses the goal line) into your opponent's soccer goals. And **minimizing the number of goals in your soccer goals**.

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/soccer.jpg" alt="Soccer" />

## Value-based, Policy-based, and Actor-critic methods

In the first unit, we saw two methods to find (or, most of the time, approximate) this optimal policy \\(\pi^{*}\\).

- In *value-based methods*, we learn a value function.
  - The idea is that an optimal value function leads to an optimal policy \\(\pi^{*}\\).
  - Our objective is to **minimize the loss between the predicted and target value** to approximate the true action-value function.
  - We have a policy, but it's implicit since it **is generated directly from the value function**. For instance, in Q-Learning, we used an (epsilon-)greedy policy.

- On the other hand, in *policy-based methods*, we directly learn to approximate \\(\pi^{*}\\) without having to learn a value function.
  - The idea is **to parameterize the policy**. For instance, using a neural network \\(\pi_\theta\\), this policy will output a probability distribution over actions (stochastic policy).
  - <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/stochastic_policy.png" alt="stochastic policy" />
  - Our objective then is **to maximize the performance of the parameterized policy using gradient ascent**.
  - To do that, we control the parameter \\(\theta\\) that will affect the distribution of actions over a state.

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/policy_based.png" alt="Policy based" />

- Next time, we'll study the *actor-critic* method, which is a combination of value-based and policy-based methods.

Consequently, thanks to policy-based methods, we can directly optimize our policy \\(\pi_\theta\\) to output a probability distribution over actions \\(\pi_\theta(a|s)\\) that leads to the best cumulative return.
To do that, we define an objective function \\(J(\theta)\\), that is, the expected cumulative reward, and we **want to find the value \\(\theta\\) that maximizes this objective function**.

## The difference between policy-based and policy-gradient methods

Policy-gradient methods, what we're going to study in this unit, is a subclass of policy-based methods. In policy-based methods, the optimization is most of the time *on-policy* since for each update, we only use data (trajectories) collected **by our most recent version of** \\(\pi_\theta\\).

The difference between these two methods **lies on how we optimize the parameter** \\(\theta\\):

- In *policy-based methods*, we search directly for the optimal policy. We can optimize the parameter \\(\theta\\) **indirectly** by maximizing the local approximation of the objective function with techniques like hill climbing, simulated annealing, or evolution strategies.
- In *policy-gradient methods*, because it is a subclass of the policy-based methods, we search directly for the optimal policy. But we optimize the parameter \\(\theta\\) **directly** by performing the gradient ascent on the performance of the objective function \\(J(\theta)\\).

Before diving more into how policy-gradient methods work (the objective function, policy gradient theorem, gradient ascent, etc.), let's study the advantages and disadvantages of policy-based methods.

---

<!-- advantages-disadvantages -->

# The advantages and disadvantages of policy-gradient methods

At this point, you might ask, "but Deep Q-Learning is excellent! Why use policy-gradient methods?". To answer this question, let's study the **advantages and disadvantages of policy-gradient methods**.

## Advantages

There are multiple advantages over value-based methods. Let's see some of them:

### The simplicity of integration

We can estimate the policy directly without storing additional data (action values).

### Policy-gradient methods can learn a stochastic policy

Policy-gradient methods can **learn a stochastic policy while value functions can't**.

This has two consequences:

1. We **don't need to implement an exploration/exploitation trade-off by hand**. Since we output a probability distribution over actions, the agent explores **the state space without always taking the same trajectory.**

2. We also get rid of the problem of **perceptual aliasing**. Perceptual aliasing is when two states seem (or are) the same but need different actions.

Let's take an example: we have an intelligent vacuum cleaner whose goal is to suck the dust and avoid killing the hamsters.

<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/hamster1.jpg" alt="Hamster 1"/>
</figure>

Our vacuum cleaner can only perceive where the walls are.

The problem is that the **two red (colored) states are aliased states because the agent perceives an upper and lower wall for each**.

<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/hamster2.jpg" alt="Hamster 1"/>
</figure>

Under a deterministic policy, the policy will either always move right when in a red state or always move left. **Either case will cause our agent to get stuck and never suck the dust**.

Under a value-based Reinforcement learning algorithm, we learn a **quasi-deterministic policy** ("greedy epsilon strategy"). Consequently, our agent can **spend a lot of time before finding the dust**.

On the other hand, an optimal stochastic policy **will randomly move left or right in red (colored) states**. Consequently, **it will not be stuck and will reach the goal state with a high probability**.

<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/hamster3.jpg" alt="Hamster 1"/>
</figure>

### Policy-gradient methods are more effective in high-dimensional action spaces and continuous actions spaces

The problem with Deep Q-learning is that their **predictions assign a score (maximum expected future reward) for each possible action**, at each time step, given the current state.

But what if we have an infinite possibility of actions?

For instance, with a self-driving car, at each state, you can have a (near) infinite choice of actions (turning the wheel at 15°, 17.2°, 19,4°, honking, etc.). **We'll need to output a Q-value for each possible action**! And **taking the max action of a continuous output is an optimization problem itself**!

Instead, with policy-gradient methods, we output a **probability distribution over actions.**

### Policy-gradient methods have better convergence properties

In value-based methods, we use an aggressive operator to **change the value function: we take the maximum over Q-estimates**.
Consequently, the action probabilities may change dramatically for an arbitrarily small change in the estimated action values if that change results in a different action having the maximal value.

For instance, if during the training, the best action was left (with a Q-value of 0.22) and the training step after it's right (since the right Q-value becomes 0.23), we dramatically changed the policy since now the policy will take most of the time right instead of left.

On the other hand, in policy-gradient methods, stochastic policy action preferences (probability of taking action) **change smoothly over time**.

## Disadvantages

Naturally, policy-gradient methods also have some disadvantages:

- **Frequently, policy-gradient methods converges to a local maximum instead of a global optimum.**
- Policy-gradient goes slower, **step by step: it can take longer to train (inefficient).**
- Policy-gradient can have high variance. We'll see in the actor-critic unit why, and how we can solve this problem.

👉 If you want to go deeper into the advantages and disadvantages of policy-gradient methods, [you can check this video](https://youtu.be/y3oqOjHilio).

---

<!-- policy-gradient -->

# Diving deeper into policy-gradient methods

## Getting the big picture

We just learned that policy-gradient methods aim to find parameters  \\( \theta \\) that **maximize the expected return**.

The idea is that we have a *parameterized stochastic policy*. In our case, a neural network outputs a probability distribution over actions. The probability of taking each action is also called the *action preference*.

If we take the example of CartPole-v1:
- As input, we have a state.
- As output, we have a probability distribution over actions at that state.

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/policy_based.png" alt="Policy based" />

Our goal with policy-gradient is to **control the probability distribution of actions** by tuning the policy such that **good actions (that maximize the return) are sampled more frequently in the future.**
Each time the agent interacts with the environment, we tweak the parameters such that good actions will be sampled more likely in the future.

But **how are we going to optimize the weights using the expected return**?

The idea is that we're going to **let the agent interact during an episode**. And if we win the episode, we consider that each action taken was good and must be more sampled in the future
since they lead to win.

So for each state-action pair, we want to increase the  \\(P(a|s)\\): the probability of taking that action at that state. Or decrease if we lost.

The Policy-gradient algorithm (simplified) looks like this:
<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/pg_bigpicture.jpg" alt="Policy Gradient Big Picture"/>
</figure>

Now that we got the big picture, let's dive deeper into policy-gradient methods.

## Diving deeper into policy-gradient methods

We have our stochastic policy  \\(\pi\\) which has a parameter  \\(\theta\\). This  \\(\pi\\), given a state, **outputs a probability distribution of actions**.

<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/stochastic_policy.png" alt="Policy"/>
</figure>

Where  \\(\pi_\theta(a_t|s_t)\\) is the probability of the agent selecting action  \\(a_t\\) from state  \\(s_t\\) given our policy.

**But how do we know if our policy is good?** We need to have a way to measure it. To know that, we define a score/objective function called  \\(J(\theta)\\).

### The objective function

The *objective function* gives us the **performance of the agent** given a trajectory (state action sequence without considering reward (contrary to an episode)), and it outputs the *expected cumulative reward*.

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/objective.jpg" alt="Return"/>

Let's give some more details on this formula:
- The *expected return* (also called expected cumulative reward), is the weighted average (where the weights are given by  \\(P(\tau;\theta)\\) of all possible values that the return  \\(R(\tau)\\) can take).

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/expected_reward.png" alt="Return"/>


- \\(R(\tau)\\) :  Return from an arbitrary trajectory. To take this quantity and use it to calculate the expected return, we need to multiply it by the probability of each possible trajectory.

- \\(P(\tau;\theta)\\) : Probability of each possible trajectory  \\(\tau\\) (that probability depends on  \\(\theta\\) since it defines the policy that it uses to select the actions of the trajectory which has an impact of the states visited).

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/probability.png" alt="Probability"/>

- \\(J(\theta)\\) : Expected return, we calculate it by summing for all trajectories, the probability of taking that trajectory given  \\(\theta \\) multiplied by the return of this trajectory.

Our objective then is to maximize the expected cumulative reward by finding the  \\(\theta \\) that will output the best action probability distributions:


<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/max_objective.png" alt="Max objective"/>


## Gradient Ascent and the Policy-gradient Theorem

Policy-gradient is an optimization problem: we want to find the values of  \\(\theta\\) that maximize our objective function  \\(J(\theta)\\), so we need to use **gradient-ascent**. It's the inverse of *gradient-descent* since it gives the direction of the steepest increase of  \\(J(\theta)\\).

(If you need a refresher on the difference between gradient descent and gradient ascent [check this](https://www.baeldung.com/cs/gradient-descent-vs-ascent) and [this](https://stats.stackexchange.com/questions/258721/gradient-ascent-vs-gradient-descent-in-logistic-regression)).

Our update step for gradient-ascent is:

\\( \theta \leftarrow \theta + \alpha *  \nabla_\theta J(\theta) \\)

We can repeatedly apply this update in the hopes that  \\(\theta \\) converges to the value that maximizes  \\(J(\theta)\\).

However, there are two problems with computing the derivative of  \\(J(\theta)\\):
1. We can't calculate the true gradient of the objective function since it requires calculating the probability of each possible trajectory, which is computationally super expensive.
So we want to **calculate a gradient estimation with a sample-based estimate (collect some trajectories)**.

2. We have another problem that I explain in the next optional section. To differentiate this objective function, we need to differentiate the state distribution, called the Markov Decision Process dynamics. This is attached to the environment. It gives us the probability of the environment going into the next state, given the current state and the action taken by the agent. The problem is that we can't differentiate it because we might not know about it.

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/probability.png" alt="Probability"/>

Fortunately we're going to use a solution called the Policy Gradient Theorem that will help us to reformulate the objective function into a differentiable function that does not involve the differentiation of the state distribution.

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/policy_gradient_theorem.png" alt="Policy Gradient"/>

If you want to understand how we derive this formula for approximating the gradient, check out the next (optional) section.

## The Reinforce algorithm (Monte Carlo Reinforce)

The Reinforce algorithm, also called Monte-Carlo policy-gradient, is a policy-gradient algorithm that **uses an estimated return from an entire episode to update the policy parameter**  \\(\theta\\):

In a loop:
- Use the policy  \\(\pi_\theta\\)  to collect an episode  \\(\tau\\)
- Use the episode to estimate the gradient  \\(\hat{g} = \nabla_\theta J(\theta)\\)

 <figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/policy_gradient_one.png" alt="Policy Gradient"/>
</figure>

- Update the weights of the policy:  \\(\theta \leftarrow \theta + \alpha \hat{g}\\)

We can interpret this update as follows:

- \\(\nabla_\theta log \pi_\theta(a_t|s_t)\\) is the direction of **steepest increase of the (log) probability** of selecting action \\(a_t\\) from state \\(s_t\\).
This tells us **how we should change the weights of policy** if we want to increase/decrease the log probability of selecting action \\(a_t\\) at state \\(s_t\\).

- \\(R(\tau)\\): is the scoring function:
  - If the return is high, it will **push up the probabilities** of the (state, action) combinations.
  - Otherwise, if the return is low, it will **push down the probabilities** of the (state, action) combinations.


We can also **collect multiple episodes (trajectories)** to estimate the gradient:
<figure class="image table text-center m-0 w-full">
 <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/policy_gradient_multiple.png" alt="Policy Gradient"/>
</figure>

---

<!-- pg-theorem -->

# (Optional) the Policy Gradient Theorem

In this optional section where we're **going to study how we differentiate the objective function that we will use to approximate the policy gradient**.

Let's first recap our different formulas:

1. The Objective function

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/expected_reward.png" alt="Return"/>


2. The probability of a trajectory (given that action comes from \\(\pi_\theta\\)):

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/probability.png" alt="Probability"/>


So we have:

\\(\nabla_\theta J(\theta) =  \nabla_\theta \sum_{\tau}P(\tau;\theta)R(\tau)\\)


We can rewrite the gradient of the sum as the sum of the gradient:

\\( =  \sum_{\tau} \nabla_\theta (P(\tau;\theta)R(\tau)) = \sum_{\tau} \nabla_\theta P(\tau;\theta)R(\tau) \\) as \\(R(\tau)\\) is not dependent on \\(\theta\\)

We then multiply every term in the sum by \\(\frac{P(\tau;\theta)}{P(\tau;\theta)}\\)(which is possible since it's = 1)

\\( = \sum_{\tau} \frac{P(\tau;\theta)}{P(\tau;\theta)}\nabla_\theta P(\tau;\theta)R(\tau) \\)


We can simplify further this since \\( \frac{P(\tau;\theta)}{P(\tau;\theta)}\nabla_\theta P(\tau;\theta) =  P(\tau;\theta)\frac{\nabla_\theta P(\tau;\theta)}{P(\tau;\theta)}  \\). 

Thus we can rewrite the sum as

\\( P(\tau;\theta)\frac{\nabla_\theta P(\tau;\theta)}{P(\tau;\theta)}= \sum_{\tau} P(\tau;\theta) \frac{\nabla_\theta P(\tau;\theta)}{P(\tau;\theta)}R(\tau) \\)

We can then use the *derivative log trick* (also called *likelihood ratio trick* or *REINFORCE trick*), a simple rule in calculus that implies that \\( \nabla_x log f(x) = \frac{\nabla_x f(x)}{f(x)} \\)

So given we have \\(\frac{\nabla_\theta P(\tau;\theta)}{P(\tau;\theta)} \\) we transform it as \\(\nabla_\theta log P(\tau|\theta) \\)



So this is our likelihood policy gradient:

\\( \nabla_\theta J(\theta) = \sum_{\tau} P(\tau;\theta)  \nabla_\theta log P(\tau;\theta) R(\tau) \\)





Thanks for this new formula, we can estimate the gradient using trajectory samples (we can approximate the likelihood ratio policy gradient with sample-based estimate if you prefer).

\\(\nabla_\theta J(\theta) = \frac{1}{m} \sum^{m}_{i=1} \nabla_\theta log P(\tau^{(i)};\theta)R(\tau^{(i)})\\) where each \\(\tau^{(i)}\\) is a sampled trajectory.


But we still have some mathematics work to do there: we need to simplify \\(  \nabla_\theta log P(\tau|\theta) \\)

We know that:

\\(\nabla_\theta log P(\tau^{(i)};\theta)= \nabla_\theta log[ \mu(s_0) \prod_{t=0}^{H} P(s_{t+1}^{(i)}|s_{t}^{(i)}, a_{t}^{(i)}) \pi_\theta(a_{t}^{(i)}|s_{t}^{(i)})]\\)

Where \\(\mu(s_0)\\) is the initial state distribution and \\( P(s_{t+1}^{(i)}|s_{t}^{(i)}, a_{t}^{(i)})  \\) is the state transition dynamics of the MDP.

We know that the log of a product is equal to the sum of the logs:

\\(\nabla_\theta log P(\tau^{(i)};\theta)= \nabla_\theta \left[log \mu(s_0) + \sum\limits_{t=0}^{H}log P(s_{t+1}^{(i)}|s_{t}^{(i)} a_{t}^{(i)}) + \sum\limits_{t=0}^{H}log \pi_\theta(a_{t}^{(i)}|s_{t}^{(i)})\right] \\)

We also know that the gradient of the sum is equal to the sum of gradient:

\\( \nabla_\theta log P(\tau^{(i)};\theta)=\nabla_\theta log\mu(s_0) + \nabla_\theta \sum\limits_{t=0}^{H} log P(s_{t+1}^{(i)}|s_{t}^{(i)} a_{t}^{(i)}) + \nabla_\theta \sum\limits_{t=0}^{H} log \pi_\theta(a_{t}^{(i)}|s_{t}^{(i)}) \\)


Since neither initial state distribution or state transition dynamics of the MDP are dependent of \\(\theta\\), the derivate of both terms are 0. So we can remove them:

Since:
\\(\nabla_\theta \sum_{t=0}^{H} log P(s_{t+1}^{(i)}|s_{t}^{(i)}  a_{t}^{(i)}) = 0 \\) and \\( \nabla_\theta \mu(s_0) = 0\\)

\\(\nabla_\theta log P(\tau^{(i)};\theta) =   \nabla_\theta \sum_{t=0}^{H} log \pi_\theta(a_{t}^{(i)}|s_{t}^{(i)})\\)

We can rewrite the gradient of the sum as the sum of gradients:

\\( \nabla_\theta log P(\tau^{(i)};\theta)=    \sum_{t=0}^{H} \nabla_\theta log \pi_\theta(a_{t}^{(i)}|s_{t}^{(i)}) \\)

So, the final formula for estimating the policy gradient is:

\\( \nabla_{\theta} J(\theta) = \hat{g} = \frac{1}{m} \sum^{m}_{i=1} \sum^{H}_{t=0} \nabla_\theta \log \pi_\theta(a^{(i)}_{t} | s_{t}^{(i)})R(\tau^{(i)}) \\)

---

<!-- glossary -->

# Glossary 

This is a community-created glossary. Contributions are welcome!

- **Deep Q-Learning:** A value-based deep reinforcement learning algorithm that uses a deep neural network to approximate Q-values for actions in a given state. The goal of Deep Q-learning is to find the optimal policy that maximizes the expected cumulative reward by learning the action-values.

- **Value-based methods:** Reinforcement Learning methods that estimate a value function as an intermediate step towards finding an optimal policy.

- **Policy-based methods:** Reinforcement Learning methods that directly learn to approximate the optimal policy without learning a value function. In practice they output a probability distribution over actions. 

    The benefits of using policy-gradient methods over value-based methods include: 
    - simplicity of integration: no need to store action values;
    - ability to learn a stochastic policy: the agent explores the state space without always taking the same trajectory, and avoids the problem of perceptual aliasing;
    - effectiveness in high-dimensional and continuous action spaces; and
    - improved convergence properties.

- **Policy Gradient:** A subset of policy-based methods where the objective is to maximize the performance of a parameterized policy using gradient ascent. The goal of a policy-gradient is to control the probability distribution of actions by tuning the policy such that good actions (that maximize the return) are sampled more frequently in the future. 

- **Monte Carlo Reinforce:** A policy-gradient algorithm that uses an estimated return from an entire episode to update the policy parameter.

If you want to improve the course, you can [open a Pull Request.](https://github.com/huggingface/deep-rl-class/pulls)

This glossary was made possible thanks to:

- [Diego Carpintero](https://github.com/dcarpintero)
---

<!-- conclusion -->

# Conclusion


**Congrats on finishing this unit**! There was a lot of information.
And congrats on finishing the tutorial. You've just coded your first Deep Reinforcement Learning agent from scratch using PyTorch and shared it on the Hub 🥳.

Don't hesitate to iterate on this unit **by improving the implementation for more complex environments** (for instance, what about changing the network to a Convolutional Neural Network to handle
frames as observation)?

In the next unit, **we're going to learn more about Unity MLAgents**, by training agents in Unity environments. This way, you will be ready to participate in the **AI vs AI challenges where you'll train your agents
to compete against other agents in a snowball fight and a soccer game.**

Sound fun? See you next time!

Finally, we would love **to hear what you think of the course and how we can improve it**. If you have some feedback then please 👉  [fill this form](https://forms.gle/BzKXWzLAGZESGNaE9)

### Keep Learning, stay awesome 🤗
