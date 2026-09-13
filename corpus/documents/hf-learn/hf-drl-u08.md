# Hugging Face Deep RL Course — Unit 8: Proximal Policy Optimization (PPO)

Source: https://huggingface.co/learn/deep-rl-course/unit8/introduction


---

<!-- introduction -->

# Introduction [[introduction]]

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/thumbnail.png" alt="Unit 8"/>

In Unit 6, we learned about Advantage Actor Critic (A2C), a hybrid architecture combining value-based and policy-based methods that helps to stabilize the training by reducing the variance with:

- *An Actor* that controls **how our agent behaves** (policy-based method).
- *A Critic* that measures **how good the action taken is** (value-based method).

Today we'll learn about Proximal Policy Optimization (PPO), an architecture that **improves our agent's training stability by avoiding policy updates that are too large**. To do that, we use a ratio that indicates the difference between our current and old policy and clip this ratio to a specific range \\( [1 - \epsilon, 1 + \epsilon] \\) .

Doing this will ensure **that our policy update will not be too large and that the training is more stable.**

This Unit is in two parts:
- In this first part, you'll learn the theory behind PPO and code your PPO agent from scratch using the [CleanRL](https://github.com/vwxyzjn/cleanrl) implementation. To test its robustness you'll use LunarLander-v2. LunarLander-v2 **is the first environment you used when you started this course**. At that time, you didn't know how PPO worked, and now, **you can code it from scratch and train it. How incredible is that 🤩**.
- In the second part, we'll get deeper into PPO optimization by using [Sample-Factory](https://samplefactory.dev/) and train an agent playing vizdoom (an open source version of Doom).

<figure>
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit10/environments.png" alt="Environment"/>
<figcaption>These are the environments you're going to use to train your agents: VizDoom environments</figcaption>
</figure>

Sound exciting? Let's get started! 🚀

---

<!-- intuition-behind-ppo -->

# The intuition behind PPO [[the-intuition-behind-ppo]]


The idea with Proximal Policy Optimization (PPO) is that we want to improve the training stability of the policy by limiting the change you make to the policy at each training epoch: **we want to avoid having too large of a policy update.**

For two reasons:
- We know empirically that smaller policy updates during training are **more likely to converge to an optimal solution.**
- A too-big step in a policy update can result in falling “off the cliff” (getting a bad policy) **and taking a long time or even having no possibility to recover.**

<figure class="image table text-center m-0 w-full">
  <img class="center" src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/cliff.jpg" alt="Policy Update cliff"/>
  <figcaption>Taking smaller policy updates to improve the training stability</figcaption>
  <figcaption>Modified version from RL — Proximal Policy Optimization (PPO) <a href="https://jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12">Explained by Jonathan Hui</a></figcaption>
</figure>

**So with PPO, we update the policy conservatively**. To do so, we need to measure how much the current policy changed compared to the former one using a ratio calculation between the current and former policy. And we clip this ratio in a range \\( [1 - \epsilon, 1 + \epsilon] \\), meaning that we **remove the incentive for the current policy to go too far from the old one (hence the proximal policy term).**

---

<!-- clipped-surrogate-objective -->

# Introducing the Clipped Surrogate Objective Function
## Recap: The Policy Objective Function

Let’s remember what the objective is to optimize in Reinforce:
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/lpg.jpg" alt="Reinforce"/>

The idea was that by taking a gradient ascent step on this function (equivalent to taking gradient descent of the negative of this function), we would **push our agent to take actions that lead to higher rewards and avoid harmful actions.**

However, the problem comes from the step size:
- Too small, **the training process was too slow**
- Too high, **there was too much variability in the training**

With PPO, the idea is to constrain our policy update with a new objective function called the *Clipped surrogate objective function* that **will constrain the policy change in a small range using a clip.**

This new function **is designed to avoid destructively large weights updates** :

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/ppo-surrogate.jpg" alt="PPO surrogate function"/>

Let’s study each part to understand how it works.

## The Ratio Function
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/ratio1.jpg" alt="Ratio"/>

This ratio is calculated as follows:

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/ratio2.jpg" alt="Ratio"/>

It’s the probability of taking action \\( a_t \\) at state \\( s_t \\) in the current policy, divided by the same for the previous policy.

As we can see, \\( r_t(\theta) \\) denotes the probability ratio between the current and old policy:

- If \\( r_t(\theta) > 1 \\), the **action \\( a_t \\) at state \\( s_t \\) is more likely in the current policy than the old policy.**
- If \\( r_t(\theta) \\) is between 0 and 1, the **action is less likely for the current policy than for the old one**.

So this probability ratio is an **easy way to estimate the divergence between old and current policy.**

## The unclipped part of the Clipped Surrogate Objective function
<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/unclipped1.jpg" alt="PPO"/>

This ratio **can replace the log probability we use in the policy objective function**. This gives us the left part of the new objective function: multiplying the ratio by the advantage.
<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/unclipped2.jpg" alt="PPO"/>
  <figcaption><a href="https://arxiv.org/pdf/1707.06347.pdf">Proximal Policy Optimization Algorithms</a></figcaption>
</figure>

However, without a constraint, if the action taken is much more probable in our current policy than in our former, **this would lead to a significant policy gradient step** and, therefore, an **excessive policy update.**

## The clipped Part of the Clipped Surrogate Objective function

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/clipped.jpg" alt="PPO"/>

Consequently, we need to constrain this objective function by penalizing changes that lead to a ratio far away from 1 (in the paper, the ratio can only vary from 0.8 to 1.2).

**By clipping the ratio, we ensure that we do not have a too large policy update because the current policy can't be too different from the older one.**

To do that, we have two solutions:

- *TRPO (Trust Region Policy Optimization)* uses KL divergence constraints outside the objective function to constrain the policy update. But this method **is complicated to implement and takes more computation time.**
- *PPO* clip probability ratio directly in the objective function with its **Clipped surrogate objective function.**

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/clipped.jpg" alt="PPO"/>

This clipped part is a version where \\( r_t(\theta) \\) is clipped between  \\( [1 - \epsilon, 1 + \epsilon] \\).

With the Clipped Surrogate Objective function, we have two probability ratios, one non-clipped and one clipped in a range between  \\( [1 - \epsilon, 1 + \epsilon] \\), epsilon is a hyperparameter that helps us to define this clip range (in the paper  \\( \epsilon = 0.2 \\).).

Then, we take the minimum of the clipped and non-clipped objective, **so the final objective is a lower bound (pessimistic bound) of the unclipped objective.**

Taking the minimum of the clipped and non-clipped objective means **we'll select either the clipped or the non-clipped objective based on the ratio and advantage situation**.

---

<!-- visualize -->

# Visualize the Clipped Surrogate Objective Function

Don't worry. **It's normal if this seems complex to handle right now**. But we're going to see what this Clipped Surrogate Objective Function looks like, and this will help you to visualize better what's going on.

<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/recap.jpg" alt="PPO"/>
  <figcaption><a href="https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf">Table from "Towards Delivering a Coherent Self-Contained
    Explanation of Proximal Policy Optimization" by Daniel Bick</a></figcaption>
</figure>

We have six different situations. Remember first that we take the minimum between the clipped and unclipped objectives.

## Case 1 and 2: the ratio is between the range

In situations 1 and 2, **the clipping does not apply since the ratio is between the range** \\( [1 - \epsilon, 1 + \epsilon] \\)

In situation 1, we have a positive advantage: the **action is better than the average** of all the actions in that state. Therefore, we should encourage our current policy to increase the probability of taking that action in that state.

Since the ratio is between intervals, **we can increase our policy's probability of taking that action at that state.**

In situation 2, we have a negative advantage: the action is worse than the average of all actions at that state. Therefore, we should discourage our current policy from taking that action in that state.

Since the ratio is between intervals, **we can decrease the probability that our policy takes that action at that state.**

## Case 3 and 4: the ratio is below the range
<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/recap.jpg" alt="PPO"/>
  <figcaption><a href="https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf">Table from "Towards Delivering a Coherent Self-Contained
    Explanation of Proximal Policy Optimization" by Daniel Bick</a></figcaption>
</figure>

If the probability ratio is lower than \\( [1 - \epsilon] \\), the probability of taking that action at that state is much lower than with the old policy.

If, like in situation 3, the advantage estimate is positive (A>0), then **you want to increase the probability of taking that action at that state.**

But if, like situation 4, the advantage estimate is negative, **we don't want to decrease further** the probability of taking that action at that state. Therefore, the gradient is = 0 (since we're on a flat line), so we don't update our weights.

## Case 5 and 6: the ratio is above the range
<figure class="image table text-center m-0 w-full">
  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/recap.jpg" alt="PPO"/>
  <figcaption><a href="https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf">Table from "Towards Delivering a Coherent Self-Contained
    Explanation of Proximal Policy Optimization" by Daniel Bick</a></figcaption>
</figure>

If the probability ratio is higher than \\( [1 + \epsilon] \\), the probability of taking that action at that state in the current policy is **much higher than in the former policy.**

If, like in situation 5, the advantage is positive, **we don't want to get too greedy**. We already have a higher probability of taking that action at that state than the former policy. Therefore, the gradient is = 0 (since we're on a flat line), so we don't update our weights.

If, like in situation 6, the advantage is negative, we want to decrease the probability of taking that action at that state.

So if we recap, **we only update the policy with the unclipped objective part**. When the minimum is the clipped objective part, we don't update our policy weights since the gradient will equal 0.

So we update our policy  only if:
- Our ratio is in the range \\( [1 - \epsilon, 1 + \epsilon] \\)
- Our ratio is outside the range, but **the advantage leads to getting closer to the range**
    - Being below the ratio but the advantage is > 0
    - Being above the ratio but the advantage is < 0

**You might wonder why, when the minimum is the clipped ratio, the gradient is 0.** When the ratio is clipped, the derivative in this case will not be the derivative of the \\( r_t(\theta) * A_t \\)   but the derivative of either \\( (1 - \epsilon)* A_t\\) or the derivative of \\( (1 + \epsilon)* A_t\\) which both = 0.


To summarize, thanks to this clipped surrogate objective, **we restrict the range that the current policy can vary from the old one.** Because we remove the incentive for the probability ratio to move outside of the interval since the clip forces the gradient to be zero. If the ratio is > \\( 1 + \epsilon \\) or < \\( 1 - \epsilon \\) the gradient will be equal to 0.

The final Clipped Surrogate Objective Loss for PPO Actor-Critic style looks like this, it's a combination of Clipped Surrogate Objective function, Value Loss Function and Entropy bonus:

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/ppo-objective.jpg" alt="PPO objective"/>

That was quite complex. Take time to understand these situations by looking at the table and the graph. **You must understand why this makes sense.** If you want to go deeper, the best resource is the article ["Towards Delivering a Coherent Self-Contained Explanation of Proximal Policy Optimization" by Daniel Bick, especially part 3.4](https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf).

---

<!-- conclusion -->

# Conclusion [[Conclusion]]

That’s all for today. Congrats on finishing this unit and the tutorial!

The best way to learn is to practice and try stuff. **Why not improve the implementation to handle frames as input?**.

See you on second part of this Unit 🔥

## Keep Learning, Stay awesome 🤗

---

<!-- introduction-sf -->

# Introduction to PPO with Sample-Factory

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit9/thumbnail2.png" alt="thumbnail"/>

In this second part of Unit 8, we'll get deeper into PPO optimization by using [Sample-Factory](https://samplefactory.dev/), an **asynchronous implementation of the PPO algorithm**, to train our agent to play [vizdoom](https://vizdoom.cs.put.edu.pl/) (an open source version of Doom).

In the notebook, **you'll train your agent to play the Health Gathering level**, where the agent must collect health packs to avoid dying. After that, you can **train your agent to play more complex levels, such as Deathmatch**.

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit10/environments.png" alt="Environment"/>

Sound exciting? Let's get started! 🚀

The hands-on is made by [Edward Beeching](https://twitter.com/edwardbeeching), a Machine Learning Research Scientist at Hugging Face. He worked on Godot Reinforcement Learning Agents, an open-source interface for developing environments and agents in the Godot Game Engine.

---

<!-- conclusion-sf -->

# Conclusion

That's all for today. Congrats on finishing this Unit and the tutorial! ⭐️

Now that you've successfully trained your Doom agent, why not try deathmatch? Remember, that's a much more complex level than the one you've just trained, **but it's a nice experiment and I advise you to try it.**

If you do it, don't hesitate to share your model in the `#rl-i-made-this` channel in our [discord server](https://www.hf.co/join/discord).

This concludes the last unit, but we are not finished yet! 🤗 The following **bonus unit includes some of the most interesting, advanced, and cutting edge work in Deep Reinforcement Learning**.

See you next time 🔥

## Keep Learning, Stay awesome 🤗
