Course.module({
  id: "m21-rl-fundamentals",
  title: "Reinforcement learning fundamentals",
  short: "RL fundamentals",
  tagline: "Train agents yourself — a from-scratch Q-learner on a gridworld, then PPO on CartPole — and understand every line of the code that later trains reasoning LLMs.",
  hours: 16,
  level: "core",
  runsOn: ["mac"],
  tags: ["rl", "q-learning", "policy-gradient", "ppo", "gymnasium", "cleanrl"],

  goal: MD(function () {/*
By the end of this module two agents that **you** trained are running on your Mac CPU:

1. A **tabular Q-learning** agent (≈40 lines of NumPy, no libraries) that learns to cross a gridworld full of holes:

```text
episode     0  avg return (last 100) = -1.080
episode   100  avg return (last 100) = +0.522
episode  1999  avg return (last 100) = +0.728
learned policy:        V(start) = 0.729  (theory: 0.729 ✓)
  v > v <
  v H v H
  > > v H
  H > > G
```

2. **PPO on CartPole** with CleanRL’s single-file `ppo.py` — which you will have read *line by line* — plus your own REINFORCE, with learning curves over 5 seeds:

```text
global_step=2048,   episodic_return=[23.]
global_step=40960,  episodic_return=[187.]
global_step=98304,  episodic_return=[500.]      # CartPole-v1 max
SPS: 2400  (Mac CPU; no GPU needed)
```

Play with the agent below before reading anything: drag ε (exploration), α (learning rate) and γ (discount) and watch the value heatmap and arrows form.
  */}),
  demo: { viz: "gridworld", params: {} },

  why: MD(function () {/*
Every frontier LLM is finished with **reinforcement learning**: RLHF made ChatGPT helpful, and RL with verifiable rewards (GRPO) made o1/R1-style models reason. The algorithm inside those systems is a direct descendant of what you build here — **PPO’s clipped ratio, advantages, baselines and KL penalties are the same equations, just with tokens as actions**.

For an inference engineer this matters concretely: RL post-training spends most of its GPU time **generating rollouts** — i.e. running inference. RL-infra and post-training-inference roles (xAI “RL Inference”, Together “Post-Training Inference”, Scale RLXF) expect you to know PPO/GRPO/REINFORCE well enough to know *which numbers must match between trainer and inference engine and why*. This module gives you that vocabulary on problems small enough to run in seconds.
  */}),

  prereqs: [
    {
      title: "PyTorch basics: tensors, nn.Module, loss.backward(), optimizer.step()",
      skipIf: "you finished M01–M03 (you trained a GPT)",
      md: MD(function () {/*
You only need the training loop pattern from Phase 1:

```python
logits = model(x)                 # forward
loss = some_function_of(logits)   # a scalar
opt.zero_grad(); loss.backward(); opt.step()   # gradients → update
```

In RL the only thing that changes is **where the loss comes from**: not from labels, but from rewards the agent collected itself. If this loop is unfamiliar, do M01 first.
      */}),
    },
    {
      title: "Probability in one minute: distributions, sampling, log-probabilities",
      skipIf: "you know what `torch.distributions.Categorical(logits=...).log_prob(a)` returns",
      math: true,
      md: MD(function () {/*
- A **probability distribution** over actions is a list of non-negative numbers that sum to 1, e.g. `[0.7, 0.2, 0.1]` for (left, right, stay).
- A network outputs **logits**; `softmax` turns them into probabilities (you did this in M02 for next-token prediction — it is *exactly* the same here).
- **Sampling** = picking action 0 with probability 0.7, etc. That randomness is how the agent explores.
- The **log-probability** $\log \pi(a)$ of the action you took is what policy-gradient methods push up or down. $\log 0.7 \approx -0.36$, $\log 0.1 \approx -2.30$. Log turns products of probabilities into sums, which is why everyone works with logs.
      */}),
    },
  ],

  lessons: [
    {
      id: "see-it",
      title: "See it: a gridworld agent learns, then PPO balances a pole",
      kind: "demo",
      minutes: 75,
      runsOn: ["mac"],
      md: MD(function () {/*
## Part 1 — watch Q-learning in your browser

::viz gridworld

Things to try (2 minutes each):

1. Set **ε = 0** (never explore). The agent often locks onto the first path that “worked” — or never finds the goal at all. That is the **exploration** problem.
2. Set **γ = 0.5**. Values far from the goal fade to almost nothing — the agent becomes short-sighted. That is **discounting**.
3. Set **α** very high. Values jitter; very low, learning crawls. That is a **learning rate**, same idea as in gradient descent.

The heatmap is the agent’s estimate of “how good is it to be here” (a **value function**). The arrows are its **policy**: the best action it currently believes in for each cell. The reward curve is the thing RL is trying to push up.

## Part 2 — train PPO on CartPole with CleanRL

[CleanRL](https://github.com/vwxyzjn/cleanrl) implements each RL algorithm as **one self-contained file** (~300 lines for PPO) so you can read the whole thing. CartPole is the “hello world” of RL: push a cart left or right to keep a pole upright; +1 reward per step; the episode ends when the pole falls or after 500 steps.

```bash
git clone https://github.com/vwxyzjn/cleanrl && cd cleanrl
uv venv .venv && source .venv/bin/activate
# CleanRL pins gymnasium 0.29.x — the vector-env info format changed in 1.x
uv pip install "gymnasium==0.29.1" torch tyro tensorboard numpy
python cleanrl/ppo.py --env-id CartPole-v1 --total-timesteps 200000 --seed 1
```

On an M-series Mac this takes ~1–3 minutes on **CPU** (the network is two 64-unit layers — far too small for the GPU to help; launching MPS kernels would cost more than the math). You’ll see:

```text
global_step=476, episodic_return=[13.]
...
global_step=61440, episodic_return=[236.]
...
global_step=120832, episodic_return=[500.]
SPS: 2533
```

Then open the logs:

```bash
tensorboard --logdir runs      # open http://localhost:6006
```

Look at `charts/episodic_return` (climbs to 500), `losses/approx_kl`, `losses/clipfrac`, `losses/entropy` (falls as the policy becomes confident). You will understand every one of these curves by the end of the module.

> [!INTUITION] What just happened
> Nobody told the network which way to push. It **tried things**, got +1 per step it survived, and adjusted itself to make the actions that preceded long survival more likely. That is the entire idea of RL; the rest is making it stable and efficient.

> [!REAL] Same loop, bigger actions
> Replace “cart state” with “a math question”, “push left/right” with “choose the next token”, and “+1 per step” with “+1 if the final answer is correct”, and you have DeepSeek-R1’s training loop (M22). Replace the single process with 1,000 GPUs split into inference and training pools and you have verl/OpenRLHF (M23).

- [ ] Played with ε, α, γ in the gridworld viz and can say what each does
- [ ] CleanRL PPO reached episodic return 500 on CartPole on your Mac
- [ ] Opened TensorBoard and found the return, entropy, approx_kl and clipfrac curves
- [ ] Re-ran with `--seed 2` and `--seed 3`: noticed the curves are *not* identical (we’ll come back to this)
      */}),
      resources: [
        { title: "CleanRL", url: "https://github.com/vwxyzjn/cleanrl", type: "repo", note: "single-file RL implementations — read, don’t import" },
        { title: "CleanRL PPO docs (with benchmark curves)", url: "https://docs.cleanrl.dev/rl-algorithms/ppo/", type: "docs", note: "expected CartPole curves to compare with yours" },
      ],
    },
    {
      id: "rl-loop",
      title: "The RL loop: agent, environment, reward — and explore vs exploit",
      kind: "concept",
      minutes: 90,
      runsOn: ["mac"],
      md: MD(function () {/*
## The vocabulary (you will use every word in M22–M23)

```text
            action aₜ
   ┌──────────────────────────┐
   │                          ▼
 AGENT (policy π)        ENVIRONMENT
   ▲                          │
   └──── state sₜ₊₁, reward rₜ ┘
```

| Term | Meaning | CartPole | LLM (M22) |
|---|---|---|---|
| **State** $s_t$ | what the agent observes | 4 numbers: cart pos/vel, pole angle/vel | prompt + tokens generated so far |
| **Action** $a_t$ | what it chooses | push left / right | next token (vocab ~150k) |
| **Reward** $r_t$ | scalar feedback | +1 per step alive | 1 if final answer correct, else 0 (only at the end) |
| **Episode** | one run from reset to done | until pole falls or 500 steps | one full completion |
| **Policy** $\pi(a\mid s)$ | the agent’s decision rule | 2-way softmax MLP | the LLM itself |
| **Return** $G$ | sum of (discounted) rewards in an episode | steps survived | the reward |
| **Trajectory / rollout** | $(s_0,a_0,r_0,s_1,\dots)$ | one episode’s data | prompt + sampled completion |

## How RL differs from supervised learning

| | Supervised (M01–M05) | Reinforcement learning |
|---|---|---|
| Data | fixed dataset of (input, correct label) | **generated by the agent itself**, changes as it learns |
| Feedback | “the right answer was X” | “that was worth 0.3” — no right answer given |
| Credit | per example | **delayed**: which of 200 actions caused the fall? |
| i.i.d.? | yes, shuffle freely | no — consecutive states are correlated |
| Loop | read data → update | **act → collect → update → act again** |

That last row is why RL systems are *systems problems*: you need an inference loop (acting) and a training loop (updating) and you must keep them in sync. That is literally M23.

## The Gymnasium API

[Gymnasium](https://gymnasium.farama.org/) (the maintained successor of OpenAI Gym) standardizes the environment interface. Every RL library speaks it:

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=0)            # obs: np.array of shape (4,)
total, done = 0.0, False
while not done:
    action = env.action_space.sample()   # random policy
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    done = terminated or truncated        # fell over  OR  hit the 500-step time limit
print("random agent return:", total)     # typically 10–40
```

> [!WARNING] terminated vs truncated
> `terminated` = the task genuinely ended (pole fell): the future is worth 0. `truncated` = we stopped the clock (time limit): the future *would* have continued. Correct algorithms bootstrap from the next state’s value on truncation but not on termination. Mixing these up is a classic silent bug.

## Explore vs exploit: the simplest RL problem

Strip away states entirely: a **multi-armed bandit** is a row of slot machines with unknown payout rates. Each pull you either **exploit** (pull the arm that looks best so far) or **explore** (try another arm in case it’s better). Too little exploring → stuck on a mediocre arm forever. Too much → you waste pulls on arms you already know are bad.

::viz bandit

Compare the strategies in the viz: **greedy** often locks onto a bad arm early; **ε-greedy** explores a fixed fraction of the time; **UCB** explores arms it is *uncertain* about (optimism in the face of uncertainty). **Regret** = reward you lost versus always pulling the best arm.

```python
import numpy as np
rng = np.random.default_rng(0)
true_p = np.array([0.2, 0.5, 0.75, 0.4])           # hidden payout rates
def run(eps, steps=2000):
    n, q, total = np.zeros(4), np.zeros(4), 0.0
    for t in range(steps):
        a = rng.integers(4) if rng.random() < eps else int(np.argmax(q))
        r = float(rng.random() < true_p[a])
        n[a] += 1; q[a] += (r - q[a]) / n[a]         # running mean
        total += r
    return total / steps
for eps in [0.0, 0.01, 0.1, 0.3]:
    print(eps, round(np.mean([run(eps) for _ in range(20)]), 3))
```

> [!REAL] Exploration in LLM RL
> In LLMs, exploration is controlled by **sampling temperature** and by sampling **many completions per prompt** (GRPO samples 8–64). If all 8 answers to a question are wrong, there is no learning signal for that question — exactly the bandit problem of never pulling the good arm. “Entropy collapse” (the model becoming too confident too early) is a top failure mode in reasoning-RL papers.

- [ ] Ran the random CartPole agent; noted its average return over 20 episodes
- [ ] Ran the bandit script; found which ε works best and explained why 0.0 is worse
- [ ] In the bandit viz, found a setting where greedy beats ε-greedy on a short horizon and explained it
- [ ] Wrote down the LLM equivalent of state, action, reward, episode in your own words
      */}),
      resources: [
        { title: "Gymnasium docs", url: "https://gymnasium.farama.org/", type: "docs", note: "the env API every RL library uses" },
        { title: "Sutton & Barto ch. 2 (bandits)", url: "http://incompleteideas.net/book/the-book-2nd.html", type: "book", note: "the canonical treatment of explore/exploit" },
      ],
    },
    {
      id: "discounted-return",
      title: "Math: expected value and the discounted return",
      kind: "math",
      minutes: 60,
      runsOn: ["browser"],
      md: MD(function () {/*
RL maximizes an **expected discounted return**. Two ideas, both gentler than they sound.

> [!PREREQ] Expectation = weighted average
> If a die pays \$1 on 1–4 and \$10 on 5–6, the **expected value** is each outcome times its probability, summed:
> $\mathbb{E}[X] = \tfrac{4}{6}\cdot 1 + \tfrac{2}{6}\cdot 10 = 4$.
> In code: `sum(p * x for p, x in zip(probs, values))`. When you can’t enumerate outcomes you **sample** and average — `np.mean(samples)` estimates $\mathbb{E}[X]$. RL does this constantly: every rollout is one sample.

## The return

The agent collects rewards $r_0, r_1, r_2, \dots$. We want one number per trajectory. The **discounted return** is

$$
G_0 = r_0 + \gamma r_1 + \gamma^2 r_2 + \dots = \sum_{t=0}^{\infty} \gamma^t r_t, \qquad 0 \le \gamma < 1
$$

**Why discount?** (1) A reward now is more certain than one in 100 steps. (2) It keeps infinite sums finite. (3) It shrinks the credit given to actions far in the past, which lowers variance.

**Worked example** ($\gamma = 0.9$, rewards `[0, 0, 0, 10]` — like the gridworld goal at step 4):

$$G_0 = 0 + 0.9\cdot 0 + 0.81 \cdot 0 + 0.729 \cdot 10 = 7.29$$

Same reward reached one step later is worth $0.9 \times 7.29 = 6.56$. **Discounting makes shorter paths better**, which is why the gridworld agent finds the shortest safe route.

> [!PREREQ] Geometric series
> $1 + \gamma + \gamma^2 + \dots = \dfrac{1}{1-\gamma}$ for $\gamma<1$. Check: with $\gamma=0.5$, $1 + 0.5 + 0.25 + 0.125 + \dots \to 2 = \tfrac{1}{0.5}$.

So a stream of +1 rewards forever is worth $\tfrac{1}{1-\gamma}$. That number is the **effective horizon**: roughly how many steps ahead the agent “cares about”.

| $\gamma$ | effective horizon $\frac{1}{1-\gamma}$ | weight of reward 100 steps away ($\gamma^{100}$) |
|---|---|---|
| 0.9 | 10 | 0.00003 |
| 0.99 | 100 | 0.37 |
| 0.999 | 1000 | 0.90 |

::viz discount

A useful recursion you’ll use in the next lesson and in every implementation:

$$G_t = r_t + \gamma\, G_{t+1}$$

```python
def discounted_returns(rewards, gamma=0.99):
    G, out = 0.0, []
    for r in reversed(rewards):      # walk backwards: G_t = r_t + γ·G_{t+1}
        G = r + gamma * G
        out.append(G)
    return out[::-1]
print(discounted_returns([0, 0, 0, 10], 0.9))   # [7.29, 8.1, 9.0, 10.0]
```

## The objective

The policy $\pi_\theta$ (a network with weights $\theta$) induces a distribution over trajectories. RL maximizes

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ G(\tau) \right]$$

In words: **the average return you’d get if you ran this policy many times.** We estimate it by running episodes and averaging — which is why RL is noisy and why you need several seeds.

> [!REAL] In LLM RL, $\gamma = 1$
> A completion is short and finite and the reward usually only arrives at the end, so GRPO/RLOO simply use $\gamma = 1$: every token of the answer gets credit for the final reward. CleanRL’s PPO uses $\gamma = 0.99$ because CartPole episodes are long.

- [ ] Computed $G_0$ by hand for rewards `[1, 1, 1, 1]` with $\gamma=0.5$ and checked with the function (answer: 1.875)
- [ ] In the discount viz, found the $\gamma$ where a reward 50 steps away still has weight > 0.5
- [ ] Explained why $\gamma=0.5$ in the gridworld viz makes far cells look worthless
      */}),
    },
    {
      id: "values-bellman",
      title: "Value functions and the Bellman equation (intuition first)",
      kind: "concept",
      minutes: 75,
      runsOn: ["browser", "mac"],
      md: MD(function () {/*
## Two questions an agent wants answered

- **State value** $V^\pi(s)$: “If I’m in state $s$ and follow my policy from here, what return do I expect?” — the **heatmap** in the gridworld viz.
- **Action value** $Q^\pi(s,a)$: “If I take action $a$ now and follow my policy after, what return do I expect?” — pick $\arg\max_a Q(s,a)$ and you have the **arrows**.

They’re just expectations of $G_t$, conditioned on where you start.

## The Bellman equation = “value now = reward now + discounted value next”

From $G_t = r_t + \gamma G_{t+1}$ and taking expectations:

$$
V^\pi(s) = \mathbb{E}\big[\, r + \gamma\, V^\pi(s') \,\big]
\qquad
Q^*(s,a) = \mathbb{E}\big[\, r + \gamma \max_{a'} Q^*(s',a') \,\big]
$$

The second is the **Bellman optimality equation**: the best value of an action is its immediate reward plus the discounted value of acting *optimally* afterwards. It lets you compute long-term value **from one-step lookahead** — no need to simulate whole futures.

> [!INTUITION] Like shortest paths
> If you’ve implemented Dijkstra or Bellman–Ford (same Bellman!), this is familiar: the distance from A to the goal = cost of the first edge + distance from the neighbor. Values flow backwards from the goal one step per update — exactly what you see spreading across the gridworld heatmap.

## Worked numeric example

A corridor of 4 cells: `A — B — C — Goal`. Actions: left/right. Each step costs −1; entering Goal gives +10 and ends the episode. $\gamma = 0.9$. Start with all $V = 0$ and repeatedly apply $V(s) \leftarrow \max_a [\, r + \gamma V(s') \,]$ (**value iteration**):

| sweep | V(A) | V(B) | V(C) | reasoning for C |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | initial guess |
| 1 | −1 | −1 | **10** | right → Goal: $10 + 0.9\cdot 0$ |
| 2 | −1.9 | **8.0** | 10 | B: $-1 + 0.9\cdot 10 = 8$ |
| 3 | **6.2** | 8.0 | 10 | A: $-1 + 0.9 \cdot 8 = 6.2$ |
| 4 | 6.2 | 8.0 | 10 | converged |

Information travels **one cell per sweep** from the goal. Greedy policy from these values: always go right. 

```python
import numpy as np
gamma, V = 0.9, np.zeros(4)            # A, B, C, Goal(terminal, stays 0)
for sweep in range(5):
    new = V.copy()
    for s in range(3):
        right = (10 if s + 1 == 3 else -1) + gamma * (0 if s + 1 == 3 else V[s + 1])
        left  = -1 + gamma * V[max(s - 1, 0)]
        new[s] = max(left, right)
    V = new; print(sweep + 1, V[:3].round(2))
```

## Why we don’t stop here

Value iteration needs a **model**: you must know $r$ and $s'$ for every $(s,a)$. Real environments (and LLMs answering math) don’t hand you a transition table. **Q-learning** (next lesson) replaces “look up the model” with “try it and see” — a *sampled* Bellman update:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\,\big[\underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{TD target}} - Q(s,a)\big]$$

The bracket is the **TD error** (temporal difference): how surprised you were. Same shape as a gradient step: move a little ($\alpha$) toward the target.

> [!REAL] Where value functions survive in LLM RL
> PPO for LLMs trains a **value model** (critic) — a whole second LLM that predicts expected reward from each token prefix (M22). GRPO’s headline trick is *deleting* it and using the average reward of a group of samples instead. Knowing what $V(s)$ is makes that trade-off obvious.

- [ ] Ran the value-iteration snippet and matched the table
- [ ] Changed the step cost to 0: what happens to V(A)? Why does the policy still go right? (hint: $\gamma$)
- [ ] In the gridworld viz, watched values propagate outward from the goal and counted how many episodes until the start cell “lights up”
      */}),
    },
    {
      id: "q-learning-dqn",
      title: "Build: tabular Q-learning from scratch — then DQN’s two tricks",
      kind: "build",
      minutes: 120,
      runsOn: ["mac"],
      md: MD(function () {/*
## Build a gridworld and a Q-learner in pure NumPy

No Gymnasium, no PyTorch. Every line is yours.

```python
# qlearn.py
import numpy as np

GRID = ["S...",
        ".H.H",
        "...H",
        "H..G"]          # S start, H hole (-1, episode ends), G goal (+1, ends), . = -0.01 step cost
N = len(GRID)
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # up, right, down, left
ARROWS = "^>v<"

def step(s, a):
    r, c = divmod(s, N)
    dr, dc = ACTIONS[a]
    r2 = min(max(r + dr, 0), N - 1)            # walls: stay inside the grid
    c2 = min(max(c + dc, 0), N - 1)
    s2 = r2 * N + c2
    cell = GRID[r2][c2]
    if cell == "G": return s2, 1.0, True
    if cell == "H": return s2, -1.0, True
    return s2, -0.01, False

def train(episodes=2000, alpha=0.5, gamma=0.95, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N * N, len(ACTIONS)))        # the whole "model": a 16x4 table
    returns = []
    for ep in range(episodes):
        s, done, G, t = 0, False, 0.0, 0
        while not done and t < 100:
            if rng.random() < eps:                 # explore
                a = int(rng.integers(len(ACTIONS)))
            else:                                  # exploit
                a = int(np.argmax(Q[s]))
            s2, r, done = step(s, a)
            target = r if done else r + gamma * Q[s2].max()   # TD target (no bootstrap past a terminal)
            Q[s, a] += alpha * (target - Q[s, a])             # TD update
            s, G, t = s2, G + r, t + 1
        returns.append(G)
    return Q, returns

if __name__ == "__main__":
    Q, returns = train()
    for k in [0, 100, 500, 1999]:
        lo = max(0, k - 99)
        print(f"episode {k:5d}  avg return (last 100) = {np.mean(returns[lo:k+1]):+.3f}")
    for r in range(N):
        print(" ".join(GRID[r][c] if GRID[r][c] in "HG" else ARROWS[int(np.argmax(Q[r*N + c]))]
                       for c in range(N)))
    print("V(start) =", Q[0].max().round(3))
```

Output (seed 0):

```text
episode     0  avg return (last 100) = -1.080
episode   100  avg return (last 100) = +0.522
episode   500  avg return (last 100) = +0.788
episode  1999  avg return (last 100) = +0.728
v > v <
v H v H
> > v H
H > > G
V(start) = 0.729
```

**Check against theory:** the shortest safe path is 6 steps: five −0.01 steps then +1. Discounted: $-0.01(1+0.95+0.95^2+0.95^3+0.95^4) + 0.95^5 \approx -0.045 + 0.774 = 0.729$. The table learned the exact optimal value. Why is the *average training return* lower than 0.95 (the undiscounted optimum)? Because with ε = 0.1 the agent still takes random steps — sometimes into holes. Evaluate greedily (ε = 0) to see the true policy performance.

> [!NOTE] Off-policy, already
> Q-learning’s target uses $\max_{a'} Q(s', a')$ — the *greedy* action — even though the agent actually behaved ε-greedily. It learns about one policy (greedy) from data generated by another (ε-greedy). That’s **off-policy** learning, and it’s why Q-learning can reuse old data. Remember this word for M23.

## When the table doesn’t fit: DQN

CartPole’s state is 4 continuous numbers; Atari’s is 84×84×4 pixels. No table. **DQN** (Mnih et al. 2013/2015) replaces the table with a neural net $Q_\theta(s,\cdot)$ and minimizes $(\text{TD target} - Q_\theta(s,a))^2$. Naively this diverges. Two tricks made it work:

| Trick | Problem it fixes | How |
|---|---|---|
| **Replay buffer** | consecutive transitions are highly correlated; SGD assumes i.i.d. | store the last ~1M $(s,a,r,s',\text{done})$ tuples; train on random minibatches from it |
| **Target network** | the target $r+\gamma\max Q_\theta(s',\cdot)$ moves every step you update $\theta$ — chasing your own tail | compute targets with a frozen copy $Q_{\bar\theta}$; copy $\theta\to\bar\theta$ every ~1k–10k steps |

```python
# the heart of DQN (PyTorch), given a batch sampled from the replay buffer
with torch.no_grad():
    target = r + gamma * (1 - done) * target_net(s2).max(dim=1).values
q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
loss = torch.nn.functional.smooth_l1_loss(q_sa, target)
```

CleanRL’s `dqn.py` is ~200 lines if you want to run it on CartPole. You won’t need DQN for LLMs (action space = vocab, and rewards come at the end), but **replay buffers and “stale copies of weights” reappear** in distributed RL (Ape-X in M23; frozen reference models in M22).

- [ ] `qlearn.py` runs and learns the optimal path; V(start) ≈ 0.729
- [ ] Added a greedy evaluation function (ε = 0) and reported the success rate over 100 episodes
- [ ] Plotted a 100-episode moving average of returns for ε ∈ {0.0, 0.1, 0.3} on one chart
- [ ] Made the grid **slippery** (with prob 0.2 the move goes in a random direction): does the policy route further from holes? Why?
- [ ] Explained in two sentences why a replay buffer is fine for Q-learning but would be “wrong” for vanilla policy gradients
      */}),
      resources: [
        { title: "Playing Atari with Deep RL (DQN)", url: "https://arxiv.org/abs/1312.5602", type: "paper", note: "replay buffer + deep Q-network; short and readable" },
      ],
    },
    {
      id: "policy-gradients",
      title: "Math: policy gradients — REINFORCE, baselines, actor–critic, GAE",
      kind: "math",
      minutes: 120,
      runsOn: ["mac"],
      md: MD(function () {/*
Q-learning learns *values* and derives a policy. **Policy-gradient** methods learn the policy $\pi_\theta(a\mid s)$ **directly** — a softmax over actions, exactly like an LLM’s next-token distribution. This is the family PPO, GRPO and every LLM-RL method belongs to.

> [!PREREQ] Gradient of a log
> $\frac{d}{dx}\log f(x) = \frac{f'(x)}{f(x)}$. Rearranged: $f'(x) = f(x)\cdot \frac{d}{dx}\log f(x)$. That one line is the “log-derivative trick”. If “gradient” is fuzzy, revisit the `derivative` viz in M01: it’s just “which way and how much does the output change if I nudge the input”.

## The problem

We want $\nabla_\theta J(\theta)$ where $J = \mathbb{E}_{\tau\sim\pi_\theta}[G(\tau)]$. But the reward comes from the **environment** — a black box you can’t differentiate through (a physics sim, a unit-test runner, a math checker). How do you get a gradient?

## The log-derivative trick, gently

Write the expectation as a weighted sum over outcomes: $J = \sum_\tau p_\theta(\tau)\, G(\tau)$. Only the probabilities depend on $\theta$:

$$
\nabla J = \sum_\tau \nabla p_\theta(\tau)\, G(\tau)
= \sum_\tau p_\theta(\tau)\, \nabla \log p_\theta(\tau)\, G(\tau)
= \mathbb{E}_{\tau}\big[ \nabla \log p_\theta(\tau)\, G(\tau) \big]
$$

The middle step is the trick ($\nabla p = p\,\nabla\log p$). Now it’s an expectation again, so we can **estimate it by sampling**. And $\log p_\theta(\tau)$ is a sum of $\log \pi_\theta(a_t\mid s_t)$ over steps (environment dynamics don’t depend on $\theta$, so they drop out of the gradient). Result — **REINFORCE** (Williams, 1992):

$$
\nabla_\theta J \approx \frac{1}{N}\sum_{\text{episodes}} \sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)\; G_t
$$

> [!INTUITION] Read it as a weighted “supervised” update
> $\nabla \log \pi(a_t\mid s_t)$ is exactly the gradient you’d use to make action $a_t$ **more likely** — as if it were the label in a classification problem (this is cross-entropy from M02!). REINFORCE multiplies it by $G_t$: actions followed by high returns get pushed up a lot, low returns a little (or down, if negative). **RL = supervised learning on your own samples, weighted by how well they turned out.**

**Numeric example.** 3 actions, probabilities $[0.5, 0.3, 0.2]$. You sample action 1 and get $G=+2$. For a softmax, $\nabla_{\text{logits}} \log \pi(a{=}1) = \text{onehot}(1) - \pi = [-0.5, 0.7, -0.2]$. Times $G=2$: $[-1.0, 1.4, -0.4]$. A step along that raises logit 1 and lowers the others → action 1 becomes more likely. If $G$ had been $-2$ it would push the other way.

::viz policy-gradient

## Baselines: same expectation, much less noise

Problem: in CartPole all returns are positive (you always survive a few steps), so REINFORCE pushes **every** sampled action up — just some more than others. The signal is buried in noise. Fix: subtract a **baseline** $b(s)$:

$$\nabla J \approx \sum_t \nabla \log\pi(a_t\mid s_t)\,\big(G_t - b(s_t)\big)$$

This doesn’t change the expected gradient (because $\mathbb{E}_a[\nabla\log\pi(a\mid s)] = \nabla \sum_a \pi(a\mid s) = \nabla 1 = 0$), but it **reduces variance** dramatically. Toggle the baseline in the viz above and watch the gradient estimates tighten.

The best simple baseline is the state value $V(s)$. Then $G_t - V(s_t)$ is the **advantage** $A_t$: “how much better than usual was what I did?” Positive → reinforce; negative → suppress.

## Actor–critic

Learn $V_\phi(s)$ with a second network (the **critic**) by regressing onto observed returns, and use it as the baseline for the policy (the **actor**). CleanRL’s `Agent` class has exactly these two heads: `self.actor` and `self.critic`.

Going further, instead of waiting for the full return you can bootstrap: the one-step TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is itself an (noisier-but-lower-variance) advantage estimate.

## GAE in one paragraph

**Generalized Advantage Estimation** (Schulman et al. 2015) blends these: $\hat A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \dots$. $\lambda = 0$ → pure one-step TD (low variance, biased by a bad critic); $\lambda = 1$ → full Monte-Carlo return minus baseline (unbiased, high variance). $\lambda=0.95$ is the default everyone uses. In CleanRL it’s the backwards loop `advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam`.

## Build: REINFORCE on CartPole (≈40 lines)

```python
# reinforce.py
import gymnasium as gym, numpy as np, torch, torch.nn as nn

def run(seed=0, episodes=600, lr=1e-2, gamma=0.99):
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    policy = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        logps, rewards, done = [], [], False
        while not done:
            dist = torch.distributions.Categorical(logits=policy(torch.as_tensor(obs, dtype=torch.float32)))
            a = dist.sample()
            logps.append(dist.log_prob(a))
            obs, r, terminated, truncated, _ = env.step(a.item())
            rewards.append(r); done = terminated or truncated
        G, rtg = 0.0, []
        for r in reversed(rewards):              # returns-to-go G_t
            G = r + gamma * G; rtg.append(G)
        rtg = torch.tensor(rtg[::-1])
        adv = (rtg - rtg.mean()) / (rtg.std() + 1e-8)   # crude baseline: centre & scale
        loss = -(torch.stack(logps) * adv).sum()       # minus: optimizers minimize
        opt.zero_grad(); loss.backward(); opt.step()
        history.append(sum(rewards))
        if ep % 50 == 0: print(ep, np.mean(history[-50:]))
    return history

if __name__ == "__main__":
    run()
```

Notice: **the loss has no meaning as a number** — it’s a device to make autograd produce the policy gradient. Don’t watch it go down; watch the return go up.

- [ ] Did the 3-action numeric example by hand, then checked it with `torch.autograd` on `Categorical(logits=...).log_prob`
- [ ] REINFORCE reaches an average return > 400 on CartPole within 600 episodes for at least one seed
- [ ] Replaced the centring with **no baseline** (`adv = rtg`) and compared learning curves — describe the difference
- [ ] Found the GAE loop in CleanRL `ppo.py` and annotated each variable
      */}),
      resources: [
        { title: "Spinning Up — Intro to Policy Optimization", url: "https://spinningup.openai.com", type: "course", note: "the clearest derivation of the policy gradient; read Part 3" },
        { title: "Lilian Weng — Policy Gradient Algorithms", url: "https://lilianweng.github.io/posts/2018-04-08-policy-gradient/", type: "article", note: "one page covering REINFORCE → actor-critic → PPO" },
        { title: "GAE paper", url: "https://arxiv.org/abs/1506.02438", type: "paper", note: "λ-weighted advantages; skim sections 1–3" },
      ],
    },
    {
      id: "ppo",
      title: "PPO: the clipped objective — and reading CleanRL ppo.py line by line",
      kind: "concept",
      minutes: 150,
      runsOn: ["mac"],
      md: MD(function () {/*
## The problem PPO solves

Vanilla policy gradient does **one** gradient step per batch of rollouts, then throws the data away (it was sampled from the *old* policy; after the update it’s stale). Rollouts are expensive — in LLM RL they are the dominant cost — so you want to take **several** gradient steps per batch. But if you push too far, the policy changes so much that the data no longer describes it, and training collapses. Collapse is catastrophic in RL: a bad policy collects bad data, which makes the next update worse.

## The ratio

Let $\pi_{\text{old}}$ be the policy that collected the data. For each sampled action define the **probability ratio**

$$r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)} = \exp\big(\log\pi_\theta - \log\pi_{\text{old}}\big)$$

$r = 1$: unchanged. $r = 1.5$: the new policy is 50% more likely to take that action. Maximizing $r_t \hat A_t$ is (to first order) the same as the policy gradient — and it’s the **importance-sampling** correction for using old data (full story in M23).

## The clip

PPO (Schulman et al. 2017) maximizes

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t \hat A_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\Big], \qquad \epsilon = 0.2
$$

::viz ppo-clip

Read the viz for both signs of the advantage:

- **$\hat A > 0$ (good action):** the objective rises with $r$ until $r = 1.2$, then goes flat. Flat = zero gradient. “You’ve made this action 20% more likely; that’s enough for this batch.”
- **$\hat A < 0$ (bad action):** the objective improves as $r$ falls until $r = 0.8$, then flat. “You’ve made it 20% less likely; stop.”
- The `min` makes it **pessimistic**: if an update has already made things *worse* (e.g. increased a bad action), the unclipped term is used and the gradient still pushes back.

> [!INTUITION] Why it’s stable
> The clip is a cheap **trust region**: each batch can move each action’s probability by only ~±20% before its gradient switches off. You can then safely run 4–10 epochs of minibatch SGD over the same rollouts. TRPO did this with an expensive second-order constraint; PPO gets 90% of the benefit with a `torch.clamp`.

## Read CleanRL `ppo.py` with this map

Open `cleanrl/ppo.py` next to this table. Every line maps to something you now know:

| Lines (approx.) | Code | Concept |
|---|---|---|
| `Args` dataclass | `num_envs=4, num_steps=128` | batch = 4 × 128 = 512 transitions per update |
| `make_env` + `SyncVectorEnv` | 4 CartPoles stepped in lockstep | **vectorized envs** (more data per forward pass) |
| `Agent` | `actor` → logits, `critic` → V(s) | **actor–critic**; `std=0.01` init on the actor head → near-uniform starting policy (exploration) |
| rollout loop | `agent.get_action_and_value(next_obs)` under `no_grad` | acting = **inference**; store `logprobs` = $\log\pi_{\text{old}}$ |
| `envs.step` | `terminations, truncations` | the Gymnasium API; `next_done` masks bootstrapping |
| reversed loop | `delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]` | TD error $\delta_t$ → **GAE** advantages |
| `returns = advantages + values` | | critic regression targets |
| `for epoch in range(update_epochs)` | 4 epochs × 4 minibatches | **reusing data** — the reason for the clip |
| `logratio = newlogprob - b_logprobs[mb_inds]` | `ratio = logratio.exp()` | $r_t(\theta)$ |
| `approx_kl = ((ratio - 1) - logratio).mean()` | | KL estimate between old/new policy — a health metric |
| `mb_advantages = (… - mean) / (std + 1e-8)` | | **advantage normalization** (per minibatch) |
| `pg_loss = torch.max(pg_loss1, pg_loss2).mean()` | negated objective, so min ↔ max swap | $-L^{\text{CLIP}}$ |
| `v_loss` (optionally clipped) | | critic MSE |
| `loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef` | | total loss; **entropy bonus** keeps exploring |
| `clip_grad_norm_(…, 0.5)` | | gradient clipping — standard stability trick |
| `clipfrac`, `explained_variance` | | diagnostics: fraction of clipped samples; how well the critic predicts returns |

> [!TIP] The 37 implementation details
> Huang et al.’s ICLR blog post “The 37 Implementation Details of PPO” explains every “magic” line (orthogonal init, LR annealing, advantage normalization, value clipping…) and which ones actually matter. It’s the best single document on making RL work in practice.

## Healthy PPO dashboards

| Metric | Healthy (CartPole) | Red flag |
|---|---|---|
| `approx_kl` | 0.001–0.02 | > 0.05: steps too large (lower LR / fewer epochs) |
| `clipfrac` | 0.05–0.2 | ~0: clip never active (LR too small); > 0.3: too aggressive |
| `entropy` | slowly decreasing | crashes to ~0 early → premature convergence |
| `explained_variance` | rising toward 1 | negative: critic is worse than predicting the mean |

> [!REAL] You will see these exact names again
> TRL’s `GRPOTrainer` logs `clip_ratio/region_mean`, `kl`, `entropy`; verl and OpenRLHF log `ppo_kl`, `pg_clipfrac`, `entropy`. Learning to read them on CartPole (seconds per run) is far cheaper than on a 7B model (hours per run).

- [ ] Read all of `ppo.py` and wrote a one-line comment above every block in your own copy
- [ ] In the ppo-clip viz, explained why the gradient is zero for $\hat A>0,\ r>1.2$ but **not** zero for $\hat A>0,\ r<0.8$
- [ ] Ran PPO with `--clip-coef 0.05` and `--clip-coef 1.0` (≈ no clipping) and `--update-epochs 20`; compared returns and `approx_kl`
- [ ] Ran with `--ent-coef 0.0` and `--ent-coef 0.1`: what happens to entropy and return?
      */}),
      resources: [
        { title: "PPO paper", url: "https://arxiv.org/abs/1707.06347", type: "paper", note: "short; section 3 is the clipped objective" },
        { title: "The 37 Implementation Details of PPO", url: "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/", type: "article", note: "every trick in CleanRL’s ppo.py, explained and ablated" },
      ],
    },
    {
      id: "rl-engineering",
      title: "Lab: practical RL engineering — seeds, variance, vectorized envs, reward hacking",
      kind: "lab",
      minutes: 120,
      runsOn: ["mac"],
      md: MD(function () {/*
RL is famous for being fragile. Most of that fragility is **engineering**, and it is exactly the engineering you’ll do at scale in M22–M23.

## 1. Seeds and variance: never trust one run

Run PPO 5 times with different seeds:

```bash
for s in 1 2 3 4 5; do
  python cleanrl/ppo.py --env-id CartPole-v1 --total-timesteps 150000 --seed $s --exp-name ppo_s$s &
done; wait
```

Export `charts/episodic_return` from TensorBoard (or parse the logs) and plot **mean ± std across seeds**:

```python
# plot_seeds.py — reads the TensorBoard event files CleanRL writes to runs/
import glob, numpy as np, matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

grid = np.linspace(0, 150_000, 100)
curves = []
for d in glob.glob("runs/CartPole-v1__ppo_s*"):
    ea = EventAccumulator(d); ea.Reload()
    ev = ea.Scalars("charts/episodic_return")
    x = np.array([e.step for e in ev]); y = np.array([e.value for e in ev])
    y_s = np.convolve(y, np.ones(20) / 20, mode="same")          # smooth
    curves.append(np.interp(grid, x, y_s))
c = np.array(curves)
plt.plot(grid, c.mean(0)); plt.fill_between(grid, c.mean(0) - c.std(0), c.mean(0) + c.std(0), alpha=0.3)
plt.xlabel("env steps"); plt.ylabel("episodic return"); plt.title(f"PPO CartPole, {len(c)} seeds")
plt.savefig("ppo_seeds.png", dpi=150)
```

You’ll see some seeds hit 500 at 60k steps and others at 130k. **A single-seed comparison between two algorithms is meaningless** — the RL literature was burned by this (Henderson et al., “Deep RL that Matters”, 2018). At LLM scale you can’t afford 5 seeds of a 1,000-GPU run, which is why labs rely on small-scale ablations with many seeds first.

## 2. Vectorized environments: batching for RL

`num_envs=4` in CleanRL steps 4 environments at once and runs **one** forward pass for all 4 observations. That is batching — the same idea as continuous batching in an inference server (M09): amortize a fixed per-call cost over more work.

| `--num-envs` | env steps / update | SPS (Mac, rough) | sample efficiency |
|---|---|---|---|
| 1 | 128 | lower | often better per step |
| 4 | 512 | baseline | default |
| 16 | 2048 | higher | fewer updates per env step |

Measure it: run `--num-envs 1`, `4`, `16` with the same `--total-timesteps` and record SPS (printed) and steps-to-500. **Throughput vs. sample efficiency** is the central trade-off of every distributed RL system you’ll meet in M23. (`gym.vector.AsyncVectorEnv` runs envs in subprocesses — worth it only when the env itself is slow, e.g. a code sandbox.)

## 3. Logging: what to always record

- Per episode: return, length. Per update: policy loss, value loss, entropy, approx_kl, clipfrac, explained variance, LR, SPS, wall time.
- The **config and git hash** of every run (CleanRL writes hyperparameters into TensorBoard; `--track` sends everything to Weights & Biases).
- A few **videos or transcripts** of the agent (`--capture-video`). For LLMs: log sampled completions every N steps — numbers alone hide reward hacking.

## 4. Reward hacking: the agent optimizes what you wrote, not what you meant

Modify CartPole’s reward with a wrapper and watch it get exploited:

```python
import gymnasium as gym
class CenterBonus(gym.Wrapper):
    """Intended: 'stay near the center'. Written: reward = 1 + 2*(1 - |x|/2.4)."""
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return obs, r + 2 * (1 - abs(obs[0]) / 2.4), term, trunc, info
```

This one is benign — now try a buggy one, e.g. reward based on pole *angular velocity* being small, or a bonus paid once per episode that the agent can re-trigger. Famous real cases: the CoastRunners boat circling forever to collect respawning targets instead of finishing the race; LLMs learning that longer answers get higher reward-model scores; code models special-casing unit tests. In M22 you will catch your own GRPO model doing this.

> [!IMPORTANT] Rules of thumb
> 1. Always log the **true task metric** separately from the (shaped) reward. 2. Look at samples, not just curves. 3. Normalize observations/rewards when scales differ wildly. 4. Change one thing at a time, with ≥3 seeds.

- [ ] Produced `ppo_seeds.png` (mean ± std over 5 seeds)
- [ ] Measured SPS and steps-to-500 for `--num-envs` 1, 4, 16; wrote down the trade-off
- [ ] Built a reward wrapper the agent exploits, and showed the exploit in a video or trajectory printout
- [ ] Wrote a 5-line “RL experiment checklist” you’ll reuse in M22
      */}),
      resources: [
        { title: "CleanRL", url: "https://github.com/vwxyzjn/cleanrl", type: "repo", note: "`--track`, `--capture-video`, seeds — good experiment hygiene built in" },
      ],
    },
    {
      id: "deep-reading",
      title: "Deep dive: Spinning Up, Sutton & Barto, David Silver",
      kind: "deep",
      minutes: 240,
      optional: true,
      runsOn: ["any"],
      md: MD(function () {/*
You now have working intuition. These three resources turn it into solid theory. You do **not** need all of them before M22; pick based on how you learn.

| Resource | Best for | What to read/watch first | Time |
|---|---|---|---|
| **Spinning Up in Deep RL** (OpenAI) | engineers; the most direct path to PPO | Part 1 (Key Concepts), Part 2 (Kinds of RL algorithms), Part 3 (Intro to Policy Optimization), then the PPO and VPG algorithm pages | 6–8 h |
| **Sutton & Barto**, *Reinforcement Learning: An Introduction*, 2nd ed. (free) | the classic textbook; precise definitions | ch. 2 (bandits), ch. 3 (MDPs, Bellman), ch. 6 (TD / Q-learning), ch. 13 (policy gradient) | 15+ h |
| **David Silver’s UCL RL course** | lectures; great for building intuition by ear | lectures 1–2 (MDPs), 4–5 (model-free prediction/control), 7 (policy gradient) | 10 h |

## Suggested path

1. **Spinning Up, Part 3** — rederive the policy gradient on paper following their proof. Compare with the log-derivative lesson here.
2. **Sutton & Barto ch. 6** — TD learning, SARSA vs Q-learning (on- vs off-policy — you’ll need it for M23). Try the “cliff walking” example (6.5) in your `qlearn.py`: it shows SARSA preferring a safer path.
3. **Silver lecture 7** — policy gradient and actor–critic, with the compatible-function-approximation view.
4. Optional: implement **SARSA** by changing one line of `qlearn.py` (use the action you’ll actually take next, not the max). Explain the behavioural difference on a slippery grid.

<details><summary>What we skipped (and when it matters)</summary>

- **Model-based RL** (learn the environment’s dynamics, plan inside it — Dyna, MuZero, Dreamer). Matters for games/robotics; mostly irrelevant for LLM post-training today.
- **Continuous actions** (Gaussian policies, DDPG/TD3/SAC). Matters for robotics; LLM actions are discrete tokens.
- **Offline RL** (learn only from a fixed dataset). Conceptually close to DPO (M22), which learns from fixed preference data.
- **Exploration bonuses** (curiosity, count-based). Research frontier for reasoning models (“how do we make LLMs explore new solution strategies?”).

</details>

- [ ] Finished Spinning Up Part 3 and rederived the policy gradient on paper
- [ ] Implemented SARSA and compared it with Q-learning on a slippery grid or cliff-walk
- [ ] Watched at least two Silver lectures and wrote a one-paragraph summary of each
      */}),
      resources: [
        { title: "Spinning Up in Deep RL", url: "https://spinningup.openai.com", type: "course", note: "best engineer-oriented intro to policy optimization" },
        { title: "Sutton & Barto (2nd ed., free)", url: "http://incompleteideas.net/book/the-book-2nd.html", type: "book", note: "the reference textbook" },
        { title: "David Silver — RL course", url: "https://www.davidsilver.uk/teaching/", type: "video", note: "classic UCL lecture series + slides" },
      ],
    },
  ],

  challenge: {
    title: "REINFORCE vs PPO-lite, from scratch, over 5 seeds",
    md: MD(function () {/*
Write **your own** single file `pg_cartpole.py` (< 300 lines, no CleanRL imports — you may consult it) that implements two algorithms on `CartPole-v1`:

1. **REINFORCE** with a learned value baseline (actor + critic networks; advantage $= G_t - V(s_t)$).
2. **PPO-lite**: vectorized envs (`gym.vector.SyncVectorEnv`, 4–8 envs), GAE ($\lambda = 0.95$), clipped surrogate ($\epsilon = 0.2$), K epochs of minibatch updates, value loss, entropy bonus, advantage normalization. Skip the extras (LR annealing, value clipping) unless you want them.

Run each for the **same number of environment steps** (e.g. 200k) over **5 seeds** and produce:

- One plot: return vs env steps, mean ± std band, both algorithms.
- A table: steps-to-first-reach-475 (median over seeds), final return, wall-clock time, SPS.
- A paragraph explaining the difference using the words *sample reuse*, *variance*, *trust region*.

Hints: log `approx_kl` and `clipfrac` for PPO-lite to debug. If PPO-lite is *worse* than REINFORCE, check (a) you store `logprob` from the rollout policy, not recompute it after updates, (b) `terminated` vs `truncated` handling, (c) advantage normalization is per minibatch.
    */}),
    checklist: [
      "`pg_cartpole.py` < 300 lines, runs both algorithms with a `--algo` flag and a `--seed` flag",
      "PPO-lite reaches return ≥ 475 on at least 4 of 5 seeds within 200k steps",
      "Plot with mean ± std over 5 seeds for both algorithms on the same axes",
      "Table with steps-to-475, final return, wall-clock and SPS",
      "PPO-lite logs approx_kl and clipfrac, and you can say what healthy values look like",
      "Short written explanation of why PPO is more sample-efficient than REINFORCE here",
    ],
    stretch: "Port PPO-lite to a 1-D ‘token’ task: the policy emits a sequence of 5 tokens from a vocab of 10 and gets reward 1 only if the sequence is sorted. Reward only at the end, γ = 1 — this is the LLM RL setting in miniature and a perfect warm-up for M22.",
  },

  connects: MD(function () {/*
Everything in M22 is this module with a bigger policy: the **LLM is the actor**, tokens are actions, the reward arrives once at the end, and PPO’s ratio/clip/KL/entropy all carry over unchanged. GRPO is essentially “PPO without the critic, using a group of samples as the baseline” — the baseline idea from the policy-gradient lesson. M23 then takes the **act → collect → update** loop you ran in one process and splits it across machines: the acting side becomes an **inference engine** (vLLM/SGLang — your M06–M10 skills) and the update side a training cluster, with weight synchronization between them.
  */}),

  interview: [
    "Explain the difference between on-policy and off-policy RL. Is Q-learning on- or off-policy? PPO?",
    "Derive the REINFORCE gradient using the log-derivative trick. Why can we ignore the environment dynamics?",
    "Why does subtracting a baseline not bias the policy gradient but reduce its variance?",
    "What problem does PPO’s clipping solve? Sketch the clipped objective for positive and negative advantages.",
    "What are the replay buffer and target network in DQN for?",
    "What does GAE’s λ trade off? What happens at λ = 0 and λ = 1?",
    "Your PPO run’s approx_kl is 0.2 and entropy collapsed after 50 updates. What do you change?",
    "Why do RL papers report results over multiple seeds, and how would you compare two algorithms fairly?",
  ],

  resources: [
    { title: "Spinning Up in Deep RL (OpenAI)", url: "https://spinningup.openai.com", type: "course", note: "the engineer’s intro to deep RL and PPO" },
    { title: "Sutton & Barto — RL: An Introduction (2nd ed.)", url: "http://incompleteideas.net/book/the-book-2nd.html", type: "book", note: "free canonical textbook" },
    { title: "David Silver — UCL RL lectures", url: "https://www.davidsilver.uk/teaching/", type: "video", note: "intuition-building lecture series" },
    { title: "CleanRL", url: "https://github.com/vwxyzjn/cleanrl", type: "repo", note: "single-file PPO/DQN you can read end to end" },
    { title: "The 37 Implementation Details of PPO", url: "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/", type: "article", note: "what actually makes PPO work" },
    { title: "PPO paper (Schulman et al. 2017)", url: "https://arxiv.org/abs/1707.06347", type: "paper", note: "the clipped surrogate objective" },
    { title: "Hugging Face Deep RL Course", url: "https://huggingface.co/learn/deep-rl-course/unit0/introduction", type: "course", note: "hands-on alternative with notebooks" },
    { title: "Gymnasium", url: "https://gymnasium.farama.org/", type: "docs", note: "standard environment API" },
    { title: "Lilian Weng — Policy Gradient Algorithms", url: "https://lilianweng.github.io/posts/2018-04-08-policy-gradient/", type: "article", note: "compact reference for the whole PG family" },
  ],
});
