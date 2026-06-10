# Social-Risk Graph-Augmented MAPPO for Dynamic Multi-Robot Navigation Without Explicit Inter-Agent Communication

Note: This document is a TIV-style English first draft. The current version focuses on paper structure, method description, innovation statements, and an executable experimental plan. Numerical results, figure numbers, and references can be updated in later revisions.

## Abstract

Dynamic multi-robot navigation requires a policy to jointly handle goal reaching, static obstacle avoidance, robot-robot interactions, and moving obstacle interference. Classical analytic approaches such as ORCA offer strong geometric interpretability and real-time execution, but they are limited in adapting to complex scene semantics and long-horizon task objectives. Deep multi-agent reinforcement learning provides stronger policy expressiveness, yet existing graph-based approaches often entangle local scene understanding, inter-agent relation modeling, and communication robustness within the same policy backbone, resulting in unstable training and unclear deployment pathways. To address this issue, this paper proposes a social-risk graph-augmented MAPPO framework that decouples a stable MLP-LSTM local navigation backbone from a lightweight social-risk GAT side branch. Instead of relying on explicit communication, the proposed method constructs an ego-centric social graph from oracle-style social-risk tokens encoded by relative position, closing speed, time-to-collision, and predicted minimum separation. The graph embedding is injected into the actor through a post-backbone residual gating mechanism, so that relation reasoning enhances but does not override the navigation backbone. In addition, the paper adopts a training-deployment consistent interface and a progressive curriculum over dynamic interaction difficulty to improve robustness and engineering viability. Expected results suggest that the proposed framework can reduce collisions and deadlocks while outperforming MLP, communication-graph GAT, and ORCA baselines in dynamic multi-robot scenarios.

Index Terms: multi-robot navigation, MAPPO, graph attention network, social risk modeling, curriculum learning, centralized training decentralized execution

## I. Introduction

Safe and efficient navigation in shared spaces is a core problem for warehouse robotics, campus delivery, and service robot systems. Unlike single-robot navigation, multi-robot systems must handle strong interaction patterns such as head-on encounters, crossing conflicts, merging behavior, and congestion under partial observability. In narrow corridors, intersections, and dense dynamic scenes, a local policy that cannot explicitly reason about robot-robot interactions may easily lead to collisions, deadlocks, or overly conservative behavior.

Classical reciprocal collision avoidance methods, such as ORCA and RVO, address this problem by using relative positions, relative velocities, and collision horizons to compute collision-free local velocities. These methods are interpretable, communication-free, and computationally efficient. However, they mainly solve short-horizon local avoidance and do not naturally leverage rich local perception, scene semantics, or long-term task rewards. By contrast, deep reinforcement learning can integrate sensory observations, target information, and interaction history into a unified decision process. In particular, MAPPO has become a practical and stable baseline for multi-agent control under centralized training and decentralized execution.

Nevertheless, directly incorporating graph neural networks into multi-robot policies remains problematic. In many existing formulations, the graph module is expected to simultaneously model local scene structure, aggregate neighbor interactions, and absorb communication imperfections such as latency, jitter, and packet loss. This design overloads the graph module and makes the optimization landscape highly coupled. As a result, poor graph performance may not imply that relation modeling is ineffective; rather, it may indicate that the graph model has been assigned too many responsibilities at once.

This work revisits the role of graph learning in dynamic multi-robot navigation and argues that graph networks should primarily model inter-agent social risk, instead of replacing the whole navigation backbone. Based on this view, we preserve the same local navigation backbone as a strong MLP-LSTM MAPPO baseline and introduce a lightweight social-risk GAT branch that only models high-risk agent-agent interactions. To avoid unfair dependence on noisy communication channels, the actor does not learn from an explicit communication graph by default; instead, it uses oracle-style social-risk tokens derived from relative geometry and short-horizon motion prediction. In this sense, the proposed design combines the geometric intuition of ORCA with the representation learning power of MAPPO in a more stable and engineering-friendly manner.

The main contributions of this paper are summarized as follows:

1. We propose a relation-decoupled actor architecture that separates a stable MLP-LSTM local navigation backbone from a lightweight social-risk GAT branch, so that graph reasoning only enhances high-risk inter-agent interactions instead of taking over the whole navigation policy.
2. We design a communication-free social graph construction mechanism based on relative position, closing speed, time-to-collision, and predicted minimum separation, enabling explicit agent-agent risk modeling without relying on explicit communication quality assumptions.
3. We introduce a post-backbone residual gating fusion strategy that allows graph features to modulate policy representations only when social-risk-aware correction is needed, thereby improving training stability.
4. We establish a training-to-deployment consistent engineering pipeline, including shared swarm-state interfaces, curriculum training over dynamic difficulty, and decentralized policy execution, which improves the practical reproducibility of multi-robot RL.

## II. Related Work

### A. Analytic Multi-Robot Collision Avoidance

ORCA and RVO are representative analytic methods for multi-robot local collision avoidance. They formulate reciprocal avoidance constraints from relative motion geometry and compute collision-free local velocities in a distributed fashion. Their strengths include interpretability, no need for policy training, and low inference cost. However, their ability to incorporate complex scene semantics and long-horizon task objectives is limited.

### B. Reinforcement Learning for Multi-Robot Navigation

Deep reinforcement learning has been widely applied to local navigation and multi-agent coordination. Compared with analytic baselines, RL methods can integrate sensor observations, target information, and interaction history into a data-driven policy. MAPPO is especially attractive because its centralized training and decentralized execution paradigm yields good practical stability. Still, standard MLP or recurrent policies do not explicitly represent structured inter-agent relations.

### C. Graph-Based Multi-Agent Relation Modeling

Graph neural networks and graph attention networks provide a natural way to represent structured interactions among agents. Prior work has shown that graph reasoning can help prioritize important neighbors and improve coordination. However, many existing graph-based policies entangle relation reasoning with communication modeling and local scene encoding, which often complicates optimization and deployment. In contrast, the proposed framework uses the graph branch only for social-risk relation enhancement.

## III. Problem Formulation

We consider a dynamic navigation task involving $N$ mobile robots, static obstacles, and moving obstacles in a two-dimensional environment. At each time step, each robot receives a local observation including lidar-derived local geometry, target-relative features, ego motion, short-horizon safety features, and robot-robot interaction risk features. The objective is to reach the assigned goal while minimizing collisions, deadlocks, and unnecessary delay.

Under the centralized training and decentralized execution paradigm, robot $i$ executes a decentralized policy $\pi_\theta(a_i^t|o_i^t)$ from local observation $o_i^t$, while the critic may access a global state $s^t$ during training. The key challenge is to incorporate explicit inter-agent relation reasoning without destabilizing the local navigation backbone.

## IV. Method

### A. Overview

The proposed framework consists of two components:

1. A local navigation backbone that reuses the same MLP-LSTM actor structure as the strong baseline.
2. A social-risk graph branch that models high-risk robot-robot interactions through an ego-centric GAT.

The graph branch does not directly output actions. Instead, it produces a relation-aware embedding that is injected into the actor through residual gating, so that the base navigation pathway remains stable.

### B. Local Navigation Backbone

The local navigation backbone processes:

1. Multi-frame lidar sector history.
2. Target-relative features in the robot body frame.
3. Ego linear and angular velocity.
4. Front-obstacle and near-collision safety features.
5. Optional short-horizon moving-obstacle prediction features.

The backbone follows a `LayerNorm + recurrent unit + policy head` structure. We deliberately keep this pathway consistent with the MLP baseline, so that performance gains can be mainly attributed to relation modeling instead of additional backbone capacity.

### C. Communication-Free Social-Risk Tokens

Instead of directly learning from a noisy communication graph, we construct oracle-style social-risk tokens for candidate neighbors. Each token contains:

1. Normalized relative position.
2. Closing or approaching speed.
3. Normalized time-to-collision (TTC).
4. Predicted minimum separation.
5. Aggregated social-risk score.

These variables are physically related to the geometric quantities used in ORCA, but here they serve as learned relation descriptors rather than explicit analytic constraints.

### D. Ego-Centric Social Graph

For each robot, the top-$K$ highest-risk tokens are selected and combined with the ego node to form an ego-centric graph. Only ego-risk node edges and self-loops are retained. Compared with dense fully connected graphs, this construction:

1. suppresses low-risk irrelevant neighbors,
2. focuses computation on the truly critical interactions, and
3. better matches the local decision scope of robot navigation.

### E. Social-Risk GAT Encoding

The ego context is encoded from local observations, while risk tokens are embedded by a shared MLP. A multi-head GAT then aggregates the graph with a risk-aware attention bias, yielding a social embedding that summarizes the local interaction pressure induced by surrounding robots. This embedding represents social relational structure rather than the full navigation state.

### F. Post-Backbone Residual Gating

Let $h_i$ denote the backbone feature and $g_i$ denote the graph embedding. The final fused feature is computed as

$$
\tilde{h}_i = h_i + \sigma(W[h_i; g_i]) \odot \Delta(g_i),
$$

where $\sigma(\cdot)$ is a gating function and $\Delta(\cdot)$ is a projection operator. This design allows the policy to fall back to near-backbone behavior under low interaction risk, while activating graph-based correction in challenging social encounters.

### G. Training-Deployment Consistency

Beyond policy design, the proposed framework aligns the training interface with decentralized deployment. The swarm-state topic schema and local observation semantics are consistent between training and policy execution nodes, making the transition from simulation to decentralized multi-robot deployment more practical.

### H. Curriculum Learning

To improve robustness under increasing interaction complexity, a progressive curriculum is used:

1. static warm-up scenes,
2. low-density moving obstacles,
3. medium-to-high dynamic disturbance,
4. dense interaction and complex topology scenarios.

The purpose is to first stabilize local navigation behavior and then gradually introduce stronger social-risk and dynamic disturbance patterns.

## V. Innovation Summary

For later abstract refinement, contribution slides, and response letters, the novelty of the proposed work can be condensed into the following four points.

### Innovation 1: Relation-Decoupled Policy Design

Instead of letting the graph module dominate the entire actor, the proposed framework structurally decouples local navigation from social relation reasoning, which improves optimization stability and enables fairer comparison with strong MLP baselines.

### Innovation 2: Communication-Free Social Graph Construction

The proposed method builds a social graph from TTC-, closing-speed-, and separation-aware risk tokens rather than directly relying on a noisy neighbor communication graph, offering a novel way to represent inter-agent interaction risk in dynamic navigation.

### Innovation 3: Risk-Adaptive Residual Gating

The post-backbone residual gating strategy lets graph features influence the action policy only when socially meaningful correction is required, thereby reducing the risk of graph-induced policy collapse.

### Innovation 4: Training-to-Deployment Engineering Consistency

The framework is supported by a practical pipeline that spans TurtleBot3 multi-robot simulation, curriculum learning, checkpoint testing, and decentralized execution nodes, making the contribution not only algorithmic but also system-oriented.

## VI. Experimental Design

### A. Baselines

The proposed method should be compared with:

1. MAPPO-MLP-LSTM,
2. the proposed Social-Risk GAT,
3. a communication-graph GAT baseline, and
4. ORCA.

### B. Evaluation Scenarios

The evaluation should follow the existing staged environments:

1. static warehouse-like scenes,
2. corridor swap scenarios,
3. dynamic obstacle stress conditions, and
4. high-density interaction scenes.

### C. Metrics

The main metrics should include:

1. success rate,
2. collision rate,
3. average completion time,
4. deadlock rate,
5. minimum inter-robot distance,
6. average social risk,
7. inference latency, and
8. model size.

### D. Core Experiments

The most important experiments are:

1. main comparison against MLP, communication-GAT, and ORCA,
2. architectural ablation on the GAT branch and fusion module,
3. scaling tests with 2/4/6 robots, and
4. robustness analysis under different moving-obstacle densities and speeds.

## VII. Expected Results and Discussion

The proposed framework is expected to:

1. significantly reduce collisions and deadlocks in interaction-heavy scenes,
2. outperform pure MLP policies in head-on, crossing, and congestion cases,
3. train more stably than communication-graph GAT policies, especially in the early and mid stages of learning, and
4. achieve better task efficiency than ORCA in dynamic scenes with richer context.

Importantly, the proposed method should not be presented as a complete replacement for ORCA. A more accurate positioning is that it is an ORCA-inspired learned social-risk modeling framework that preserves geometric interaction priors while benefiting from long-horizon policy learning.

## VIII. Conclusion

This paper presents a social-risk graph-augmented MAPPO framework for dynamic multi-robot navigation. By decoupling the local navigation backbone from inter-agent relation reasoning, introducing a communication-free social graph, and using residual gating for risk-aware fusion, the proposed design offers a more stable and practically meaningful direction for graph-enhanced multi-robot reinforcement learning. If validated experimentally, the framework can provide a promising contribution to the TIV community at the intersection of intelligent vehicles, robot interaction, and learning-based navigation.

## Appendix: Suggested Next Updates

The next revision should prioritize the following additions:

1. one overview figure of the architecture,
2. one training-curve figure,
3. one main quantitative comparison table,
4. one ablation table, and
5. qualitative visualization cases for head-on and crossing interactions.

