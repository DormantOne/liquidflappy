🔥 In What Furnace Was Thy Brain?
A Liquid Neural Network Learns to Fly from Raw Vision

Flappy, flappy, flying right,
In the pipescape of the night,
What immortal hand or eye,
Could frame thy neural symmetry?

To forge a brain capable of navigating the chaos, we built a Digital Furnace—a four-stage engine designed to hammer random matrices into intelligent thought.
Show Image

Quick Start
pip install torch numpy flask
python honest_flap.py
# Open http://127.0.0.1:5000

The Four Stages
🪨 The Ore — Six random numbers become a brain's blueprint
🌊 The Liquid Metal — A reservoir of chaos learns to ripple with meaning
🔥 The Fire — Policy gradients melt and reshape the connections
⚗️ The Crucible — Champions are captured; their souls passed to offspring

🪨 Stage 1: The Ore (The Genome)

What the hammer? What the chain? In what furnace was thy brain?

Every agent begins as six random numbers—and those numbers determine everything.
The furnace doesn't hand-tune architecture. It evolves it. Each agent is born from a 6-dimensional genome with values between 0 and 1. These six genes control: reservoir size (100–350 neurons), connection density (5–30% sparse), leak rate (0.05–0.65), spectral radius (0.5–1.5), learning rate (10⁻⁴ to 10⁻²), and input gain (0.2–2.7).
This is architecture search as evolution. The system doesn't know that 161 neurons with 0.60 leak rate and 1.14 spectral radius will work. It discovers this through competition.
After 23 generations, the winning genome converged to: 161 neurons, 0.60 leak rate, 1.14 spectral radius, and 3.5e-3 learning rate. The spectral radius is fascinating—evolution pushed it above 1.0, into technically unstable territory. But the high leak rate compensates, rapidly "forgetting" and preventing runaway dynamics. The system discovered its own stability trick.

🌊 Stage 2: The Liquid Metal (The Reservoir)

In what distant deeps or skies, burnt the fire of thine eyes?

A pool of randomly-connected neurons that we never train directly—we only teach a thin readout layer to interpret its ripples.
This is an Echo State Network (also called a Liquid State Machine). The visual input (150 pixels) plus velocity (1 value) flows into a VisualCortex network (151 → 64 → 32), which is trained. This feeds into the Reservoir—a sparse, randomly-connected recurrent network where the input projection and recurrence weights are frozen. Only a thin readout layer at the end is trained to map reservoir state to actions.
The random reservoir acts as a temporal feature expander. When sensory data pours in, it ripples through the recurrent connections. Recent inputs leave echoes in the state. The readout layer learns to interpret these ripples.
The bird doesn't need explicit memory of pipe velocity—the reservoir holds that information in its dynamics. The leak rate controls how long echoes persist. The spectral radius controls how richly they interact.
We train only ~5% of the parameters. The rest is beautiful, frozen chaos.

🔥 Stage 3: The Fire (Training)

And what shoulder, and what art, could twist the sinews of thy heart?

Policy gradients heat the metal—good flights harden the weights, crashes melt them down to try again.
Each agent learns through the REINFORCE algorithm: play the game while sampling actions from the policy, record the log-probability of each action and the reward received, compute discounted returns with γ=0.95, then update weights to increase the probability of actions that led to high returns.
Actions that led to survival get reinforced. Actions that led to death get weakened. Simple, brutal, effective.
Not all ore deserves the full furnace. The Varsity System filters agents: light training (80 episodes), then a scout evaluation (3 test runs). If the agent averages less than 0.3 pipes, discard it—weak ore. If it shows promise, apply heavy training (300 episodes). This focuses compute on promising genomes. About 80% of random architectures produce garbage. Why waste fire on slag?

⚗️ Stage 4: The Crucible (Lamarckian Inheritance)

Did he smile his work to see? Did he who made the Lamb make thee?

When a Titan emerges, we trap its soul—and inject those learned weights directly into its children.
Normal evolution is Darwinian: parents pass genes, children learn from scratch. A bird that mastered flight has children who crash into the first pipe.
Our furnace uses Lamarckian inheritance: learned traits pass directly to offspring. When an agent scores 2 or more pipes, we capture its entire neural network state. In the next generation, children load these weights before training begins. They don't learn to fly—they are born flying.
Skills compound. Each generation starts where the last one peaked. The fitness ratchets upward. In Darwinian evolution, generation N+10 might still be struggling. In Lamarckian evolution, generation N+10 has built a dynasty of champions.
Champions are saved to disk: the genome in honest_pantheon_v2.json and the weights in honest_pantheon_weights.pt. Stop the furnace, restart it tomorrow—the Titans remain.

👁️ The Honest Vision (No Cheating)
This is the critical part.
The bird receives only 151 inputs: a 10×15 visual grid (150 values where pipes=1.0, bird=0.5, empty=0.0) plus its own velocity (1 value for proprioception).
The bird does NOT receive distance to the next pipe or delta-Y to the gap center. Those would be cheating—oracle knowledge that makes the problem trivial.
The network must learn to see: scan the grid for vertical bars of 1.0s (pipes), find the break (the gap), determine if self (0.5) is above or below the gap, and time the flap accordingly.
This is genuine visual processing, not a lookup table.

📊 Results
After approximately 23 generations (about 10 minutes on 4 CPU cores), the system achieved: 100+ pipes cleared, fitness score of 3038, using an architecture of 161 neurons with 0.60 leak rate and 1.14 spectral radius.
The bird flies indefinitely. Smooth, efficient, adaptive.

🧠 What I Learned
Hyperparameter search IS architecture search. Those 6 numbers control everything about how the brain is built and trained.
Liquid State Machines are underrated. You get temporal memory for free from the reservoir dynamics, without training recurrent connections.
Lamarckian inheritance is overpowered. Skills shouldn't reset each generation. Let children inherit what parents learned.
Vision-only learning is possible. But it's much harder than giving the network oracle sensors. The honest version took more generations to converge.
Evolution finds weird solutions. A spectral radius above 1.0 should be unstable, but paired with high leak rate, it works. The system discovered its own stability trick.

📁 Files

honest_flap.py — The complete furnace (~750 lines)
requirements.txt — Dependencies (torch, numpy, flask)


