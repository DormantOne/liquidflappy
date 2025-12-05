🔥 In What Furnace Was Thy Brain?
A Liquid Neural Network Learns to Fly from Raw Vision
Flappy, flappy, flying right,
In the pipescape of the night,
What immortal hand or eye,
Could frame thy neural symmetry?


To forge a brain capable of navigating the chaos, we built a Digital Furnace—a four-stage engine designed to hammer random matrices into intelligent thought.

The Four Stages
StageThe Work🪨The OreSix random numbers become a brain's blueprint🌊The Liquid MetalA reservoir of chaos learns to ripple with meaning🔥The FirePolicy gradients melt and reshape the connections⚗️The CrucibleChampions are captured; their souls passed to offspring

⛓️⛓️⛓️

🪨 Stage 1: The Ore (The Genome)

What the hammer? What the chain? In what furnace was thy brain?

One-liner: Every agent begins as six random numbers—and those numbers determine everything.
The Hyperparameter Vector
The furnace doesn't hand-tune architecture. It evolves it. Each agent is born from a 6-dimensional genome, values between 0 and 1:
GeneControlsDecoded Range0Reservoir Size100–350 neurons1Connection Density5–30% sparse2Leak Rate0.05–0.65 (memory decay)3Spectral Radius0.5–1.5 (edge of chaos)4Learning Rate10⁻⁴ to 10⁻² (log scale)5Input Gain0.2–2.7 (sensory amplification)
This is architecture search as evolution. The system doesn't know that 161 neurons with 0.60 leak rate and 1.14 spectral radius will work. It discovers this through competition.
What Evolution Found
After 23 generations, the winning genome converged to:
Neurons:        161    (moderate capacity)
Leak Rate:      0.60   (reactive, short memory)  
Spectral Radius: 1.14  (slightly chaotic—past the edge!)
Learning Rate:  3.5e-3 (aggressive)
The spectral radius is fascinating—evolution pushed it above 1.0, into technically unstable territory. But the high leak rate compensates, rapidly "forgetting" and preventing runaway dynamics. The system discovered its own stability trick.

⛓️⛓️⛓️

🌊 Stage 2: The Liquid Metal (The Reservoir)

In what distant deeps or skies, burnt the fire of thine eyes?

One-liner: A pool of randomly-connected neurons that we never train directly—we only teach a thin readout layer to interpret its ripples.
How It Works
This is an Echo State Network (also called a Liquid State Machine). The architecture:
Visual Input (150) + Velocity (1)
            ↓
    ┌───────────────────┐
    │   VisualCortex    │  ← Trained (learns to see)
    │   151 → 64 → 32   │
    └───────────────────┘
            ↓
    ┌───────────────────────────────────────┐
    │           RESERVOIR                   │
    │   ┌─────────────────────────────┐    │
    │   │  W_in: input projection     │ ← FROZEN (random)
    │   │  W_rec: sparse recurrence   │ ← FROZEN (random, scaled)
    │   │  h(t) = (1-α)h(t-1) + α·f() │    │
    │   └─────────────────────────────┘    │
    │              ↓                        │
    │   ┌─────────────────────────────┐    │
    │   │  Readout: reservoir → action │ ← Trained
    │   └─────────────────────────────┘    │
    └───────────────────────────────────────┘
            ↓
      Action (flap / don't flap)
Why This Works
The random reservoir acts as a temporal feature expander. When sensory data pours in:

It ripples through the recurrent connections
Recent inputs leave echoes in the state
The readout layer learns to interpret these ripples

The bird doesn't need explicit memory of pipe velocity—the reservoir holds that information in its dynamics. The leak rate controls how long echoes persist. The spectral radius controls how richly they interact.
We train only ~5% of the parameters. The rest is beautiful, frozen chaos.

⛓️⛓️⛓️

🔥 Stage 3: The Fire (Training)

And what shoulder, and what art, could twist the sinews of thy heart?

One-liner: Policy gradients heat the metal—good flights harden the weights, crashes melt them down to try again.
The REINFORCE Algorithm
Each agent learns through trial by fire:
pythonfor each episode:
    1. Play the game, sampling actions from policy
    2. Record: log_prob(action), reward
    3. Compute discounted returns (γ = 0.95)
    4. Update: θ ← θ + α · ∇log(π) · Return
```

Actions that led to survival get reinforced. Actions that led to death get weakened. Simple, brutal, effective.

### The Varsity System

Not all ore deserves the full furnace. We filter:
```
┌─────────────────────────────────────────────┐
│           THE VARSITY FILTER                │
├─────────────────────────────────────────────┤
│  1. Light training (80 episodes)            │
│  2. Scout evaluation (3 test runs)          │
│  3. Average score ≥ 0.3 pipes?              │
│     ├─ NO  → Discard. Weak ore.             │
│     └─ YES → Heavy training (300 episodes)  │
└─────────────────────────────────────────────┘
This focuses compute on promising genomes. ~80% of random architectures produce garbage. Why waste fire on slag?

⛓️⛓️⛓️

⚗️ Stage 4: The Crucible (Lamarckian Inheritance)

Did he smile his work to see? Did he who made the Lamb make thee?

One-liner: When a Titan emerges, we trap its soul—and inject those learned weights directly into its children.
The Trap
Normal evolution is Darwinian: parents pass genes, children learn from scratch. A bird that mastered flight has children who crash into the first pipe.
Our furnace uses Lamarckian inheritance: learned traits pass directly to offspring.
python# When an agent scores ≥2 pipes:
if env.score >= 2:
    trapped_weights = agent.state_dict()  # Capture the soul
    
# Next generation:
if parent.weights is not None:
    child.load_state_dict(parent.weights)  # Born knowing how to fly
```

### Why This Is Overpowered

| Generation | Darwinian | Lamarckian |
|------------|-----------|------------|
| N | Bird learns to fly after 300 episodes | Bird learns to fly after 300 episodes |
| N+1 | Children crash. Start over. | Children are **born flying**. |
| N+2 | Still crashing. | Children **improve** on parents. |
| N+10 | Maybe one figures it out again. | Dynasty of champions. |

Skills compound. Each generation starts where the last one peaked. The fitness ratchets upward.

### The Weight File

Champions are saved to disk:
```
honest_pantheon_v2.json   → Genome (the blueprint)
honest_pantheon_weights.pt → Weights (the learned soul)
Stop the furnace, restart it tomorrow—the Titans remain.

⛓️⛓️⛓️

👁️ The Honest Vision (No Cheating)
This is the critical part. The bird receives only:
InputSizeDescriptionVisual field15010×15 grid (pipes=1.0, bird=0.5, empty=0.0)Velocity1Proprioception—"am I rising or falling?"Total151—
NOT provided (these would be cheating):

❌ Distance to next pipe
❌ Delta-Y to gap center

The network must learn to see:

Scan the grid for vertical bars of 1.0s (pipes)
Find the break (the gap)
Determine if self (0.5) is above or below the gap
Time the flap

This is genuine visual processing, not a lookup table.

⛓️⛓️⛓️

📊 Results
After ~23 generations (~10 minutes on 4 CPU cores):
MetricValueBest Score100+ pipesFitness3038Architecture161 neurons, 0.60 leak, 1.14 spectral radius
The bird flies indefinitely. Smooth, efficient, adaptive.

🧠 What I Learned

Hyperparameter search IS architecture search — those 6 numbers control everything
Liquid State Machines are underrated — temporal memory for free
Lamarckian inheritance is overpowered — skills shouldn't reset each generation
Vision-only learning is possible — but much harder than oracle sensors
Evolution finds weird solutions — spectral radius >1.0 with high leak rate


