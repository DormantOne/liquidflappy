"""
================================================================================
🔥 HONEST FURNACE v2.0 - No Cheating Allowed
================================================================================

    Flappy, flappy, flying right,
    In the pipescape of the night,
    What immortal hand or eye,
    Could frame thy neural symmetry?

================================================================================
WHAT MAKES THIS "HONEST"?
================================================================================

The bird receives ONLY:
  - 150 values: A 10x15 visual grid (pipes=1.0, self=0.5, empty=0.0)
  - 1 value: Its own velocity (proprioception - it can "feel" this)
  
Total: 151 inputs

The bird does NOT receive:
  - Distance to next pipe (would be CHEATING - oracle knowledge)
  - Delta-Y to gap center (would be CHEATING - oracle knowledge)
  - Its own Y position as a number (redundant - visible in grid as 0.5)

The network must LEARN to:
  1. Scan the visual field for obstacles (1.0 values)
  2. Find the vertical gap (contiguous 0.0 values)
  3. Determine if it's above or below that gap
  4. Time the flap correctly based on velocity and distance

This is MUCH harder than the cheating version. It may take longer to learn,
or may fail entirely. But if it succeeds, it's REAL visual learning.

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import random
import threading
import logging
import json
import os
from flask import Flask, render_template_string, jsonify, request
from concurrent.futures import ProcessPoolExecutor

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = torch.device("cpu")
CORE_COUNT = 4                # Parallel workers for evolution
TRAIN_FRAME_LIMIT = 6000      # Max frames per evaluation run
MIN_SHOW_SCORE = 40           # Fitness threshold to trigger demo (lowered for honest)
GRID_W, GRID_H = 20, 15       # World dimensions  
VISION_W = 10                 # How far ahead the bird can see

# File paths for persistence
SAVE_FILE = "honest_pantheon_v2.json"
WEIGHTS_FILE = "honest_pantheon_weights.pt"

# Training intensity - increased for harder task
INITIAL_TRAIN_EPOCHS = 80     # Was 60 in cheating version
VARSITY_TRAIN_EPOCHS = 300    # Was 250 in cheating version
VARSITY_THRESHOLD = 0.3       # Was 0.5 - lower bar since honest is harder

# =============================================================================
# SHARED STATE (for web visualization)
# =============================================================================

SYSTEM_STATE = {
    'status': 'INITIALIZING HONEST FURNACE...',
    'generation': 0,
    'best_score': 0.0,
    'best_genome': None,
    'best_weights': None, 
    'mode': 'TRAINING',
    'logs': [],
    'game_view': {},   
    'brain_view': {},
    'pop_vectors': [],         # Population genome vectors for visualization
    'history_vectors': [],     # Elite trajectory through hyperspace
    'hyperparams': {},
    'current_id': 'Waiting...',
    'manual_demo_request': False
}

def add_log(msg):
    """Thread-safe logging to both console and web UI."""
    print(f"[HONEST] {msg}")
    SYSTEM_STATE['logs'].insert(0, msg)
    if len(SYSTEM_STATE['logs']) > 25: 
        SYSTEM_STATE['logs'].pop()

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# =============================================================================
# 1. GENETICS - The 6-dimensional genome controlling brain architecture
# =============================================================================

class GeneDecoder:
    """
    Converts a 6-dimensional vector [0,1]^6 into concrete hyperparameters.
    
    This is the "ore" that gets refined in the furnace. Evolution searches
    this 6D space to find architectures that can actually learn to see.
    
    Gene meanings:
      0: SIZE     - Number of neurons in reservoir (computational capacity)
      1: DENSITY  - Connection sparsity (efficiency vs expressiveness)
      2: LEAK     - Memory decay rate (how fast reservoir "forgets")
      3: RADIUS   - Spectral radius (chaos/stability tradeoff)
      4: LR       - Learning rate for policy gradient updates
      5: GAIN     - Input amplification (sensitivity to visual input)
    """
    
    GENE_NAMES = ['SIZE', 'DENSITY', 'LEAK', 'RADIUS', 'LR', 'GAIN']
    
    @staticmethod
    def decode(vector):
        v = np.clip(vector, 0.0, 1.0)
        return {
            # Gene 0: Reservoir size - more neurons = more capacity for visual processing
            'n_reservoir': int(v[0] * 250 + 100),   # Range: 100-350 (increased for harder task)
            
            # Gene 1: Connection density - sparse networks are efficient but less expressive
            'density': v[1] * 0.25 + 0.05,          # Range: 5-30%
            
            # Gene 2: Leak rate - controls temporal memory in the reservoir
            # Low leak = long memory, high leak = reactive/forgetful
            'leak_rate': v[2] * 0.6 + 0.05,         # Range: 0.05-0.65
            
            # Gene 3: Spectral radius - the "edge of chaos" parameter
            # < 1.0 = stable, fading dynamics
            # > 1.0 = chaotic, potentially richer but unstable
            'spectral_radius': v[3] * 1.0 + 0.5,    # Range: 0.5-1.5
            
            # Gene 4: Learning rate (log scale for better search coverage)
            'lr': 10 ** (-4.0 + (2.0 * v[4])),      # Range: 1e-4 to 1e-2
            
            # Gene 5: Input gain - how strongly visual signals affect reservoir
            'input_gain': v[5] * 2.5 + 0.2          # Range: 0.2-2.7
        }


# =============================================================================
# 2. NEURAL ARCHITECTURE
# =============================================================================

class VisualCortex(nn.Module):
    """
    Processes the raw visual field into compressed features.
    
    In the HONEST version, this is critically important - it must learn to:
    - Detect pipes in the visual field (vertical bars of 1.0s)
    - Locate gaps (breaks in the vertical bars)
    - Encode the bird's position relative to gaps
    
    Architecture: 151 → 64 → 32
    (Larger than cheating version's 154 → 48 → 24 because task is harder)
    """
    
    def __init__(self, input_size):
        super().__init__()
        # Larger hidden layer for complex visual processing
        self.l1 = nn.Linear(input_size, 64)
        self.l2 = nn.Linear(64, 32) 
    
    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x 


class DeepReservoir(nn.Module):
    """
    A Liquid State Machine (Echo State Network).
    
    The key insight: we DON'T train the recurrent connections (w_rec).
    We only train the linear readout layer to interpret reservoir dynamics.
    
    The reservoir provides:
    - Temporal memory (recent inputs leave "ripples" in the state)
    - Nonlinear expansion of the feature space
    - Implicit computation through chaotic dynamics
    
    Architecture:
        Visual Features (32) → W_in (frozen) → Reservoir State (N) 
                                                      ↓
                              W_rec (frozen, sparse) ←┘
                                                      ↓
                                            Readout (trained) → Action (3)
    """
    
    def __init__(self, input_dim, params):
        super().__init__()
        self.size = params['n_reservoir']
        self.leak = params['leak_rate']
        
        # --- INPUT PROJECTION (frozen) ---
        self.w_in = nn.Linear(input_dim, self.size, bias=False)
        with torch.no_grad():
            self.w_in.weight.uniform_(-params['input_gain'], params['input_gain'])
            self.w_in.weight.requires_grad_(False)

        # --- RECURRENT WEIGHTS (frozen, sparse) ---
        # Create sparse random connectivity matrix
        mask = (torch.rand(self.size, self.size) < params['density']).float()
        w_rec = (torch.rand(self.size, self.size) * 2 - 1) * mask
        
        # Scale by spectral radius for stability control
        # This tunes the reservoir to the "edge of chaos"
        try:
            eigenvalues = torch.linalg.eigvals(w_rec)
            max_eig = torch.max(torch.abs(eigenvalues)).item()
        except:
            # Fallback: use spectral norm approximation
            max_eig = torch.linalg.norm(w_rec, ord=2).item()
        
        if max_eig > 1e-6:
            w_rec = w_rec * (params['spectral_radius'] / max_eig)
            
        self.w_rec = nn.Parameter(w_rec, requires_grad=False)
        
        # --- READOUT LAYER (trained) ---
        # This is the ONLY thing we train in the reservoir
        # 3 outputs: [idle, flap, idle] - redundant outputs help exploration
        self.readout = nn.Linear(self.size, 3) 
        
        # Store connection indices for visualization
        indices = mask.nonzero().tolist()
        self.links = random.sample(indices, min(len(indices), 200))

    def forward(self, u, h):
        """
        Update reservoir state and compute action logits.
        
        The update rule implements leaky integration:
            h_new = (1 - leak) * h_old + leak * tanh(W_in @ u + W_rec @ h_old)
        
        Args:
            u: Visual features from VisualCortex [batch, 32]
            h: Previous reservoir state [batch, n_reservoir]
            
        Returns:
            logits: Action logits [batch, 3]
            h_new: Updated reservoir state [batch, n_reservoir]
        """
        recurrence = F.linear(h, self.w_rec)
        injection = self.w_in(u)
        update = torch.tanh(injection + recurrence)
        
        # Leaky integration blends old state with new
        h_new = (1 - self.leak) * h + self.leak * update
        
        # Readout interprets reservoir state as action preferences
        logits = self.readout(h_new)
        
        return logits, h_new


class Agent(nn.Module):
    """
    The complete agent: VisualCortex → DeepReservoir → Action
    
    ============================================================
    HONEST VERSION - NO CHEATING
    ============================================================
    
    Input breakdown (151 total):
    - Visual field: VISION_W × GRID_H = 10 × 15 = 150 values
      - 0.0 = empty space
      - 0.5 = bird's position (the agent can see itself)
      - 1.0 = pipe obstacle
    - Proprioception: 1 value
      - Velocity (the bird can "feel" if it's rising or falling)
    
    NOT included (these would be cheating):
    - Distance to next pipe (oracle knowledge)
    - Delta-Y to gap center (oracle knowledge)
    - Bird's Y position as a separate number (redundant - visible in grid)
    
    ============================================================
    """
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # HONEST INPUT DIMENSION: 150 visual + 1 velocity = 151
        # Compare to cheating version: 150 visual + 4 sensors = 154
        input_dim = (VISION_W * GRID_H) + 1
        
        self.vision = VisualCortex(input_dim)   # 151 → 64 → 32
        self.brain = DeepReservoir(32, params)  # 32 → reservoir → 3
        
    def forward(self, x, hidden):
        features = self.vision(x)
        return self.brain(features, hidden)


# =============================================================================
# 3. ENVIRONMENT - The Anvil
# =============================================================================

class GameEnv:
    """
    Flappy Bird environment with HONEST observation space.
    
    The bird receives:
    ✓ A 10×15 visual grid showing pipes (1.0) and itself (0.5)
    ✓ Its own velocity (proprioception - it can feel this)
    
    The bird does NOT receive:
    ✗ Distance to next pipe (would be cheating)
    ✗ Delta-Y to gap center (would be cheating)
    ✗ Its own Y position as a number (redundant - visible in grid)
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.bird_y = GRID_H / 2.0
        self.bird_vel = 0
        self.score = 0
        self.pipes = []
        self.timer = 0
        return self.get_obs()

    def get_obs(self):
        """
        Build the HONEST observation tensor.
        
        No cheating. No oracle knowledge. Just vision and proprioception.
        
        Returns:
            Tensor of shape [1, 151] containing:
            - 150 values: flattened 10×15 visual grid
            - 1 value: bird's velocity
        """
        # --- BUILD VISUAL GRID ---
        grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        
        # Mark bird position with 0.5 (distinguishable from pipes at 1.0)
        by = int(np.clip(self.bird_y, 0, GRID_H - 1))
        grid[by, 2] = 0.5
        
        # Mark pipes with 1.0
        for p in self.pipes:
            px = int(p['x'])
            if 0 <= px < GRID_W:
                grid[0:p['gap_top'], px] = 1.0        # Top obstacle
                grid[p['gap_bot']:GRID_H, px] = 1.0   # Bottom obstacle

        # Extract visual field: columns 2-11 (bird position to 10 ahead)
        visual_field = grid[:, 2 : 2 + VISION_W]
        
        # Handle edge case at world boundary
        if visual_field.shape[1] < VISION_W:
            pad = np.zeros((GRID_H, VISION_W - visual_field.shape[1]), dtype=np.float32)
            visual_field = np.hstack([visual_field, pad])

        # =====================================================================
        # HONEST SENSORS - NO CHEATING
        # =====================================================================
        # 
        # We provide ONLY velocity as proprioceptive information.
        # The bird can "feel" whether it's rising or falling.
        #
        # We do NOT provide:
        #   - dist: distance to next pipe (the bird must FIND this visually)
        #   - dy: delta-y to gap center (the bird must COMPUTE this from vision)
        #   - bird_y: explicit position (redundant - visible as 0.5 in grid)
        #
        # =====================================================================
        
        sensors = torch.tensor([
            self.bird_vel / 2.0,   # Normalized velocity (proprioception)
        ], dtype=torch.float32)
        
        # Flatten visual grid and concatenate with sensors
        flat_vis = torch.flatten(torch.tensor(visual_field))
        return torch.cat([flat_vis, sensors]).unsqueeze(0)

    def step(self, action):
        """
        Execute one game step.
        
        Args:
            action: 0 = do nothing, 1 = flap, 2 = do nothing
        
        Returns:
            observation: New observation tensor [1, 151]
            reward: Reward signal (float)
            done: Whether episode has ended (bool)
        """
        # --- PHYSICS ---
        if action == 1:
            self.bird_vel = -0.7  # Flap impulse (upward is negative Y)
        self.bird_vel += 0.1      # Gravity
        self.bird_y += self.bird_vel
        
        # --- PIPE SPAWNING ---
        self.timer += 1
        if self.timer > 20: 
            self.timer = 0
            gap_size = 5
            gap_y = random.randint(1, GRID_H - 1 - gap_size)
            self.pipes.append({
                'x': GRID_W, 
                'gap_top': gap_y, 
                'gap_bot': gap_y + gap_size, 
                'passed': False
            })
        
        # --- REWARD COMPUTATION ---
        # Note: Reward shaping uses internal state, but the AGENT doesn't
        # see this information. Reward shaping is a legitimate RL technique.
        # The bird just receives a number; it doesn't know WHY.
        reward = 0.0
        done = False
        
        # Shaping: small reward for being near the gap center
        target_y = GRID_H / 2.0
        for p in self.pipes:
            if p['x'] > 1.0:
                target_y = (p['gap_top'] + p['gap_bot']) / 2.0
                break
        dist_to_ideal = abs(self.bird_y - target_y)
        reward += max(0, (1.0 - (dist_to_ideal / GRID_H))) * 0.1

        # --- PIPE MOVEMENT AND COLLISION ---
        for p in self.pipes:
            p['x'] -= 0.5  # Pipes move left
            
            # Score: successfully passed a pipe!
            if not p['passed'] and p['x'] < 2:
                self.score += 1
                reward = 5.0 
                p['passed'] = True
            
            # Collision detection
            if 1.5 < p['x'] < 2.5:
                if not (p['gap_top'] <= self.bird_y <= p['gap_bot']):
                    reward = -5.0
                    done = True
        
        # --- BOUNDARY COLLISION ---
        if self.bird_y < 0 or self.bird_y >= GRID_H:
            reward = -5.0
            done = True
            
        # Clean up off-screen pipes
        self.pipes = [p for p in self.pipes if p['x'] > -1]
        
        return self.get_obs(), reward, done


# =============================================================================
# 4. TRAINING WORKER - The Fire
# =============================================================================

def run_life_cycle(data_packet):
    """
    Train and evaluate a single agent. Runs in a separate process.
    
    This implements the REINFORCE policy gradient algorithm on the
    trainable parts of the network (VisualCortex + Readout layer).
    
    Lifecycle:
    1. Build agent from genome (decode hyperparameters → construct network)
    2. Optional: Load pre-trained weights (Lamarckian inheritance)
    3. Initial training (INITIAL_TRAIN_EPOCHS episodes)
    4. Scout evaluation (3 runs) → Check if agent shows promise
    5. If promising: Extended "Varsity" training (VARSITY_TRAIN_EPOCHS)
    6. Final evaluation → Compute fitness score
    7. If good enough: Capture weights for offspring (soul trapping)
    
    Args:
        data_packet: Dict with 'genome', 'gen', optional 'weights'
        
    Returns:
        Dict with 'fitness', 'pipes', 'genome', 'params', 'id', 
        'varsity', 'weights', 'titan'
    """
    genome = data_packet['genome']
    gen = data_packet['gen']
    pretrained = data_packet.get('weights')
    torch.set_num_threads(1)  # Prevent thread explosion in multiprocessing
    
    try:
        # --- BUILD AGENT FROM GENOME ---
        params = GeneDecoder.decode(genome)
        env = GameEnv()
        agent = Agent(params)
        
        # --- LAMARCKIAN INHERITANCE ---
        # If parent had captured weights AND architecture matches, inherit them
        if pretrained:
            try: 
                agent.load_state_dict(pretrained)
            except:
                # Architecture mismatch (different n_reservoir) - skip
                pass

        # --- OPTIMIZER ---
        # We train VisualCortex (learns to see) and Readout (learns to decide)
        # The reservoir recurrent weights stay frozen
        optimizer = torch.optim.Adam([
            {'params': agent.vision.parameters()},
            {'params': agent.brain.readout.parameters()}
        ], lr=params['lr'])
        
        # --- TRAINING FUNCTION (REINFORCE) ---
        def train_block(epochs):
            """
            Policy gradient training using REINFORCE algorithm.
            
            For each episode:
            1. Roll out trajectory, sampling actions from policy
            2. Compute discounted returns (future reward from each state)
            3. Update policy to increase probability of good actions
            """
            agent.train()
            for _ in range(epochs):
                state = env.reset()
                h = torch.zeros(1, params['n_reservoir'])
                log_probs, rewards = [], []
                
                # --- ROLLOUT ---
                step_count = 0
                while step_count < TRAIN_FRAME_LIMIT:
                    logits, h = agent(state, h)
                    dist = torch.distributions.Categorical(F.softmax(logits, dim=1))
                    action = dist.sample()
                    
                    state, r, done = env.step(action.item())
                    log_probs.append(dist.log_prob(action))
                    rewards.append(r)
                    step_count += 1
                    
                    if done: 
                        break
                
                # --- COMPUTE DISCOUNTED RETURNS ---
                R = 0
                returns = []
                for r in reversed(rewards):
                    R = r + 0.95 * R  # Discount factor γ = 0.95
                    returns.insert(0, R)
                returns = torch.tensor(returns)
                
                # Normalize returns for stable gradients
                if len(returns) > 1:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                
                # --- POLICY GRADIENT UPDATE ---
                # Loss = -log_prob × return (negative because we maximize)
                loss = [(-lp * ret) for lp, ret in zip(log_probs, returns)]
                if loss:
                    optimizer.zero_grad()
                    torch.stack(loss).sum().backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()

        # --- PHASE 1: INITIAL TRAINING ---
        train_block(INITIAL_TRAIN_EPOCHS)

        # --- PHASE 2: SCOUT EVALUATION ---
        # Test if this agent shows any promise before investing more compute
        agent.eval()
        scout_score = 0
        for _ in range(3):
            s = env.reset()
            h = torch.zeros(1, params['n_reservoir'])
            while True:
                with torch.no_grad():
                    l, h = agent(s, h)
                    a = torch.argmax(l).item()
                s, _, d = env.step(a)
                if d:
                    scout_score += env.score
                    break
        
        # --- PHASE 3: VARSITY TRAINING (if promising) ---
        is_varsity = False
        if (scout_score / 3.0) >= VARSITY_THRESHOLD:
            is_varsity = True
            train_block(VARSITY_TRAIN_EPOCHS)

        # --- PHASE 4: FINAL EVALUATION ---
        agent.eval()
        state = env.reset()
        h = torch.zeros(1, params['n_reservoir'])
        fitness = 0
        
        while fitness < TRAIN_FRAME_LIMIT:
            with torch.no_grad():
                l, h = agent(state, h)
                a = torch.argmax(l).item()
            state, _, done = env.step(a)
            fitness += 1
            if done:
                break
        
        # Fitness = survival frames + bonus for pipes passed
        final_score = fitness + (env.score * 50)
        
        # --- SOUL CAPTURE (Lamarckian inheritance) ---
        # If agent is good enough, save weights for offspring
        trapped_weights = None
        if env.score >= 2:  # Threshold for weight capture
            trapped_weights = {k: v.cpu() for k, v in agent.state_dict().items()}

        return {
            'fitness': final_score,
            'pipes': env.score,
            'genome': genome,
            'params': params,
            'id': f"G{gen}-{random.randint(100,999)}",
            'varsity': is_varsity,
            'weights': trapped_weights,
            'titan': True if pretrained else False
        }
        
    except Exception as e:
        return {'fitness': 0, 'error': str(e), 'genome': genome}


# =============================================================================
# 5. EVOLUTION ENGINE - The Crucible
# =============================================================================

def load_data():
    """Load previous best genome and weights from disk."""
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, 'r') as f:
                data = json.load(f)
                SYSTEM_STATE['best_score'] = data.get('score', 0)
                SYSTEM_STATE['generation'] = data.get('gen', 0)
                SYSTEM_STATE['best_genome'] = data.get('genome', None)
                add_log(f"📂 Loaded save. Best: {SYSTEM_STATE['best_score']:.0f}")
        except Exception as e:
            add_log(f"⚠️ Save load error: {e}")
    
    if os.path.exists(WEIGHTS_FILE):
        try:
            SYSTEM_STATE['best_weights'] = torch.load(WEIGHTS_FILE, map_location=DEVICE)
            add_log("🧠 Loaded neural weights")
        except Exception as e:
            add_log(f"⚠️ Weight load error: {e}")


def save_data(score, gen, genome, weights=None):
    """Save current best genome and weights to disk."""
    with open(SAVE_FILE, 'w') as f:
        json.dump({
            'score': score,
            'gen': gen,
            'genome': genome.tolist() if isinstance(genome, np.ndarray) else genome
        }, f)
    
    if weights is not None:
        try:
            torch.save(weights, WEIGHTS_FILE)
        except Exception as e:
            add_log(f"⚠️ Weight save error: {e}")


class EvolutionEngine(threading.Thread):
    """
    The main evolutionary loop.
    
    Each generation:
    1. Spawn population of agents (parallel training)
    2. Train and evaluate each agent
    3. Select top 3 as elites
    4. Create next generation:
       - Elites pass BOTH genome AND weights (Lamarckian)
       - Children get mutated genome, may inherit weights
    5. Repeat
    """
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.pop_size = CORE_COUNT * 4  # 16 agents per generation
        
        # Initialize population with random genomes
        self.population_data = [
            {'genome': np.random.rand(6), 'weights': None} 
            for _ in range(self.pop_size)
        ]
        
    def run(self):
        load_data()
        gen = SYSTEM_STATE['generation'] + 1
        
        # Inject saved champion if available
        if SYSTEM_STATE['best_genome'] is not None:
            self.population_data[0]['genome'] = np.array(SYSTEM_STATE['best_genome'])
            if SYSTEM_STATE['best_weights']:
                self.population_data[0]['weights'] = SYSTEM_STATE['best_weights']
            add_log("🧬 Injected saved champion")

        with ProcessPoolExecutor(max_workers=CORE_COUNT) as executor:
            while self.running:
                
                # --- CHECK FOR MANUAL DEMO REQUEST ---
                if SYSTEM_STATE['manual_demo_request']:
                    SYSTEM_STATE['manual_demo_request'] = False
                    if SYSTEM_STATE['best_genome'] is not None:
                        add_log("▶️ User requested demo...")
                        champ_data = {
                            'params': GeneDecoder.decode(SYSTEM_STATE['best_genome']),
                            'id': "CHAMPION",
                            'weights': SYSTEM_STATE.get('best_weights'),
                            'genome': SYSTEM_STATE['best_genome']
                        }
                        self.run_demo_mode(champ_data)
                    else:
                        add_log("⚠️ No champion to demo yet")

                # Pause during demo
                while SYSTEM_STATE['mode'] == 'DEMO':
                    time.sleep(0.5)

                SYSTEM_STATE['status'] = f"🔥 FORGING GEN {gen}"
                
                # --- PARALLEL TRAINING ---
                tasks = []
                for p_data in self.population_data:
                    packet = {
                        'genome': p_data['genome'], 
                        'weights': p_data['weights'], 
                        'gen': gen
                    }
                    tasks.append(executor.submit(run_life_cycle, packet))
                
                # Collect results
                results = []
                for f in tasks:
                    res = f.result()
                    if 'error' not in res:
                        results.append(res)
                    else:
                        add_log(f"⚠️ Agent error: {str(res.get('error', '?'))[:40]}")
                
                if not results:
                    add_log("❌ All agents failed! Resetting...")
                    self.population_data = [
                        {'genome': np.random.rand(6), 'weights': None} 
                        for _ in range(self.pop_size)
                    ]
                    continue
                
                # Sort by fitness (best first)
                results.sort(key=lambda x: x['fitness'], reverse=True)
                
                # --- VISUALIZATION DATA ---
                pop_vecs = []
                for i, r in enumerate(results):
                    vec = list(r['genome']) + [1.0 if i < 3 else 0.0]
                    pop_vecs.append(vec)
                SYSTEM_STATE['pop_vectors'] = pop_vecs

                # Track elite trajectory through hyperspace
                elites = results[:3]
                avg_genome = np.mean([e['genome'] for e in elites], axis=0).tolist()
                SYSTEM_STATE['history_vectors'].append(avg_genome)
                if len(SYSTEM_STATE['history_vectors']) > 150: 
                    SYSTEM_STATE['history_vectors'].pop(0)

                best = results[0]
                SYSTEM_STATE['generation'] = gen
                
                # --- CHECK FOR NEW RECORD ---
                if best['fitness'] > SYSTEM_STATE['best_score']:
                    SYSTEM_STATE['best_score'] = best['fitness']
                    SYSTEM_STATE['best_genome'] = best['genome']
                    if best['weights']:
                        SYSTEM_STATE['best_weights'] = best['weights']
                    save_data(best['fitness'], gen, best['genome'], best.get('weights'))
                    add_log(f"🏆 NEW BEST: {best['id']} | Pipes: {best['pipes']} | Fit: {best['fitness']:.0f}")
                else:
                    tag = " [T]" if best.get('titan') else (" [V]" if best.get('varsity') else "")
                    add_log(f"Gen {gen}: {best['id']}{tag} | Pipes: {best['pipes']}")
                
                SYSTEM_STATE['hyperparams'] = best['params']

                # --- AUTO DEMO ---
                if best['fitness'] > MIN_SHOW_SCORE:
                    self.run_demo_mode(best)

                # --- SELECTION AND REPRODUCTION ---
                new_pop_data = []
                
                # Elites survive with genome AND weights (Lamarckian)
                for elite in elites:
                    new_pop_data.append({
                        'genome': elite['genome'], 
                        'weights': elite['weights']
                    })
                
                # Fill rest with mutated children
                while len(new_pop_data) < self.pop_size:
                    parent = random.choice(elites)
                    # Gaussian mutation on genome
                    child_genome = np.clip(
                        parent['genome'] + np.random.normal(0, 0.05, 6), 
                        0, 1
                    )
                    # Children may inherit parent weights (Lamarckian boost)
                    new_pop_data.append({
                        'genome': child_genome, 
                        'weights': parent.get('weights')
                    })

                self.population_data = new_pop_data
                gen += 1

    def run_demo_mode(self, agent_data):
        """Replay a champion agent in real-time for visualization."""
        SYSTEM_STATE['mode'] = 'DEMO'
        SYSTEM_STATE['current_id'] = agent_data['id']
        SYSTEM_STATE['status'] = "👁️ WATCHING"
        
        params = agent_data['params']
        env = GameEnv()
        agent = Agent(params)
        
        # Try to load weights
        loaded = False
        if agent_data.get('weights'):
            try:
                agent.load_state_dict(agent_data['weights'])
                loaded = True
            except:
                pass
            
        if not loaded and SYSTEM_STATE['best_weights']:
            try:
                agent.load_state_dict(SYSTEM_STATE['best_weights'])
                loaded = True
            except:
                pass

        # If no weights, do quick training
        if not loaded:
            add_log("⚠️ No weights - quick training...")
            optimizer = torch.optim.Adam([
                {'params': agent.vision.parameters()},
                {'params': agent.brain.readout.parameters()}
            ], lr=params['lr'])
            
            agent.train()
            for _ in range(50):
                s = env.reset()
                h = torch.zeros(1, params['n_reservoir'])
                lp, rw = [], []
                while True:
                    l, h = agent(s, h)
                    d = torch.distributions.Categorical(F.softmax(l, dim=1))
                    a = d.sample()
                    s, r, done = env.step(a.item())
                    lp.append(d.log_prob(a))
                    rw.append(r)
                    if done:
                        break
                
                R = 0
                ret = []
                for r in reversed(rw):
                    R = r + 0.95 * R
                    ret.insert(0, R)
                ret = torch.tensor(ret)
                if len(ret) > 1:
                    ret = (ret - ret.mean()) / (ret.std() + 1e-9)
                loss = [(-l * r) for l, r in zip(lp, ret)]
                if loss:
                    optimizer.zero_grad()
                    torch.stack(loss).sum().backward()
                    optimizer.step()

        # --- RUN DEMO ---
        agent.eval()
        state = env.reset()
        h = torch.zeros(1, params['n_reservoir'])
        
        while True:
            with torch.no_grad():
                vis_tensor = agent.vision(state)
                logits, h_new = agent.brain(vis_tensor, h)
                
                vis_features = vis_tensor.tolist()[0]
                res_activations = h_new.tolist()[0]
                action = torch.argmax(logits).item()
            
            state, _, done = env.step(action)
            h = h_new
            
            # Update visualization state
            SYSTEM_STATE['game_view'] = {
                'px': env.bird_y, 
                'pipes': env.pipes, 
                'score': env.score
            }
            SYSTEM_STATE['brain_view'] = {
                'vis': vis_features, 
                'res': res_activations,
                'links': agent.brain.links, 
                'out': F.softmax(logits, dim=1).tolist()[0]
            }
            
            time.sleep(0.06)  # ~16 FPS
            
            if done:
                add_log(f"👁️ Demo: {env.score} pipes")
                time.sleep(1.0)
                break
        
        SYSTEM_STATE['mode'] = 'TRAINING'


# =============================================================================
# 6. WEB UI - Window into the Furnace
# =============================================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🔥 HONEST FURNACE</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;800&display=swap" rel="stylesheet">
    <style>
        :root { 
            --bg: #050508; 
            --panel: #0c0c10; 
            --c1: #ff9d00;   /* Orange - honest color */
            --c2: #ff5500;   /* Deep orange */
            --err: #ff0055; 
        }
        body { background: var(--bg); color: var(--c1); font-family: 'JetBrains Mono', monospace; margin: 0; height: 100vh; overflow: hidden; display: flex; }
        aside { width: 340px; background: var(--panel); border-right: 1px solid #333; padding: 20px; display: flex; flex-direction: column; gap: 12px; z-index: 10; }
        h1 { margin: 0; font-size: 22px; text-shadow: 0 0 15px rgba(255, 157, 0, 0.3); }
        .subtitle { font-size: 10px; color: #666; margin-bottom: 10px; line-height: 1.4; }
        
        .warning { 
            background: #1a0a00; 
            border: 1px solid #ff5500; 
            padding: 10px; 
            border-radius: 4px; 
            font-size: 9px; 
            color: #ff9d00; 
            line-height: 1.5;
        }
        .warning b { color: #fff; }
        
        .card { background: #000; border: 1px solid #333; padding: 12px; border-radius: 6px; }
        .stat { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 11px; color: #888; }
        .val { color: #fff; font-weight: 800; }
        
        button {
            background: #1a0a00; 
            color: var(--c1); 
            border: 1px solid var(--c2); 
            padding: 12px; 
            font-family: inherit; 
            font-size: 12px; 
            cursor: pointer; 
            transition: all 0.2s;
            text-transform: uppercase; 
            font-weight: bold;
            border-radius: 4px;
        }
        button:hover { background: var(--c2); color: #000; }
        button:active { transform: translateY(2px); }
        
        #logs { flex: 1; overflow-y: auto; font-size: 10px; color: #555; margin-top: 10px; border-top: 1px solid #222; padding-top: 10px; }
        .log-item { margin-bottom: 4px; border-bottom: 1px solid #111; padding-bottom: 2px;}
        
        main { flex: 1; display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 2fr 1fr; gap: 15px; padding: 15px; }
        .viz-box { background: #000; border: 1px solid #333; border-radius: 8px; position: relative; overflow: hidden; }
        .viz-label { position: absolute; top: 10px; left: 10px; color: #666; font-size: 10px; font-weight: 800; text-transform: uppercase; z-index: 20; text-shadow: 0 0 5px #000;}
        canvas { display: block; width: 100%; height: 100%; }
        #status-bar { position: absolute; bottom: 15px; right: 20px; font-size: 11px; color: #444; }
        .demo-active { color: var(--c2) !important; text-shadow: 0 0 10px var(--c2); }
    </style>
</head>
<body>
    <aside>
        <h1>🔥 HONEST FURNACE</h1>
        <div class="subtitle">
            No cheating. No oracle sensors.<br>
            The bird must learn to SEE.
        </div>
        <div class="warning">
            <b>⚠️ HARD MODE ACTIVE</b><br><br>
            Input: <b>151</b> values (150 visual + 1 velocity)<br><br>
            <b>NOT provided:</b><br>
            • Distance to pipe<br>
            • Delta-Y to gap<br><br>
            The network must parse the visual grid, find pipes, locate gaps, and compute its own position.
        </div>
        <div class="card">
            <div class="stat"><span>STATUS</span><span id="status" class="val">INIT</span></div>
            <div class="stat"><span>GENERATION</span><span id="gen" class="val">0</span></div>
            <div class="stat"><span>HIGH SCORE</span><span id="best" class="val" style="color:var(--c2)">0</span></div>
            <div class="stat"><span>CURRENT</span><span id="aid" class="val">---</span></div>
        </div>
        <div class="card">
            <div class="stat"><span>NEURONS</span><span id="p-neu" class="val">0</span></div>
            <div class="stat"><span>LEAK RATE</span><span id="p-leak" class="val">0.0</span></div>
            <div class="stat"><span>SPECTRAL R</span><span id="p-sr" class="val">0.0</span></div>
            <div class="stat"><span>LEARN RATE</span><span id="p-lr" class="val">0.0</span></div>
        </div>
        <button onclick="triggerDemo()">▶ Demo Champion</button>
        <div id="logs"></div>
    </aside>
    <main>
        <div class="viz-box">
            <div class="viz-label">Live Feed (What Bird Sees)</div>
            <canvas id="gCanvas" width="600" height="400"></canvas>
        </div>
        <div class="viz-box">
            <div class="viz-label">Reservoir State (Neural Ripples)</div>
            <canvas id="bCanvas" width="600" height="400"></canvas>
        </div>
        <div class="viz-box" style="grid-column: span 2;">
            <div class="viz-label">Hyperparameter Space (Evolution Trajectory)</div>
            <canvas id="gMap" width="1200" height="200"></canvas>
        </div>
    </main>
    <div id="status-bar">HONEST VISION • 151 INPUTS • NO CHEATING</div>
    <script>
        const gc = document.getElementById('gCanvas').getContext('2d');
        const bc = document.getElementById('bCanvas').getContext('2d');
        const mc = document.getElementById('gMap').getContext('2d');
        
        function triggerDemo() {
            fetch('/trigger_demo', {method: 'POST'})
            .then(r => r.json())
            .then(d => {
                if(d.status !== 'ok') alert(d.message);
            });
        }

        let nodeCache = [];
        function getNodes(count, w, h) {
            if (nodeCache.length !== count) {
                nodeCache = [];
                for(let i=0; i<count; i++) nodeCache.push({x: 50+Math.random()*(w-100), y: 50+Math.random()*(h-100)});
            }
            return nodeCache;
        }

        const LABELS = ["SIZE", "DENSITY", "LEAK", "RADIUS", "LR", "GAIN"];
        let axisX = 2;
        let axisY = 4;
        let lastSwitch = Date.now();

        setInterval(() => {
            if (Date.now() - lastSwitch > 4000) {
                lastSwitch = Date.now();
                axisX = Math.floor(Math.random() * 6);
                do {
                    axisY = Math.floor(Math.random() * 6);
                } while(axisY === axisX);
            }
        }, 100);

        setInterval(() => {
            fetch('/status').then(r=>r.json()).then(d => {
                document.getElementById('status').innerText = d.mode;
                document.getElementById('status').className = d.mode === 'DEMO' ? 'val demo-active' : 'val';
                document.getElementById('gen').innerText = d.gen;
                document.getElementById('best').innerText = d.score.toFixed(0);
                document.getElementById('aid').innerText = d.id;
                document.getElementById('logs').innerHTML = d.logs.map(l=>`<div class="log-item">> ${l}</div>`).join('');
                if(d.params) {
                    document.getElementById('p-neu').innerText = d.params.n_reservoir || 0;
                    document.getElementById('p-leak').innerText = (d.params.leak_rate || 0).toFixed(2);
                    document.getElementById('p-sr').innerText = (d.params.spectral_radius || 0).toFixed(2);
                    document.getElementById('p-lr').innerText = (d.params.lr || 0).toExponential(1);
                }

                // Hyperparameter scanner
                if (d.pop_vectors && d.history_vectors) {
                    const W = 1200, H = 200;
                    mc.fillStyle = '#050508'; mc.fillRect(0,0,W,H);
                    
                    mc.strokeStyle = '#222'; mc.lineWidth = 1;
                    mc.beginPath(); mc.moveTo(0, H/2); mc.lineTo(W, H/2); mc.stroke();
                    mc.beginPath(); mc.moveTo(W/2, 0); mc.lineTo(W/2, H); mc.stroke();

                    mc.fillStyle = '#ff9d00'; mc.font = '12px monospace';
                    mc.fillText(`X: ${LABELS[axisX]}`, W-100, H-10);
                    mc.fillText(`Y: ${LABELS[axisY]}`, 10, 20);

                    if (d.history_vectors.length > 1) {
                        mc.beginPath();
                        d.history_vectors.forEach((v, i) => {
                            const x = 50 + v[axisX] * (W - 100);
                            const y = (H - 20) - v[axisY] * (H - 40);
                            if (i===0) mc.moveTo(x, y);
                            else mc.lineTo(x, y);
                        });
                        mc.strokeStyle = 'rgba(255, 157, 0, 0.5)'; mc.lineWidth = 2; 
                        mc.stroke();

                        const last = d.history_vectors[d.history_vectors.length-1];
                        const lx = 50 + last[axisX] * (W - 100);
                        const ly = (H - 20) - last[axisY] * (H - 40);
                        mc.fillStyle = '#fff'; mc.beginPath(); mc.arc(lx, ly, 4, 0, 6.28); mc.fill();
                    }

                    d.pop_vectors.forEach(v => {
                        const x = 50 + v[axisX] * (W - 100);
                        const y = (H - 20) - v[axisY] * (H - 40);
                        const isElite = v[6] > 0.5;
                        mc.beginPath(); mc.arc(x, y, isElite ? 5 : 2, 0, 6.28);
                        mc.fillStyle = isElite ? '#ff5500' : '#444'; 
                        mc.fill();
                        if (isElite) { mc.strokeStyle='#000'; mc.lineWidth=1; mc.stroke(); }
                    });
                }

                if (d.mode === 'DEMO' && d.game && d.game.score !== undefined) {
                    // Game render
                    const W = 600, H = 400;
                    const GW = 20, GH = 15; const S = W / GW; 
                    gc.fillStyle = '#050508'; gc.fillRect(0,0,W,H);
                    
                    // Vision cone highlight
                    gc.fillStyle = 'rgba(255, 157, 0, 0.03)';
                    gc.fillRect(2 * S, 0, 10 * S, H);
                    gc.strokeStyle = 'rgba(255, 157, 0, 0.2)';
                    gc.strokeRect(2 * S, 0, 10 * S, H);
                    
                    gc.fillStyle = '#ff5500';
                    d.game.pipes.forEach(p => {
                        const px = p.x * S;
                        gc.fillRect(px, 0, S, p.gap_top * S);
                        gc.fillRect(px, p.gap_bot * S, S, (GH - p.gap_bot) * S);
                        gc.fillStyle = 'rgba(255,255,255,0.1)';
                        const center = (p.gap_top + p.gap_bot)/2;
                        gc.fillRect(px, center*S-1, S, 2);
                        gc.fillStyle = '#ff5500';
                    });
                    
                    const by = d.game.px * S; 
                    gc.fillStyle = '#ff9d00'; gc.fillRect(2 * S, by, S, S);
                    gc.fillStyle = '#fff'; gc.font = '20px monospace'; gc.fillText(d.game.score, 20, 30);
                    gc.fillStyle = '#555'; gc.font = '10px monospace'; 
                    gc.fillText('VISION CONE (10x15)', 2.2*S, H-10);

                    // Brain render
                    bc.fillStyle = 'rgba(0,0,0,0.2)'; bc.fillRect(0,0,W,H);
                    const brain = d.brain;
                    if(brain && brain.res) {
                        const nodes = getNodes(brain.res.length, W, H);
                        if (brain.links) {
                            bc.lineWidth = 1;
                            brain.links.forEach(([src, dst]) => {
                                if (src < brain.res.length && dst < brain.res.length) {
                                    const val = Math.abs(brain.res[src]);
                                    if (val > 0.05) {
                                        const n1 = nodes[src], n2 = nodes[dst];
                                        bc.beginPath(); bc.moveTo(n1.x, n1.y); bc.lineTo(n2.x, n2.y);
                                        bc.strokeStyle = `rgba(255, 157, 0, ${val * 0.5})`; bc.stroke();
                                    }
                                }
                            });
                        }
                        nodes.forEach((n, i) => {
                            if (i < brain.res.length) {
                                const act = brain.res[i];
                                bc.beginPath(); bc.arc(n.x, n.y, 2+Math.abs(act)*5, 0, 6.28);
                                bc.fillStyle = act > 0 ? '#ff9d00' : '#ff0055'; bc.fill();
                            }
                        });
                        if (brain.out && brain.out[1] !== undefined) {
                            const jump = brain.out[1]; 
                            bc.fillStyle = `rgba(255,157,0,${jump})`;
                            bc.fillRect(W-40, H-40, 20, 20);
                            bc.strokeStyle = '#fff'; bc.strokeRect(W-40, H-40, 20, 20);
                            bc.fillStyle = '#666'; bc.font = '10px monospace'; bc.fillText('FLAP', W-45, H-45);
                        }
                    }
                }
            });
        }, 50);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/trigger_demo', methods=['POST'])
def trigger_demo():
    """API endpoint to manually trigger a champion demo."""
    if SYSTEM_STATE['best_genome'] is None:
        return jsonify({'status': 'error', 'message': 'No champion yet!'})
    
    if SYSTEM_STATE['mode'] == 'DEMO':
        return jsonify({'status': 'error', 'message': 'Demo already running'})
    
    SYSTEM_STATE['manual_demo_request'] = True
    return jsonify({'status': 'ok'})

@app.route('/status')
def get_status():
    """API endpoint for UI polling."""
    return jsonify({
        'status': SYSTEM_STATE['status'],
        'gen': SYSTEM_STATE['generation'],
        'score': SYSTEM_STATE['best_score'],
        'mode': SYSTEM_STATE['mode'],
        'id': SYSTEM_STATE['current_id'],
        'logs': SYSTEM_STATE['logs'],
        'game': SYSTEM_STATE['game_view'],
        'brain': SYSTEM_STATE['brain_view'],
        'params': SYSTEM_STATE.get('hyperparams', {}),
        'pop_vectors': SYSTEM_STATE.get('pop_vectors', []),
        'history_vectors': SYSTEM_STATE.get('history_vectors', [])
    })


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║              🔥 HONEST FURNACE v2.0 - No Cheating Allowed 🔥           ║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║                                                                       ║
    ║  The bird receives ONLY:                                              ║
    ║    • 150 values: 10×15 visual grid (pipes=1.0, bird=0.5)             ║
    ║    • 1 value: velocity (proprioception)                               ║
    ║    • TOTAL: 151 inputs                                                ║
    ║                                                                       ║
    ║  NOT provided (would be cheating):                                    ║
    ║    ✗ Distance to next pipe                                            ║
    ║    ✗ Delta-Y to gap center                                            ║
    ║                                                                       ║
    ║  The network must LEARN TO SEE:                                       ║
    ║    1. Parse visual grid for obstacles                                 ║
    ║    2. Find gaps in pipes                                              ║
    ║    3. Compute relative position                                       ║
    ║    4. Time flaps correctly                                            ║
    ║                                                                       ║
    ║  This is MUCH harder. It may fail. But if it works, it's REAL.       ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    add_log("🔥 HONEST FURNACE IGNITED")
    add_log(f"👁️ Input: 150 visual + 1 velocity = 151")
    add_log(f"🚫 NO oracle sensors (dist, dy)")
    
    engine = EvolutionEngine()
    engine.daemon = True
    engine.start()
    
    print("\n🌐 HONEST FURNACE: http://127.0.0.1:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
