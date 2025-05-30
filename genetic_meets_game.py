import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

# ---- PARAMETERS ----
st.title("Genetic Algorithm: Iterated Evolutionary Game (No Neutral Actions)")

GRID_SIZE = st.sidebar.slider("Grid size", min_value=5, max_value=30, value=10)
OBJECTIVE = st.sidebar.selectbox("Objective", ["Stability", "Equal richness", "One richest bot"])
GENS_PER_RUN = st.sidebar.slider("Generations per run", 1, 100, 10)

INIT_RICHNESS = 1000
ATTACK_COST = 10

# ---- BOT CLASS ----
class Bot:
    def __init__(self, richness=INIT_RICHNESS, dna=None):
        self.richness = richness
        # DNA: [p_attack_if_partnered, p_attack_if_attacked]
        if dna is None:
            self.dna = np.random.rand(2)
        else:
            self.dna = dna.copy()
        # For each neighbor (0-7): 0=partnered, 1=attacked (last round)
        self.memory = {i: random.choice([0,1]) for i in range(8)}
        self.alive = True

    def decide(self, neighbors, round_num):
        partners = []
        attacks = []
        for idx, nb in enumerate(neighbors):
            if not nb.alive:
                continue
            last = self.memory.get(idx, random.choice([0,1])) if round_num > 0 else random.choice([0,1])
            p_attack = self.dna[last]
            if np.random.rand() < p_attack:
                attacks.append(idx)
            else:
                partners.append(idx)
        return partners, attacks

    def mutate(self):
        new_dna = self.dna.copy()
        idx = np.random.randint(0, 2)
        new_dna[idx] += np.random.normal(0, 0.12)
        new_dna[idx] = np.clip(new_dna[idx], 0, 1)
        return Bot(self.richness, new_dna)

    @staticmethod
    def crossover(parent1, parent2):
        new_dna = np.zeros(2)
        for i in range(2):
            new_dna[i] = random.choice([parent1.dna[i], parent2.dna[i]])
        return Bot((parent1.richness + parent2.richness) / 2, new_dna)

# ---- GRID UTILITIES ----
def get_neighbors(x, y, grid):
    size = grid.shape[0]
    deltas = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]
    coords = []
    for dx, dy in deltas:
        nx = (x + dx) % size
        ny = (y + dy) % size
        coords.append((nx, ny))
    return coords

def init_grid(size):
    return np.array([[Bot() for _ in range(size)] for _ in range(size)])

def count_alive(grid):
    return sum(bot.alive for row in grid for bot in row)

# ---- FITNESS FUNCTIONS ----
def compute_fitness(grid, objective):
    alive_bots = [bot for row in grid for bot in row if bot.alive]
    richness = [bot.richness for bot in alive_bots]
    if objective == "Stability":
        return len(alive_bots)
    elif objective == "Equal richness":
        return -np.std(richness) if richness else 0
    elif objective == "One richest bot":
        return max(richness) if richness else 0
    else:
        return 0

# ---- GAME STEP ----
def step(grid, round_num):
    size = grid.shape[0]
    attacks_map = [[[] for _ in range(size)] for _ in range(size)]
    partners_map = [[[] for _ in range(size)] for _ in range(size)]
    death_queue = []
    attack_cost_map = [[0 for _ in range(size)] for _ in range(size)]
    actions_by_bot = dict()  # (x, y) -> list of 8 ("attack" or "partner") per neighbor

    # 1. Each bot decides actions for each neighbor
    for x in range(size):
        for y in range(size):
            bot = grid[x][y]
            if not bot.alive:
                continue
            neighbor_coords = get_neighbors(x, y, grid)
            neighbors = [grid[nx][ny] for nx, ny in neighbor_coords]
            partners_idx, attacks_idx = bot.decide(neighbors, round_num)
            action_list = ["none"] * 8
            for idx in partners_idx:
                nx, ny = neighbor_coords[idx]
                partners_map[nx][ny].append((x, y, idx))
                action_list[idx] = "partner"
            for idx in attacks_idx:
                nx, ny = neighbor_coords[idx]
                attacks_map[nx][ny].append((x, y, idx))
                attack_cost_map[x][y] += ATTACK_COST
                action_list[idx] = "attack"
            actions_by_bot[(x, y)] = action_list

    # 2. Deduct attack costs
    for x in range(size):
        for y in range(size):
            bot = grid[x][y]
            if not bot.alive:
                continue
            cost = attack_cost_map[x][y]
            bot.richness -= cost
            if bot.richness < 0:
                bot.richness = 0

    # 3. Bots attacked by more neighbors than they have partners die
    for x in range(size):
        for y in range(size):
            bot = grid[x][y]
            if not bot.alive:
                continue
            attackers = attacks_map[x][y]
            partners = partners_map[x][y]
            if len(attackers) > len(partners):
                bot.alive = False
                death_queue.append((x, y, bot.richness, attackers))

    # 4. Distribute richness of dead bots among attackers
    for x, y, richness, attackers in death_queue:
        if attackers:
            share = richness / len(attackers)
            for ax, ay, idx in attackers:
                if grid[ax][ay].alive:
                    grid[ax][ay].richness += share
        grid[x][y].richness = 0

    # 5. After all steps, kill bots that ran out of richness
    for x in range(size):
        for y in range(size):
            bot = grid[x][y]
            if bot.alive and bot.richness <= 0:
                bot.alive = False

    # 6. Update memory: What did each neighbor do to this bot?
    for x in range(size):
        for y in range(size):
            bot = grid[x][y]
            if not bot.alive:
                continue
            neighbor_coords = get_neighbors(x, y, grid)
            for idx, (nx, ny) in enumerate(neighbor_coords):
                if not grid[nx][ny].alive:
                    bot.memory[idx] = random.choice([0,1])  # treat dead as random action
                    continue
                their_actions = actions_by_bot.get((nx, ny), ["none"] * 8)
                mirror_idx = (idx + 4) % 8
                action = their_actions[mirror_idx]
                if action == "partner":
                    bot.memory[idx] = 0
                elif action == "attack":
                    bot.memory[idx] = 1
                else:
                    # Should not happen, but fallback to random
                    bot.memory[idx] = random.choice([0,1])
    return grid

# ---- GENETIC ALGORITHM ----
def evolve(grid, objective):
    size = grid.shape[0]
    new_grid = np.copy(grid)
    survivors = [(x, y) for x in range(size) for y in range(size) if grid[x][y].alive]
    dead = [(x, y) for x in range(size) for y in range(size) if not grid[x][y].alive]

    if not survivors:
        return init_grid(size)

    survivor_bots = [grid[x][y] for x, y in survivors]
    fitness_scores = [bot.richness for bot in survivor_bots]
    top_n = max(1, len(survivor_bots) // 5)
    sorted_survivors = [
        bot for score, bot in sorted(
            zip(fitness_scores, survivor_bots),
            key=lambda pair: pair[0],
            reverse=True
        )
    ]
    parents = sorted_survivors[:top_n]

    for x, y in dead:
        if len(parents) >= 2:
            p1, p2 = random.sample(parents, 2)
            child = Bot.crossover(p1, p2)
            if random.random() < 0.6:
                child = child.mutate()
        else:
            child = parents[0].mutate()
        child.richness = INIT_RICHNESS
        child.alive = True
        child.memory = {i: random.choice([0,1]) for i in range(8)}
        new_grid[x][y] = child

    # Survivors keep their alive status, memory, and current richness

    return new_grid

# ---- VISUALIZATION ----
def plot_grid(grid):
    size = grid.shape[0]
    img = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            bot = grid[x][y]
            img[x, y] = bot.richness if bot.alive else 0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap="viridis", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Grid: Color = Richness (0 = dead)")
    return fig

def plot_fitness(fitness_history, label):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(fitness_history, label=label)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend()
    ax.set_title("Objective Progress")
    return fig

def plot_histogram(grid):
    richness = [bot.richness for row in grid for bot in row if bot.alive]
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.hist(richness, bins=10, color="goldenrod", edgecolor="k")
    ax.set_title("Alive Bots Richness Histogram")
    return fig

# ---- STREAMLIT STATE ----
if "grid" not in st.session_state or st.session_state.grid.shape[0] != GRID_SIZE:
    st.session_state.grid = init_grid(GRID_SIZE)
    st.session_state.fitness_history = []
    st.session_state.round_num = 0

if st.button("Reset grid"):
    st.session_state.grid = init_grid(GRID_SIZE)
    st.session_state.fitness_history = []
    st.session_state.round_num = 0

if st.button("Step (1 gen)"):
    st.session_state.grid = step(st.session_state.grid, st.session_state.round_num)
    st.session_state.grid = evolve(st.session_state.grid, OBJECTIVE)
    st.session_state.round_num += 1

if st.button("Run (N generations)"):
    for _ in range(GENS_PER_RUN):
        st.session_state.grid = step(st.session_state.grid, st.session_state.round_num)
        st.session_state.grid = evolve(st.session_state.grid, OBJECTIVE)
        st.session_state.round_num += 1

# ---- FITNESS TRACKING ----
current_fitness = compute_fitness(st.session_state.grid, OBJECTIVE)
if "fitness_history" not in st.session_state:
    st.session_state.fitness_history = []
st.session_state.fitness_history.append(current_fitness)

# ---- DISPLAY ----
st.pyplot(plot_grid(st.session_state.grid))
st.pyplot(plot_fitness(st.session_state.fitness_history, OBJECTIVE))
st.pyplot(plot_histogram(st.session_state.grid))

with st.expander("See key statistics"):
    alive = count_alive(st.session_state.grid)
    total = GRID_SIZE * GRID_SIZE
    richness = [bot.richness for row in st.session_state.grid for bot in row if bot.alive]
    st.write(f"Alive bots: {alive}/{total}")
    st.write(f"Mean richness: {np.mean(richness) if richness else 0:.2f}")
    st.write(f"Richest bot: {max(richness) if richness else 0:.2f}")
    st.write(f"Richness stddev: {np.std(richness) if richness else 0:.2f}")
    st.write(f"Last fitness: {current_fitness:.2f}")
    st.write(f"Generation: {st.session_state.round_num}")

st.markdown(
    """
    **Grid color:**  
    - Dark = low richness  
    - Bright = high richness  
    - Black = dead  
    """
)
