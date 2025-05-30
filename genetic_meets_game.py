import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# ---- PARAMETERS ----
st.title("Closed Society Bot Evolution - Automatic Periods Model")

GRID_SIZE = 30
OBJECTIVE = st.sidebar.selectbox("Objective", ["Stability", "Equal richness", "One richest bot"])
MAX_STEPS_PER_PERIOD = 1000
INIT_RICHNESS = 1000
ATTACK_COST = st.sidebar.slider("Attack cost: ", 0, 1000, 50)

if "period_stats" not in st.session_state:
    st.session_state.period_stats = []
if "run_state" not in st.session_state:
    st.session_state.run_state = "waiting"  # "running", "period_over"
if "autoplay" not in st.session_state:
    st.session_state.autoplay = False

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

def init_grid(size, inherited_dnas=None):
    grid = []
    k = 0
    for i in range(size):
        row = []
        for j in range(size):
            if inherited_dnas and k < len(inherited_dnas):
                row.append(Bot(dna=inherited_dnas[k]))
                k += 1
            else:
                row.append(Bot())
        grid.append(row)
    return np.array(grid)

def count_alive(grid):
    return sum(bot.alive for row in grid for bot in row)

def living_neighbors(x, y, grid):
    neighbors = get_neighbors(x, y, grid)
    count = 0
    for idx, (nx, ny) in enumerate(neighbors):
        if grid[nx][ny].alive:
            count += 1
    return count

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

def maximal_fitness(objective, grid_size):
    N = grid_size*grid_size
    if objective == "Stability":
        return N
    elif objective == "Equal richness":
        return 0  # zero stddev is perfect equality
    elif objective == "One richest bot":
        return INIT_RICHNESS*N
    return 1

def average_dna(grid):
    alive_bots = [bot for row in grid for bot in row if bot.alive]
    if not alive_bots:
        return [0.0, 0.0]
    dna_matrix = np.stack([bot.dna for bot in alive_bots])
    return list(np.mean(dna_matrix, axis=0))

# ---- GAME STEP ----
def step(grid, round_num):
    size = grid.shape[0]
    attacks_map = [[[] for _ in range(size)] for _ in range(size)]
    partners_map = [[[] for _ in range(size)] for _ in range(size)]
    death_queue = []
    attack_cost_map = [[0 for _ in range(size)] for _ in range(size)]
    actions_by_bot = dict()

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

    for x in range(size):
        for y in range(size):
            bot = grid[x][y]
            if not bot.alive:
                continue
            cost = attack_cost_map[x][y]
            bot.richness -= cost
            if bot.richness < 0:
                bot.richness = 0

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

    for x, y, richness, attackers in death_queue:
        if attackers:
            share = richness / len(attackers)
            for ax, ay, idx in attackers:
                if grid[ax][ay].alive:
                    grid[ax][ay].richness += share
        grid[x][y].richness = 0

    for x in range(size):
        for y in range(size):
            bot = grid[x][y]
            if bot.alive and bot.richness <= 0:
                bot.alive = False

    for x in range(size):
        for y in range(size):
            bot = grid[x][y]
            if not bot.alive:
                continue
            neighbor_coords = get_neighbors(x, y, grid)
            for idx, (nx, ny) in enumerate(neighbor_coords):
                if not grid[nx][ny].alive:
                    bot.memory[idx] = random.choice([0,1])
                    continue
                their_actions = actions_by_bot.get((nx, ny), ["none"] * 8)
                mirror_idx = (idx + 4) % 8
                action = their_actions[mirror_idx]
                if action == "partner":
                    bot.memory[idx] = 0
                elif action == "attack":
                    bot.memory[idx] = 1
                else:
                    bot.memory[idx] = random.choice([0,1])
    return grid

def period_over(grid):
    size = grid.shape[0]
    for x in range(size):
        for y in range(size):
            bot = grid[x][y]
            if not bot.alive:
                continue
            if living_neighbors(x, y, grid) > 0:
                return False
    return True

# --- Mixed-inheritance for DNA
def start_new_period(prev_grid=None):
    inherited_dnas = []
    if prev_grid is not None:
        survivors = [bot for row in prev_grid for bot in row if bot.alive]
        N = GRID_SIZE * GRID_SIZE
        n_survivor = int(0.8 * N)
        n_random = N - n_survivor
        if survivors:
            inherited_dnas = [bot.dna.copy() for bot in survivors]
            while len(inherited_dnas) < n_survivor:
                parent = random.choice(survivors)
                child = parent.mutate()
                inherited_dnas.append(child.dna)
        for _ in range(n_random):
            inherited_dnas.append(np.random.rand(2))
        random.shuffle(inherited_dnas)
    st.session_state.grid = init_grid(GRID_SIZE, inherited_dnas)
    st.session_state.round_num = 0
    st.session_state.period_num += 1

def run_period(max_steps=MAX_STEPS_PER_PERIOD):
    for i in range(max_steps):
        st.session_state.grid = step(st.session_state.grid, st.session_state.round_num)
        st.session_state.round_num += 1
        if period_over(st.session_state.grid):
            break
    # Compute stats
    duration = st.session_state.round_num
    final_fitness = compute_fitness(st.session_state.grid, OBJECTIVE)
    max_fit = maximal_fitness(OBJECTIVE, GRID_SIZE)
    avg_dna = average_dna(st.session_state.grid)
    if max_fit != 0:
        normalized_fitness = final_fitness / max_fit
    else:
        alive = [bot for row in st.session_state.grid for bot in row if bot.alive]
        stddev = np.std([bot.richness for bot in alive]) if alive else 0
        normalized_fitness = 1.0 - (stddev / INIT_RICHNESS)
    st.session_state.period_stats.append({
        "Period": st.session_state.period_num,
        "Duration": duration,
        "Normalized Fitness": round(normalized_fitness, 3),
        "Avg DNA - p_attack_if_partnered": round(avg_dna[0], 3),
        "Avg DNA - p_attack_if_attacked": round(avg_dna[1], 3),
    })

# ---- AUTOPLAY EVOLUTION ----
def run_evolution():
    for i in range(100):
        print("Period {}".format(i))
        start_new_period(prev_grid=st.session_state.grid)
        run_period()

def stop_evolution():
    st.session_state.autoplay = False

# ---- STREAMLIT STATE INIT ----
if "grid" not in st.session_state or st.session_state.grid.shape[0] != GRID_SIZE:
    st.session_state.grid = init_grid(GRID_SIZE)
    st.session_state.round_num = 0
    st.session_state.period_num = 1
    st.session_state.period_stats = []
    st.session_state.autoplay = False

# ---- UI ----
st.write(f"**Period {st.session_state.period_num}**")
st.write(f"Step {st.session_state.round_num}")

c1, c2, c3, c4 = st.columns([1,1,1,1])
if c1.button("Step (one turn)"):
    if not period_over(st.session_state.grid):
        st.session_state.grid = step(st.session_state.grid, st.session_state.round_num)
        st.session_state.round_num += 1

if c2.button("Run Period"):
    run_period()

if c3.button("Start New Period"):
    if not period_over(st.session_state.grid):
        run_period()
    start_new_period(prev_grid=st.session_state.grid)

if c4.button("Run 100 periods"):
    run_evolution()

# ---- DISPLAY ----
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
    ax.set_title("Grid: Color = Richness")
    return fig

# Compute stats for current grid
alive = count_alive(st.session_state.grid)
total = GRID_SIZE * GRID_SIZE
richness = [bot.richness for row in st.session_state.grid for bot in row if bot.alive]
avg_dna = average_dna(st.session_state.grid)
all_dna = np.stack([bot.dna for row in st.session_state.grid for bot in row if bot.alive]) if alive else np.zeros((1,2))
std_dna = np.std(all_dna, axis=0) if alive else np.zeros(2)

current_fitness = compute_fitness(st.session_state.grid, OBJECTIVE)
max_fit = maximal_fitness(OBJECTIVE, GRID_SIZE)
if max_fit != 0:
    normalized_fitness = current_fitness / max_fit
else:
    # For "Equal richness" (where max_fit is 0), treat 0 stddev as best (score=1)
    alive = [bot for row in st.session_state.grid for bot in row if bot.alive]
    stddev = np.std([bot.richness for bot in alive]) if alive else 0
    normalized_fitness = 1.0 - (stddev / INIT_RICHNESS)

summary1, summary2, summary3 = st.columns(3)
summary1.metric("Period", st.session_state.period_num)
summary2.metric("Current Fitness", f"{100*normalized_fitness:.0f}%")
summary3.markdown(
    f"**Avg DNA**<br/>"
    f"p_att_if_partnered: {avg_dna[0]:.3f} ±{std_dna[0]:.3f}<br/>"
    f"p_att_if_attacked: {avg_dna[1]:.3f} ±{std_dna[1]:.3f}",
    unsafe_allow_html=True
)

st.pyplot(plot_grid(st.session_state.grid))

alive = count_alive(st.session_state.grid)
total = GRID_SIZE * GRID_SIZE
richness = [bot.richness for row in st.session_state.grid for bot in row if bot.alive]
st.write(f"Alive bots: {alive}/{total}")
st.write(f"Mean richness: {np.mean(richness) if richness else 0:.2f}")
st.write(f"Richest bot: {max(richness) if richness else 0:.2f}")
st.write(f"Richness stddev: {np.std(richness) if richness else 0:.2f}")

if st.session_state.period_stats:
    st.write("### Period Summary")
    st.dataframe(st.session_state.period_stats)
    # Line chart for fitness over periods
    fitness_list = [p["Normalized Fitness"] for p in st.session_state.period_stats]
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(fitness_list, marker='o')
    ax.set_xlabel("Period")
    ax.set_ylabel("Normalized Fitness")
    ax.set_title("Fitness Evolution Over Periods")
    st.pyplot(fig)

    # Line charts for average DNA values over periods
    avg_dna_partnered = [p["Avg DNA - p_attack_if_partnered"] for p in st.session_state.period_stats]
    avg_dna_attacked = [p["Avg DNA - p_attack_if_attacked"] for p in st.session_state.period_stats]
    fig2, ax2 = plt.subplots(figsize=(5, 2))
    ax2.plot(avg_dna_partnered, label="p_attack_if_partnered", marker='o')
    ax2.plot(avg_dna_attacked, label="p_attack_if_attacked", marker='x')
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Average DNA (attack prob)")
    ax2.set_title("DNA Evolution Over Periods")
    ax2.legend()
    st.pyplot(fig2)

st.markdown(
    """
    **Grid color:**  
    - Dark = low richness  
    - Bright = high richness  
    - Black = dead  
    """
)
