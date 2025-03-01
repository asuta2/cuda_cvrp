# -------- Start of the importing part -----------
from numba import cuda, jit, int32, float32, int64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32
from math import pow, hypot, ceil
import numpy as np
import time
import random
from timeit import default_timer as timer
import sys
from google.colab import drive
np.set_printoptions(threshold=sys.maxsize)
# -------- End of the importing part -----------

# ------------------------- Start reading the data file -------------------------------------------
class vrp():
    def __init__(self, capacity=0, opt=0):
        self.capacity = capacity
        self.opt = opt
        self.nodes = np.zeros((1,4), dtype=np.float32)

    def addNode(self, label, demand, posX, posY):
        newrow = np.array([label, demand, posX, posY], dtype=np.float32)
        self.nodes = np.vstack((self.nodes, newrow))


def readInput():
    # Mount Google Drive
    drive.mount('/content/drive')

    # Create VRP object:
    vrpManager = vrp()

    # Define the direct file path
    dataset_path = "/content/drive/My Drive/test_set/P/P-n55-k15.vrp"

    # Read the VRP file
    print(f'Reading data file from {dataset_path}...', end=' ')
    with open(dataset_path, "r") as fo:
        lines = fo.readlines()

    for i, line in enumerate(lines):
        while line.upper().startswith('COMMENT'):
            inputs = line.split()
            if inputs[-1][:-1].isnumeric():
                vrpManager.opt = np.int32(inputs[-1][:-1])
            else:
                try:
                    vrpManager.opt = float(inputs[-1][:-1])
                except:
                    print('\nNo optimal value detected, taking optimal as 0.0')
                    vrpManager.opt = 0.0
            break

        # Validating positive non-zero capacity
        if vrpManager.opt < 0:
            print("Invalid input: optimal value can't be negative!", file=sys.stderr)
            exit(1)

        while line.upper().startswith('CAPACITY'):
            inputs = line.split()
            vrpManager.capacity = np.float32(inputs[2])
            # Validating positive non-zero capacity
            if vrpManager.capacity <= 0:
                print('Invalid input: capacity must be neither negative nor zero!', file=sys.stderr)
                exit(1)
            break

        while line.upper().startswith('NODE_COORD_SECTION'):
            i += 1
            line = lines[i]
            while not (line.upper().startswith('DEMAND_SECTION') or line == '\n'):
                inputs = line.split()
                vrpManager.addNode(np.int16(inputs[0]), 0.0, np.float32(inputs[1]), np.float32((inputs[2])))
                i += 1
                line = lines[i]
                while (line == '\n'):
                    i += 1
                    line = lines[i]
                    if line.upper().startswith('DEMAND_SECTION'):
                        break
                if line.upper().startswith('DEMAND_SECTION'):
                    i += 1
                    line = lines[i]
                    while not (line.upper().startswith('DEPOT_SECTION')):
                        inputs = line.split()
                        # Validating demand not greater than capacity
                        if float(inputs[1]) > vrpManager.capacity:
                            print('Invalid input: the demand of the node %s is greater than the vehicle capacity!' % vrpManager.nodes[0], file=sys.stderr)
                            exit(1)
                        if float(inputs[1]) < 0:
                            print('Invalid input: the demand of the node %s cannot be negative!' % vrpManager.nodes[0], file=sys.stderr)
                            exit(1)
                        vrpManager.nodes[int(inputs[0])][1] = float(inputs[1])
                        i += 1
                        line = lines[i]
                        while (line == '\n'):
                            i += 1
                            line = lines[i]
                            if line.upper().startswith('DEPOT_SECTION'):
                                break
                        if line.upper().startswith('DEPOT_SECTION'):
                            vrpManager.nodes = np.delete(vrpManager.nodes, 0, 0)
                            print('Done.')
                            return (vrpManager.capacity, vrpManager.nodes, vrpManager.opt)
# ------------------------- End of reading the input data file ------------------------------------

# ------------------------- Start calculating the cost table --------------------------------------
def calc_cost(data_d, cost_table_d):
    for row in range(data_d.shape[0]):
        for col in range(data_d.shape[0]):
            cost_table_d[row, col] = round(hypot(data_d[row, 2] - data_d[col, 2], data_d[row, 3] - data_d[col, 3]))
# ------------------------- End calculating the cost table ----------------------------------------

# ------------------------- Start fitness calculation ---------------------------------------------
def fitness(cost_table_d, pop, fitness_val_d):
    for row in range(pop.shape[0]):
        fitness_val_d[row, 0] = 0
        pop[row, -1] = 1

        for i in range(pop.shape[1] - 2):
            fitness_val_d[row, 0] += cost_table_d[pop[row, i] - 1, pop[row, i + 1] - 1]
        pop[row, -1] = fitness_val_d[row, 0]

# ------------------------- End fitness calculation ---------------------------------------------

# ------------------------- Start adjusting individuals ---------------------------------------------
def find_duplicates(pop, r_flag):
    for row in range(pop.shape[0]):
        # Detect duplicate nodes:
        for i in range(2, pop.shape[1] - 1):
            for j in range(i, pop.shape[1] - 1):
                if pop[row, i] != r_flag and pop[row, j] == pop[row, i] and i != j:
                    pop[row, j] = r_flag

def shift_r_flag(r_flag, pop):
    for row in range(pop.shape[0]):
        # Shift all r_flag values to the end of the list:
        for i in range(2, pop.shape[1]-2):
            if pop[row,i] == r_flag:
                k = i
                while pop[row,k] == r_flag:
                    k += 1
                if k < pop.shape[1]-1:
                    pop[row,i], pop[row,k] = pop[row,k], pop[row,i]

def find_missing_nodes(data, missing, pop):
    for row in range(pop.shape[0]):
        missing[row, 15] = 0  # The column at threadId_col == 15 is set to 0 (hardcoded)

        # Find missing nodes in the solutions:
        for i in range(1, data.shape[0]):
            for j in range(2, pop.shape[1] - 1):
                if data[i, 0] == pop[row, j]:
                    missing[row, i] = 0
                    break
                else:
                    missing[row, i] = data[i, 0]

def add_missing_nodes(r_flag, missing_d, pop):
    for row in range(pop.shape[0]):
        # Add the missing nodes to the solution:
        for k in range(missing_d.shape[1]):
            for l in range(2, pop.shape[1]-1):
                if missing_d[row, k] != 0 and pop[row, l] == r_flag:
                    pop[row, l] = missing_d[row, k]
                    break

def cap_adjust(r_flag, vrp_capacity, data_d, pop):
    for row in range(pop.shape[0]):
        reqcap = 0.0        # required capacity

        # Accumulate capacity:
        i = 1
        while pop[row, i] != r_flag:
            i += 1
            if pop[row,i] == r_flag:
                break

            if pop[row, i] != 1:
                reqcap += data_d[pop[row, i]-1, 1] # index starts from 0 while individuals start from 1
                if reqcap > vrp_capacity:
                    reqcap = 0
                    # Insert '1' and shift right:
                    new_val = 1
                    rep_val = pop[row, i]
                    for j in range(i, pop.shape[1] - 2):
                        pop[row, j] = new_val
                        new_val = rep_val
                        rep_val = pop[row, j + 1]
            else:
                reqcap = 0.0

def cleanup_r_flag(r_flag, pop):
    for row in range(pop.shape[0]):
        for col in range(pop.shape[1]):
            if pop[row, col] == r_flag:
                pop[row, col] = 1

# ------------------------- End adjusting individuals ---------------------------------------------

# ------------------------- Start initializing individuals ----------------------------------------
def initialize_pop(data_d, pop_d):
    # Iterate over each row in pop_d
    for row in range(pop_d.shape[0]):
    # Generate the individuals from the nodes in data_d:
        for col in range(data_d.shape[0] + 1):
            pop_d[row, col] = data_d[col - 1, 0]  # Index shift for data_d

        # Set initial values for the first two columns in pop_d
        pop_d[row, 0], pop_d[row, 1] = 1, 1

# ------------------------- End initializing individuals ------------------------------------------

# ------------------------- Start two-opt calculations --------------------------------------------
def reset_to_ones(pop):
    for row in range(pop.shape[0]):
        for col in range(pop.shape[1]):
            pop[row, col] = 1

def two_opt(pop, cost_table, candid_d_3):
    for row in range(pop.shape[0]):
        for col in range(pop.shape[1]):
            if col + 2 < pop.shape[1] :
                # Divide solution into routes:
                if pop[row, col] == 1 and pop[row, col + 1] != 1 and pop[row, col + 2] != 1:
                    route_length = 1
                    while pop[row, col + route_length] != 1 and col + route_length < pop.shape[1]:
                        candid_d_3[row, col + route_length] = pop[row, col + route_length]
                        route_length += 1

                    # Now we have candid_d_3 has the routes to be optimized for every row solution
                    total_cost = 0
                    min_cost = 0

                    for i in range(0, route_length):
                        min_cost += \
                            cost_table[candid_d_3[row,col + i] - 1, candid_d_3[row,col + i + 1] - 1]

                    # ------- The two opt algorithm --------

                    # So far, the best route is the given one (in candid_d_3)
                    improved = True
                    while improved:
                        improved = False
                        for i in range(1, route_length - 1):
                                # swap every two pairs
                                candid_d_3[row, col + i], candid_d_3[row, col + i + 1] = \
                                candid_d_3[row, col + i + 1], candid_d_3[row, col + i]

                                for j in range(0, route_length):
                                    total_cost += cost_table[candid_d_3[row,col + j] - 1,\
                                                candid_d_3[row,col + j + 1] - 1]

                                if total_cost < min_cost:
                                    min_cost = total_cost
                                    improved = True
                                else:
                                    candid_d_3[row, col + i + 1], candid_d_3[row, col + i]=\
                                    candid_d_3[row, col + i], candid_d_3[row, col + i + 1]

                    for k in range(0, route_length):
                        pop[row, col + k] = candid_d_3[row, col + k]
# ------------------------- End two-opt calculations --------------------------------------------

# ------------------------- Start evolution process ---------------------------------------------
# --------------------------------- Cross Over part ---------------------------------------------
def select_candidates(pop_d, random_arr_d, candid_d_1, candid_d_2, candid_d_3, candid_d_4, assign_child_1):
    for row in range(pop_d.shape[0]):
        for col in range(pop_d.shape[1]):
            if assign_child_1:
                # First individual in pop_d must be selected:
                candid_d_1[row, col] = pop_d[0, col]
                candid_d_2[row, col] = pop_d[random_arr_d[row, 1], col]
                candid_d_3[row, col] = pop_d[random_arr_d[row, 2], col]
                candid_d_4[row, col] = pop_d[random_arr_d[row, 3], col]
            else:
                #Create a pool of 4 randomly selected individuals:
                candid_d_1[row, col] = pop_d[random_arr_d[row, 0], col]
                candid_d_2[row, col] = pop_d[random_arr_d[row, 1], col]
                candid_d_3[row, col] = pop_d[random_arr_d[row, 2], col]
                candid_d_4[row, col] = pop_d[random_arr_d[row, 3], col]

def select_parents(pop_d, candid_d_1, candid_d_2, candid_d_3, candid_d_4, parent_d_1, parent_d_2):
    for row in range(pop_d.shape[0]):
        for col in range(pop_d.shape[1]):
        # Selecting 2 parents with binary tournament
        # ----------------------------1st Parent--------------------------------------------------
            if candid_d_1[row, -1] < candid_d_2[row, -1]:
                parent_d_1[row, col] = candid_d_1[row, col]
            else:
                parent_d_1[row, col] = candid_d_2[row, col]

            # ----------------------------2nd Parent--------------------------------------------------
            if candid_d_3[row, -1] < candid_d_4[row, -1]:
                parent_d_2[row, col] = candid_d_3[row, col]
            else:
                parent_d_2[row, col] = candid_d_4[row, col]

def number_cut_points(candid_d_1, candid_d_2, candid_d_3, candid_d_4, parent_d_1, parent_d_2, count, min_n, max_n):
    for row in range(candid_d_1.shape[0]):
        for col in range(candid_d_1.shape[1]):
            candid_d_1[row, col] = 1
            candid_d_2[row, col] = 1
            candid_d_3[row, col] = 1
            candid_d_4[row, col] = 1

        # Calculate the actual length of parents
        for i in range(0, candid_d_1.shape[1]-2):
            if not (parent_d_1[row, i] == 1 and parent_d_1[row, i+1] == 1):
                candid_d_1[row, 2] += 1

            if not (parent_d_2[row, i] == 1 and parent_d_2[row, i+1] == 1):
                candid_d_2[row, 2] += 1

        # Minimum length of the two parents
        candid_d_1[row, 3] = \
        min(candid_d_1[row, 2], candid_d_2[row, 2])

        # Number of cutting points = (n/5 - 2)
        n_points = max(min_n, (count % (max_n * 4000)) // 4000) # the n_points increases one every 5000 iterations till 20 then resets to 2 and so on
        candid_d_1[row, 4] = n_points

def add_cut_points(candid_d_1, candid_d_2):

    # Iterate over each row in the candid_d_1 array
    for row in range(candid_d_1.shape[0]):
        no_cuts = candid_d_1[row, 4]

        # Generate unique random numbers as cut indices
        for i in range(1, no_cuts + 1):
            rnd_val = 0

            for j in range(1, no_cuts + 1):
                # Ensure the random value is not 0 and not equal to any previous cut points
                while rnd_val == 0 or rnd_val == candid_d_2[row, j]:
                    rnd = random.random() * (candid_d_1[row, 3] - 2) + 2
                    rnd_val = int(rnd) + 2

            # Store the unique random cut value
            candid_d_2[row, i + 1] = rnd_val

        # Sorting the crossover points
        for i in range(2, no_cuts + 2):
            min_index = i

            for j in range(i + 1, no_cuts + 2):
                if candid_d_2[row, j] < candid_d_2[row, min_index]:
                    min_index = j

            candid_d_2[row, min_index], candid_d_2[row, i] = candid_d_2[row, i], candid_d_2[row, min_index]

def cross_over(candid_d_1, candid_d_2, child_d_1, child_d_2, parent_d_1, parent_d_2):
    for row in range(candid_d_1.shape[0]):
        for col in range(candid_d_1.shape[1]):
            if col > 1:
                child_d_1[row, col] = parent_d_1[row, col]
                child_d_2[row, col] = parent_d_2[row, col]

                # Perform the crossover:
                no_cuts = candid_d_1[row, 4]
                if col < candid_d_2[row, 2]: # Swap from first element to first cut point
                    child_d_1[row, col], child_d_2[row, col] =\
                    child_d_2[row, col], child_d_1[row, col]

                if no_cuts%2 == 0: # For even number of cuts, swap from the last cut point to the end
                    if col > candid_d_2[row, no_cuts+1] and col < child_d_1.shape[1]-1:
                        child_d_1[row, col], child_d_2[row, col] =\
                        child_d_2[row, col], child_d_1[row, col]

                for j in range(2, no_cuts+1):
                    cut_idx = candid_d_2[row, j]
                    if no_cuts%2 == 0:
                        if j%2==1 and col >= cut_idx and col < candid_d_2[row, j+1]:
                            child_d_1[row, col], child_d_2[row, col] =\
                            child_d_2[row, col], child_d_1[row, col]

                    elif no_cuts%2 == 1:
                        if j%2==1 and col>=cut_idx and col < candid_d_2[row, j+1]:
                            child_d_1[row, col], child_d_2[row, col] =\
                            child_d_2[row, col], child_d_1[row, col]

# ------------------------------------Mutation part -----------------------------------------------
def mutate(child_d_1, child_d_2):

    for row in range(child_d_1.shape[0]):
    # Swap two positions in the children, with 1:40 probability
            mutation_prob = 15

            rnd = np.random.uniform(0, 1) * (mutation_prob - 1) + 1  # random*(max-min)+min
            rnd_val = int(rnd) + 2

            if rnd_val == 1:
                i1 = 1

                # Repeat random selection if depot was selected:
                while child_d_1[row, i1] == 1:
                    rnd = np.random.uniform(0, 1) * (child_1.shape[1] - 4) + 2  # random*(max-min)+min
                    i1 = int(rnd)+2

                i2 = 1
                while child_d_1[row, i2] == 1:
                    rnd = np.random.uniform(0, 1) * (child_1.shape[1] - 4) + 2  # random*(max-min)+min
                    i2 = int(rnd) + 2

                child_d_1[row, i1], child_d_1[row, i2] = \
                child_d_1[row, i2], child_d_1[row, i1]

            # Repeat for the second child:
                i1 = 1
                # Repeat random selection if depot was selected:
                while child_d_2[row, i1] == 1:
                    rnd = np.random.uniform(0, 1) * (child_2.shape[1] - 4) + 2  # random*(max-min)+min
                    i1 = int(rnd) + 2

                i2 = 1
                while child_d_2[row, i2] == 1:
                    rnd = np.random.uniform(0, 1) * (child_2.shape[1] - 4) + 2  # random*(max-min)+min
                    i2 = int(rnd)+2

                child_d_2[row, i1], child_d_1[row, i2] = \
                child_d_2[row, i2], child_d_1[row, i1]

# -------------------------- Update population part -----------------------------------------------
def select_individual(index, pop_d, individual):
  # Update the population with the individual at the specified index
    for col in range(pop_d.shape[1]):
        pop_d[index, col] = individual[index, col]

def update_pop(count, parent_d_1, parent_d_2, child_d_1, child_d_2, pop_d):
    for row in range(pop_d.shape[0]):
        for col in range(pop_d.shape[1]):
            if child_d_1[row, -1] <= parent_d_1[row, -1] and \
            child_d_1[row, -1] <= parent_d_2[row, -1] and \
            child_d_1[row, -1] <= child_d_2[row, -1]:

                pop_d[row, col] = child_d_1[row, col]
                pop_d[row, 0] = count

            elif child_d_2[row, -1] <= parent_d_1[row, -1] and \
            child_d_2[row, -1] <= parent_d_2[row, -1] and \
            child_d_2[row, -1] <= child_d_1[row, -1]:

                pop_d[row, col] = child_d_2[row, col]
                pop_d[row, 0] = count

            elif parent_d_1[row, -1] <= parent_d_2[row, -1] and \
            parent_d_1[row, -1] <= child_d_1[row, -1] and \
            parent_d_1[row, -1] <= child_d_2[row, -1]:

                pop_d[row, col] = parent_d_1[row, col]
                pop_d[row, 0] = count

            elif parent_d_2[row, -1] <= parent_d_1[row, -1] and \
            parent_d_2[row, -1] <= child_d_1[row, -1] and \
            parent_d_2[row, -1] <= child_d_2[row, -1]:

                pop_d[row, col] = parent_d_2[row, col]
                pop_d[row, 0] = count

# ------------------------- Definition of CPU functions ----------------------------------------------
def select_bests(parent_d_1, parent_d_2, child_d_1, child_d_2, pop_d, popsize):
    # Select the best 5% from paernt 1 & parent 2:
    num_to_select = int(0.05 * popsize)  # Ensure this is an integer
    pool = parent_d_1[parent_d_1[:,-1].argsort()][:num_to_select,:]
    pool = np.concatenate((pool, parent_2[parent_2[:, -1].argsort()][:num_to_select, :]))
    pool = pool[pool[:,-1].argsort()]

    # Sort child 1 & child 2:
    child_d_1 = child_d_1[child_d_1[:,-1].argsort()]
    child_d_2 = child_d_2[child_d_2[:,-1].argsort()]

   # Ensure the indices for slicing are integers:
    num_to_select_1 = int(0.05 * popsize)
    num_to_select_2 = int(0.48 * popsize)
    num_to_select_3 = int(0.47 * popsize)

    pop_d[:num_to_select_1, :] = pool[:num_to_select_1, :]
    pop_d[num_to_select_1:num_to_select_1 + num_to_select_2, :] = child_d_1[:num_to_select_2, :]
    pop_d[num_to_select_1 + num_to_select_2:, :] = child_d_2[:num_to_select_3, :]
# ------------------------- Start Main ------------------------------------------------------------
vrp_capacity, data, opt = readInput()
popsize = 100
min_n = 2 # Maximum number of crossover points
max_n = 2 # Maximum number of crossover points

try:
    generations = int(sys.argv[2])
except:
    print('No generation number provided, taking 500 generations...')
    generations = 20

r_flag = 9999 # A flag for removal/replacement

# Initialize population and auxiliary structures
pop = np.ones((popsize, int(2 * data.shape[0]) + 2), dtype=np.int32)
missing = np.zeros(shape=(popsize, pop.shape[1]), dtype=np.int32)

fitness_val = np.zeros(shape=(popsize, 1), dtype=np.int32)

cost_table = np.zeros((data.shape[0], data.shape[0]), dtype=np.int32)

# --------------Calculate the cost table----------------------------------------------
calc_cost(data, cost_table)

# --------------Initialize population----------------------------------------------
initialize_pop(data, pop)

for individual in pop:
    np.random.shuffle(individual[2:-1])

find_duplicates(pop, r_flag)
find_missing_nodes(data, missing, pop)
add_missing_nodes(r_flag, missing, pop)
shift_r_flag(r_flag, pop)
cap_adjust(r_flag, vrp_capacity, data, pop)
cleanup_r_flag(r_flag, pop)

# --------------Calculate fitness----------------------------------------------
fitness(cost_table, pop, fitness_val)

pop = pop[pop[:,-1].argsort()] # Sort the population to get the best later

# ------------------------- End Main ------------------------------------------------------------

# --------------Evolve population for some generations----------------------------------------------
# Create the pool of 6 arrays of the same length
candid_1 = np.ones((popsize, pop.shape[1]), dtype=np.int32)
candid_2 = np.ones((popsize, pop.shape[1]), dtype=np.int32)
candid_3 = np.ones((popsize, pop.shape[1]), dtype=np.int32)
candid_4 = np.ones((popsize, pop.shape[1]), dtype=np.int32)

parent_1 = np.ones((popsize, pop.shape[1]), dtype=np.int32)
parent_2 = np.ones((popsize, pop.shape[1]), dtype=np.int32)

child_1 = np.ones((popsize, pop.shape[1]), dtype=np.int32)
child_2 = np.ones((popsize, pop.shape[1]), dtype=np.int32)

cut_idx = np.ones(shape=(pop.shape[1]), dtype=np.int32)

minimum_cost = float('Inf')
old_time = timer()
count = 0
count_index = 0
best_sol = 0
assign_child_1 = False
last_shuffle = 10000

while count <= generations:
    if minimum_cost <= opt:
        break

    random_arr = np.arange(popsize, dtype=np.int32).reshape((popsize,1))
    random_arr = np.repeat(random_arr, 4, axis=1)

    random.shuffle(random_arr[:,0])
    random.shuffle(random_arr[:,1])
    random.shuffle(random_arr[:,2])
    random.shuffle(random_arr[:,3])

    select_candidates(pop, random_arr, candid_1, candid_2, candid_3, candid_4, assign_child_1)
    select_parents(pop, candid_1, candid_2, candid_3, candid_4, parent_1, parent_2)

    number_cut_points(candid_1, candid_2, candid_3, candid_4, parent_1, parent_2, count, min_n, max_n)
    add_cut_points(candid_1, candid_2)

    cross_over(candid_1, candid_2, child_1, child_2, parent_1, parent_2)
    # Performing mutation
    mutate(child_1, child_2)

    # Adjusting child_1 array
    find_duplicates(child_1, r_flag)
    find_missing_nodes(data, missing, child_1)
    add_missing_nodes(r_flag, missing, child_1)
    shift_r_flag(r_flag, child_1)
    cap_adjust(r_flag, vrp_capacity, data, child_1)
    cleanup_r_flag(r_flag, child_1)

    # Adjusting child_2 array
    find_duplicates(child_2, r_flag)
    find_missing_nodes(data, missing, child_2)
    add_missing_nodes(r_flag, missing, child_2)
    shift_r_flag(r_flag, child_2)
    cap_adjust(r_flag, vrp_capacity, data, child_2)
    cleanup_r_flag(r_flag, child_2)

    # --------------------------------------------------------------------------
    # Performing the two-opt optimization and Calculating fitness for child_1 array
    reset_to_ones(candid_3)
    two_opt(child_1, cost_table, candid_3)
    fitness(cost_table, child_1, fitness_val)

    # --------------------------------------------------------------------------
    # Performing the two-opt optimization and Calculating fitness for child_2 array
    reset_to_ones(candid_3)
    two_opt(child_2, cost_table, candid_3)
    fitness(cost_table, child_2, fitness_val)

    # --------------------------------------------------------------------------
    # Creating the new population from parents and children
    select_bests(parent_1, parent_2, child_1, child_2, pop, popsize)

    # --------------------------------------------------------------------------
    # Replacing duplicates with random individuals from child_1
    asnumpy_pop = np.array(pop) # Convert pop_d to numpy array
    asnumpy_child_1 = np.array(child_1) # Convert child_d_1 to numpy array
    repeats = 0

    x = np.unique(asnumpy_pop[:,1:], axis=0)
    while x.shape[0] < popsize:
        if repeats >= popsize-1:
            break
        rndm = random.randint(0, popsize-1)
        x = np.append(x, [asnumpy_child_1[rndm,1:]], axis=0)
        x = np.unique(x, axis=0)
        repeats += 1

    # --------------------------------------------------------------------------
    # Replacing duplicates with random individuals from parent_2
    asnumpy_parent_2 = np.array(parent_2) # Convert parent_d_2 to numpy array
    repeats = 0
    while x.shape[0] < popsize:
        if repeats >= popsize-1:
            break
        rndm = random.randint(0, popsize-1)
        x = np.append(x, [asnumpy_parent_2[rndm,1:]], axis=0)
        repeats += 1

    # --------------------------------------------------------------------------
    x = np.insert(x, 0, count, axis=1)
    pop = np.array(x)

    # --------------------------------------------------------------------------
    # Picking best solution
    old_cost = minimum_cost
    best_sol = pop[pop[:,-1].argmin()]
    minimum_cost = best_sol[-1]

    worst_sol = pop[pop[:,-1].argmax()]
    worst_cost = worst_sol[-1]

    delta = worst_cost-minimum_cost
    average = np.mean(pop[:,-1])

    if minimum_cost == old_cost: # To calculate for how long the quality did not improve
        count_index += 1
    else:
        count_index = 0

    # Shuffle the population after a certain number of generations without improvement
    assign_child_1 = False

    if count == 1:
        print('At first generation, Best: %d,'%minimum_cost, 'Worst: %d'%worst_cost, \
            'delta: %d'%delta, 'Avg: %.2f'%average)
    elif (count+1)%100 == 0:
        print('After %d generations, Best: %d,'%(count+1, minimum_cost), 'Worst: %d'%worst_cost, \
            'delta: %d'%delta, 'Avg: %.2f'%average)
    count += 1

current_time = timer()
total_time = float('{0:.4f}'.format((current_time - old_time)))
time_per_loop = float('{0:.4f}'.format((current_time - old_time)/(count-1)))

best_sol = np.subtract(best_sol, np.ones_like(best_sol))
best_sol[0] = best_sol[0] + 1
best_sol[-1] = best_sol[-1] + 1

print('---------\nProblem:', sys.argv[1], ', Best known:', opt)
print('Time elapsed:', total_time, 'secs', 'Time per loop:', time_per_loop, 'secs', end = '\n---------\n')
print('Stopped at generation %d, Best cost: %d, from Generation: %d'\
      %(count-1, best_sol[-1], best_sol[0]), end = '\n---------\n')
print('Best solution:', best_sol, end = '\n---------\n')
lo = minimum_cost  # Best solution obtained
l_star = opt
percent_gap = ((lo - l_star) / l_star) * 100 if l_star != 0 else float('inf')  # Avoid division by zero
print(f"Percentage Gap (%Gap): {percent_gap:.2f}%")
