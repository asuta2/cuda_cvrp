# -------- Start of the importing part -----------
from numba import cuda, jit, int32, float32, int64
from numba.core.errors import NumbaPerformanceWarning
import warnings
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32
import cupy as cp
from math import pow, hypot, ceil
from timeit import default_timer as timer
import numpy as np
import random
import sys
from google.colab import drive
np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
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
    dataset_path = "/content/drive/My Drive/test_set/M/M-n151-k12-v1.vrp"

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
@cuda.jit
def calc_cost_gpu(data_d, popsize, vrp_capacity, cost_table_d):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, data_d.shape[0], stride_x):
        for col in range(threadId_col, data_d.shape[0], stride_y):
            cost_table_d[row, col] = \
            round(hypot(data_d[row, 2] - data_d[col, 2], data_d[row, 3] - data_d[col, 3]))
# ------------------------- End calculating the cost table ----------------------------------------

# ------------------------- Start fitness calculation ---------------------------------------------
@cuda.jit
def fitness_gpu(cost_table_d, pop, fitness_val_d):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        fitness_val_d[row, 0] = 0
        pop[row, -1] = 1

        if threadId_col == 15:
            for i in range(pop.shape[1]-2):
                fitness_val_d[row, 0] += \
                cost_table_d[pop[row, i]-1, pop[row, i+1]-1]
            pop[row, -1] = fitness_val_d[row,0]

    cuda.syncthreads()

@cuda.jit
def fitness_gpu_new(cost_table_d, pop, fitness_val_d):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    if threadId_row < pop.shape[0]:
        fitness_val_d[threadId_row, 0] = 0
        pop[threadId_row, -1] = 1

        for col in range(threadId_col, pop.shape[1]-2, stride_y):
            if col > 0:
                cuda.atomic.add(fitness_val_d, (threadId_row, 0), cost_table_d[pop[threadId_row, col]-1, pop[threadId_row, col+1]-1])

        pop[threadId_row, -1] = fitness_val_d[threadId_row,0]

    cuda.syncthreads()
# ------------------------- End fitness calculation ---------------------------------------------

# ------------------------- Start adjusting individuals ---------------------------------------------
@cuda.jit
def find_duplicates(pop, r_flag):

    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        if threadId_col == 15:
            # Detect duplicate nodes:
            for i in range(2, pop.shape[1]-1):
                for j in range(i, pop.shape[1]-1):
                    if pop[row, i] != r_flag and pop[row, j] == pop[row, i] and i != j:
                        pop[row, j] = r_flag
@cuda.jit
def shift_r_flag(r_flag, vrp_capacity, data_d, pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        if threadId_col == 15:
            # Shift all r_flag values to the end of the list:
            for i in range(2, pop.shape[1]-2):
                if pop[row,i] == r_flag:
                    k = i
                    while pop[row,k] == r_flag:
                        k += 1
                    if k < pop.shape[1]-1:
                        pop[row,i], pop[row,k] = pop[row,k], pop[row,i]
@cuda.jit
def find_missing_nodes(r_flag, data_d, missing_d, pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        if threadId_col == 15:
            missing_d[row, threadId_col] = 0
            # Find missing nodes in the solutions:
            for i in range(1, data_d.shape[0]):
                for j in range(2, pop.shape[1]-1):
                    if data_d[i,0] == pop[row,j]:
                        missing_d[row, i] = 0
                        break
                    else:
                        missing_d[row, i] = data_d[i,0]

@cuda.jit
def add_missing_nodes(r_flag, data_d, missing_d, pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        if threadId_col == 15:
            # Add the missing nodes to the solution:
            for k in range(missing_d.shape[1]):
                for l in range(2, pop.shape[1]-1):
                    if missing_d[row, k] != 0 and pop[row, l] == r_flag:
                        pop[row, l] = missing_d[row, k]
                        break
@cuda.jit
def cap_adjust(r_flag, vrp_capacity, data_d, pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        if threadId_col == 15:
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
                        for j in range(i, pop.shape[1]-2):
                            pop[row, j] = new_val
                            new_val = rep_val
                            rep_val = pop[row, j+1]
                else:
                    reqcap = 0.0
    cuda.syncthreads()

@cuda.jit
def cleanup_r_flag(r_flag, pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        for col in range(threadId_col, pop.shape[1], stride_y):
            if pop[row, col] == r_flag:
                pop[row, col] = 1

    cuda.syncthreads()
# ------------------------- End adjusting individuals ---------------------------------------------

# ------------------------- Start initializing individuals ----------------------------------------
@cuda.jit
def initializePop_gpu(rng_states, data_d, missing_d, pop_d):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):
    # Generate the individuals from the nodes in data_d:
        for col in range(threadId_col, data_d.shape[0]+1, stride_y):
            pop_d[row, col] = data_d[col-1, 0]

        pop_d[row, 0], pop_d[row, 1] = 1, 1

# ------------------------- End initializing individuals ------------------------------------------

# ------------------------- Start two-opt calculations --------------------------------------------
@cuda.jit
def reset_to_ones(pop):

    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        for col in range(threadId_col, pop.shape[1], stride_y):
            pop[row, col] = 1
    cuda.syncthreads()

@cuda.jit
def two_opt(pop, cost_table, candid_d_3):

    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        for col in range(threadId_col, pop.shape[1], stride_y):
            # candid_d_3[row, col] = 1
            if col+2 < pop.shape[1] :
                # Divide solution into routes:
                if pop[row, col] == 1 and pop[row, col+1] != 1 and pop[row, col+2] != 1:
                    route_length = 1
                    while pop[row, col+route_length] != 1 and col+route_length < pop.shape[1]:
                        candid_d_3[row, col+route_length] = pop[row, col+route_length]
                        route_length += 1

                    # Now we have candid_d_3 has the routes to be optimized for every row solution
                    total_cost = 0
                    min_cost =0

                    for i in range(0, route_length):
                        min_cost += \
                            cost_table[candid_d_3[row,col+i]-1, candid_d_3[row,col+i+1]-1]

                    # ------- The two opt algorithm --------

                    # So far, the best route is the given one (in candid_d_3)
                    improved = True
                    while improved:
                        improved = False
                        for i in range(1, route_length-1):
                                # swap every two pairs
                                candid_d_3[row, col+i], candid_d_3[row, col+i+1] = \
                                candid_d_3[row, col+i+1], candid_d_3[row, col+i]

                                for j in range(0, route_length):
                                    total_cost += cost_table[candid_d_3[row,col+j]-1,\
                                                candid_d_3[row,col+j+1]-1]

                                if total_cost < min_cost:
                                    min_cost = total_cost
                                    improved = True
                                else:
                                    candid_d_3[row, col+i+1], candid_d_3[row, col+i]=\
                                    candid_d_3[row, col+i], candid_d_3[row, col+i+1]

                    for k in range(0, route_length):
                        pop[row, col+k] = candid_d_3[row, col+k]
# ------------------------- End two-opt calculations --------------------------------------------

# ------------------------- Start evolution process ---------------------------------------------
# --------------------------------- Cross Over part ---------------------------------------------
@cuda.jit
def select_candidates(pop_d, random_arr_d, candid_d_1, candid_d_2, candid_d_3, candid_d_4, assign_child_1):

    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):
        for col in range(threadId_col, pop_d.shape[1], stride_y):
            if assign_child_1:
            #   First individual in pop_d must be selected:
                candid_d_1[row, col] = pop_d[0, col]
                candid_d_2[row, col] = pop_d[random_arr_d[row, 1], col]
                candid_d_3[row, col] = pop_d[random_arr_d[row, 2], col]
                candid_d_4[row, col] = pop_d[random_arr_d[row, 3], col]
            else:
            #   Create a pool of 4 randomly selected individuals:
                candid_d_1[row, col] = pop_d[random_arr_d[row, 0], col]
                candid_d_2[row, col] = pop_d[random_arr_d[row, 1], col]
                candid_d_3[row, col] = pop_d[random_arr_d[row, 2], col]
                candid_d_4[row, col] = pop_d[random_arr_d[row, 3], col]

    cuda.syncthreads()
@cuda.jit
def select_parents(pop_d, candid_d_1, candid_d_2, candid_d_3, candid_d_4, parent_d_1, parent_d_2):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):
        for col in range(threadId_col, pop_d.shape[1], stride_y):
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

    cuda.syncthreads()

@cuda.jit
def number_cut_points(candid_d_1, candid_d_2, candid_d_3, candid_d_4, parent_d_1, parent_d_2, count, min_n, max_n):

    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, candid_d_1.shape[0], stride_x):
        for col in range(threadId_col, candid_d_1.shape[1], stride_y):
            candid_d_1[row, col] = 1
            candid_d_2[row, col] = 1
            candid_d_3[row, col] = 1
            candid_d_4[row, col] = 1

        # Calculate the actual length of parents
        if threadId_col == 15:
            for i in range(0, candid_d_1.shape[1]-2):
                if not (parent_d_1[row, i] == 1 and parent_d_1[row, i+1] == 1):
                    candid_d_1[row, 2] += 1

                if not (parent_d_2[row, i] == 1 and parent_d_2[row, i+1] == 1):
                    candid_d_2[row, 2] += 1

            # Minimum length of the two parents
            candid_d_1[row, 3] = \
            min(candid_d_1[row, 2], candid_d_2[row, 2])

            # Number of cutting points = (n/5 - 2)
            # candid_d_1[row, 4] = candid_d_1[row, 3]//20 - 2
            n_points = max(min_n, (count%(max_n*4000))//4000) # the n_points increases one every 5000 iterations till 20 then resets to 2 and so on
            candid_d_1[row, 4] = n_points

    cuda.syncthreads()
@cuda.jit
def add_cut_points(candid_d_1, candid_d_2, rng_states):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, candid_d_1.shape[0], stride_x):
        if threadId_col == 15:
            no_cuts = candid_d_1[row, 4]

            for i in range(1, no_cuts+1):
                rnd_val = 0

            # Generate unique random numbers as cut indices:
                for j in range(1, no_cuts+1):
                    while rnd_val == 0 or rnd_val == candid_d_2[row, j]:
                        # rnd = xoroshiro128p_uniform_float32(rng_states, row)\
                        #       *(candid_d_1[row, 3] - 2) + 2 # random*(max-min)+min
                        rnd = xoroshiro128p_uniform_float32(rng_states, row*candid_d_1.shape[1])\
                            *(candid_d_1[row, 3] - 2) + 2 # random*(max-min)+min
                        # rnd = xoroshiro128p_normal_float32(rng_states, row*candid_d_1.shape[1])\
                        #       *(candid_d_1[row, 3] - 2) + 2 # random*(max-min)+min
                        rnd_val = int(rnd)+2

                candid_d_2[row, i+1] = rnd_val

            # Sorting the crossover points:
            if threadId_col == 15: # Really! it is already up there! see the main if statement.
                for i in range(2, no_cuts+2):
                    min_val = candid_d_2[row, i]
                    min_index = i

                    for j in range(i + 1, no_cuts+2):
                        # Select the smallest value
                        if candid_d_2[row, j] < candid_d_2[row, min_index]:
                            min_index = j

                    candid_d_2[row, min_index], candid_d_2[row, i] = \
                    candid_d_2[row, i], candid_d_2[row, min_index]

    cuda.syncthreads()
@cuda.jit
def cross_over_gpu(candid_d_1, candid_d_2, child_d_1, child_d_2, parent_d_1, parent_d_2):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, candid_d_1.shape[0], stride_x):
        for col in range(threadId_col, candid_d_1.shape[1], stride_y):
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

    cuda.syncthreads()
# ------------------------------------Mutation part -----------------------------------------------
@cuda.jit
def mutate(rng_states, child_d_1, child_d_2):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, child_d_1.shape[0], stride_x):
    # Swap two positions in the children, with 1:40 probability
        if threadId_col == 15:
            mutation_prob = 15

            # rnd = xoroshiro128p_uniform_float32(rng_states, row)\
            #       *(mutation_prob - 1) + 1 # random*(max-min)+min
            rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_1.shape[1])\
                *(mutation_prob - 1) + 1 # random*(max-min)+min
            # rnd = xoroshiro128p_normal_float32(rng_states, row*child_d_1.shape[1])\
            #       *(mutation_prob - 1) + 1 # random*(max-min)+min
            rnd_val = int(rnd)+2
            if rnd_val == 1:
                i1 = 1

                # Repeat random selection if depot was selected:
                while child_d_1[row, i1] == 1:
                    # rnd = xoroshiro128p_uniform_float32(rng_states, row)\
                    #       *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_1.shape[1])\
                        *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    # rnd = xoroshiro128p_normal_float32(rng_states, row*child_d_1.shape[1])\
                    #     *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    i1 = int(rnd)+2

                i2 = 1
                while child_d_1[row, i2] == 1:
                    # rnd = xoroshiro128p_uniform_float32(rng_states, row)\
                    #     *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_1.shape[1])\
                        *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    # rnd = xoroshiro128p_normal_float32(rng_states, row*child_d_1.shape[1])\
                    #     *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    i2 = int(rnd)+2

                child_d_1[row, i1], child_d_1[row, i2] = \
                child_d_1[row, i2], child_d_1[row, i1]

            # Repeat for the second child:
                i1 = 1

                # Repeat random selection if depot was selected:
                while child_d_2[row, i1] == 1:
                    # rnd = xoroshiro128p_uniform_float32(rng_states, row)\
                    #     *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_2.shape[1])\
                        *(child_d_2.shape[1] - 4) + 2 # random*(max-min)+min
                    # rnd = xoroshiro128p_normal_float32(rng_states, row*child_d_2.shape[1])\
                    #     *(child_d_2.shape[1] - 4) + 2 # random*(max-min)+min
                    i1 = int(rnd)+2

                i2 = 1
                while child_d_2[row, i2] == 1:
                    # rnd = xoroshiro128p_uniform_float32(rng_states, row)\
                    #     *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_2.shape[1])\
                        *(child_d_2.shape[1] - 4) + 2 # random*(max-min)+min
                    # rnd = xoroshiro128p_normal_float32(rng_states, row*child_d_2.shape[1])\
                    #     *(child_d_2.shape[1] - 4) + 2 # random*(max-min)+min
                    i2 = int(rnd)+2

                child_d_2[row, i1], child_d_1[row, i2] = \
                child_d_2[row, i2], child_d_1[row, i1]

        cuda.syncthreads()
# -------------------------- Update population part -----------------------------------------------
@cuda.jit
def select_individual(index, pop_d, individual):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):
        if row == index and threadId_col < pop_d.shape[1]:
            pop_d[row, threadId_col] = individual[row, threadId_col]

@cuda.jit
def update_pop(count, parent_d_1, parent_d_2, child_d_1, child_d_2, pop_d):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):
        for col in range(threadId_col, pop_d.shape[1], stride_y):
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

    cuda.syncthreads()

# ------------------------- Definition of CPU functions ----------------------------------------------
def select_bests(parent_d_1, parent_d_2, child_d_1, child_d_2, pop_d, popsize):
    # Select the best 5% from paernt 1 & parent 2:
    pool = parent_d_1[parent_d_1[:,-1].argsort()][0:0.05*popsize,:]
    pool = cp.concatenate((pool, parent_d_2[parent_d_2[:,-1].argsort()][0:0.05*popsize,:]))
    pool = pool[pool[:,-1].argsort()]

    # Sort child 1 & child 2:
    child_d_1 = child_d_1[child_d_1[:,-1].argsort()]
    child_d_2 = child_d_2[child_d_2[:,-1].argsort()]

    pop_d[0:0.05*popsize, :] = pool[0:0.05*popsize, :]
    pop_d[0.05*popsize:0.53*popsize, :] = child_d_1[0:0.48*popsize, :]
    pop_d[0.53*popsize:popsize, :] = child_d_2[0:0.47*popsize, :]
# ------------------------- Start Main ------------------------------------------------------------
vrp_capacity, data, opt = readInput()
popsize = 100
min_n = 2 # Maximum number of crossover points
max_n = 2 # Maximum number of crossover points

try:
    generations = int(sys.argv[2])
except:
    print('No generation number provided, taking 500 generations...')
    generations = 500

r_flag = 9999 # A flag for removal/replacement

data_d = cuda.to_device(data)

cost_table_d = cuda.device_array(shape=(data.shape[0], data.shape[0]), dtype=np.int32)

pop_d = cp.ones((popsize, int(2*data.shape[0])+2), dtype=np.int32)

missing_d = cp.zeros(shape=(popsize, pop_d.shape[1]), dtype=np.int32)

missing = np.ones(shape=(popsize,1), dtype=bool)
missing_elements = cuda.to_device(missing)

fitness_val = np.zeros(shape=(popsize,1), dtype=np.int32)
fitness_val_d = cuda.to_device(fitness_val)

# GPU grid configurations:
threads_per_block = (20, 20)
blocks_no = 5

blocks = (blocks_no, blocks_no)
# --------------Calculate the cost table----------------------------------------------
start_event = cuda.event()
end_event = cuda.event()
start_event.record()

calc_cost_gpu[blocks, threads_per_block](data_d, popsize, vrp_capacity, cost_table_d)

end_event.record()
end_event.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
print(f"GPU kernel execution time for calc_cost_gpu: {elapsed_time:.3f} ms")

# --------------Initialize population----------------------------------------------
rng_states = create_xoroshiro128p_states(threads_per_block[0]**2 * blocks[0]**2, seed=random.randint(2,2*10**5))

start_event1 = cuda.event()
end_event1 = cuda.event()
start_event1.record()

initializePop_gpu[blocks, threads_per_block](rng_states, data_d, missing_d, pop_d)

end_event1.record()
end_event1.synchronize()
elapsed_time1 = start_event1.elapsed_time(end_event1)
print(f"GPU kernel execution time for initializePop_gpu: {elapsed_time1:.3f} ms")

for individual in pop_d:
    cp.random.shuffle(individual[2:-1])

start_event2 = cuda.event()
end_event2 = cuda.event()
start_event2.record()

find_duplicates[blocks, threads_per_block](pop_d, r_flag)

end_event2.record()
end_event2.synchronize()
elapsed_time2 = start_event2.elapsed_time(end_event2)
print(f"GPU kernel execution time for find_duplicates: {elapsed_time2:.3f} ms")

start_event3 = cuda.event()
end_event3 = cuda.event()
start_event3.record()

find_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, pop_d)

end_event3.record()
end_event3.synchronize()
elapsed_time3 = start_event3.elapsed_time(end_event3)
print(f"GPU kernel execution time for find_missing_nodes: {elapsed_time3:.3f} ms")

start_event4 = cuda.event()
end_event4 = cuda.event()
start_event4.record()

add_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, pop_d)

end_event4.record()
end_event4.synchronize()
elapsed_time4 = start_event4.elapsed_time(end_event4)
print(f"GPU kernel execution time for add_missing_nodes: {elapsed_time4:.3f} ms")

start_event5 = cuda.event()
end_event5 = cuda.event()
start_event5.record()

shift_r_flag[blocks, threads_per_block](r_flag, vrp_capacity, data_d, pop_d)

end_event5.record()
end_event5.synchronize()
elapsed_time5 = start_event5.elapsed_time(end_event5)
print(f"GPU kernel execution time for shift_r_flag: {elapsed_time5:.3f} ms")

start_event6 = cuda.event()
end_event6 = cuda.event()
start_event6.record()

cap_adjust[blocks, threads_per_block](r_flag, vrp_capacity, data_d, pop_d)

end_event6.record()
end_event6.synchronize()
elapsed_time6 = start_event6.elapsed_time(end_event6)
print(f"GPU kernel execution time for cap_adjust: {elapsed_time6:.3f} ms")

start_event7 = cuda.event()
end_event7 = cuda.event()
start_event7.record()

cleanup_r_flag[blocks, threads_per_block](r_flag, pop_d)

end_event7.record()
end_event7.synchronize()
elapsed_time7 = start_event7.elapsed_time(end_event7)
print(f"GPU kernel execution time for cleanup_r_flag: {elapsed_time7:.3f} ms")

# --------------Calculate fitness----------------------------------------------
start_event8 = cuda.event()
end_event8 = cuda.event()
start_event8.record()

fitness_gpu[blocks, threads_per_block](cost_table_d, pop_d, fitness_val_d)

end_event8.record()
end_event8.synchronize()
elapsed_time8 = start_event8.elapsed_time(end_event8)
print(f"GPU kernel execution time for fitness_gpu: {elapsed_time8:.3f} ms")

pop_d = pop_d[pop_d[:,-1].argsort()] # Sort the population to get the best later

asnumpy_first_pop = cp.asnumpy(pop_d)
# ------------------------- End Main ------------------------------------------------------------

# --------------Evolve population for some generations----------------------------------------------
# Create the pool of 6 arrays of the same length
candid_d_1 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)
candid_d_2 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)
candid_d_3 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)
candid_d_4 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)

parent_d_1 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)
parent_d_2 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)

child_d_1 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)
child_d_2 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)

cut_idx = np.ones(shape=(pop_d.shape[1]), dtype=np.int32)
cut_idx_d = cuda.to_device(cut_idx)

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

    random_arr_d = cuda.to_device(random_arr)

    start_event9 = cuda.event()
    end_event9 = cuda.event()
    start_event9.record()

    select_candidates[blocks, threads_per_block]\
                    (pop_d, random_arr_d, candid_d_1, candid_d_2, candid_d_3, candid_d_4, assign_child_1)

    end_event9.record()
    end_event9.synchronize()
    elapsed_time9 = start_event9.elapsed_time(end_event9)
    if count == 0:
      print(f"GPU kernel execution time for select_candidates: {elapsed_time9:.3f} ms")

    start_event10 = cuda.event()
    end_event10 = cuda.event()
    start_event10.record()

    select_parents[blocks, threads_per_block]\
                (pop_d, candid_d_1, candid_d_2, candid_d_3, candid_d_4, parent_d_1, parent_d_2)

    end_event10.record()
    end_event10.synchronize()
    elapsed_time10 = start_event10.elapsed_time(end_event10)
    if count == 0:
      print(f"GPU kernel execution time for select_parents: {elapsed_time10:.3f} ms")

    start_event11 = cuda.event()
    end_event11 = cuda.event()
    start_event11.record()

    number_cut_points[blocks, threads_per_block](candid_d_1, candid_d_2, \
                        candid_d_3, candid_d_4, parent_d_1, parent_d_2, count, min_n, max_n)

    end_event11.record()
    end_event11.synchronize()
    elapsed_time11 = start_event11.elapsed_time(end_event11)
    if count == 0:
      print(f"GPU kernel execution time for number_cut_points: {elapsed_time11:.3f} ms")

    rng_states = create_xoroshiro128p_states(popsize*pop_d.shape[1], seed=random.randint(2,2*10**5))

    start_event12 = cuda.event()
    end_event12 = cuda.event()
    start_event12.record()

    add_cut_points[blocks, threads_per_block](candid_d_1, candid_d_2, rng_states)

    end_event12.record()
    end_event12.synchronize()
    elapsed_time12 = start_event12.elapsed_time(end_event12)
    if count == 0:
      print(f"GPU kernel execution time for add_cut_points: {elapsed_time12:.3f} ms")

    start_event13 = cuda.event()
    end_event13 = cuda.event()
    start_event13.record()

    cross_over_gpu[blocks, threads_per_block](candid_d_1, candid_d_2, child_d_1, child_d_2, parent_d_1, parent_d_2)

    end_event13.record()
    end_event13.synchronize()
    elapsed_time13 = start_event13.elapsed_time(end_event13)
    if count == 0:
      print(f"GPU kernel execution time for cross_over_gpu: {elapsed_time13:.3f} ms")

    # Performing mutation
    rng_states = create_xoroshiro128p_states(popsize*child_d_1.shape[1], seed=random.randint(2,2*10**5))

    start_event14 = cuda.event()
    end_event14 = cuda.event()
    start_event14.record()

    mutate[blocks, threads_per_block](rng_states, child_d_1, child_d_2)

    end_event14.record()
    end_event14.synchronize()
    elapsed_time14 = start_event14.elapsed_time(end_event14)
    if count == 0:
      print(f"GPU kernel execution time for mutate: {elapsed_time14:.3f} ms")


    # Adjusting child_1 array
    find_duplicates[blocks, threads_per_block](child_d_1, r_flag)

    find_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, child_d_1)
    add_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, child_d_1)

    shift_r_flag[blocks, threads_per_block](r_flag, vrp_capacity, data_d, child_d_1)
    cap_adjust[blocks, threads_per_block](r_flag, vrp_capacity, data_d, child_d_1)
    cleanup_r_flag[blocks, threads_per_block](r_flag, child_d_1)

    # Adjusting child_2 array
    find_duplicates[blocks, threads_per_block](child_d_2, r_flag)

    find_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, child_d_2)
    add_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, child_d_2)

    shift_r_flag[blocks, threads_per_block](r_flag, vrp_capacity, data_d, child_d_2)
    cap_adjust[blocks, threads_per_block](r_flag, vrp_capacity, data_d, child_d_2)
    cleanup_r_flag[blocks, threads_per_block](r_flag, child_d_2)
    # --------------------------------------------------------------------------
    # Performing the two-opt optimization and Calculating fitness for child_1 array

    start_event15 = cuda.event()
    end_event15 = cuda.event()
    start_event15.record()

    reset_to_ones[blocks, threads_per_block](candid_d_3)

    end_event15.record()
    end_event15.synchronize()
    elapsed_time15 = start_event15.elapsed_time(end_event15)
    if count == 0:
      print(f"GPU kernel execution time for reset_to_ones: {elapsed_time15:.3f} ms")

    start_event16 = cuda.event()
    end_event16 = cuda.event()
    start_event16.record()

    two_opt[blocks, threads_per_block](child_d_1, cost_table_d, candid_d_3)

    end_event16.record()
    end_event16.synchronize()
    elapsed_time16 = start_event16.elapsed_time(end_event16)
    if count == 0:
      print(f"GPU kernel execution time for two_opt: {elapsed_time16:.3f} ms")

    fitness_gpu[blocks, threads_per_block](cost_table_d, child_d_1, fitness_val_d)
    # --------------------------------------------------------------------------
    # Performing the two-opt optimization and Calculating fitness for child_2 array
    reset_to_ones[blocks, threads_per_block](candid_d_3)

    two_opt[blocks, threads_per_block](child_d_2, cost_table_d, candid_d_3)

    fitness_gpu[blocks, threads_per_block](cost_table_d, child_d_2, fitness_val_d)
    # --------------------------------------------------------------------------
    # Creating the new population from parents and children
    select_bests(parent_d_1, parent_d_2, child_d_1, child_d_2, pop_d, popsize)
    # --------------------------------------------------------------------------

    # Replacing duplicates with random individuals from child_d_1
    asnumpy_pop_d = cp.asnumpy(pop_d) # copy pop_d to host, HOWEVER, it throws cudaErrorIllegalAddress in >= 800 nodes
    asnumpy_child_d_1 = cp.asnumpy(child_d_1) # copy child_d_1 to host
    repeats = 0

    x = np.unique(asnumpy_pop_d[:,1:], axis=0)
    # print('X pre:', x.shape[0], x, '\n-------------\n')
    while x.shape[0] < popsize:
        if repeats >= popsize-1:
            break
        rndm = random.randint(0, popsize-1)
        x = np.append(x, [asnumpy_child_d_1[rndm,1:]], axis=0)
        x = np.unique(x, axis=0)
        repeats += 1

    # --------------------------------------------------------------------------
    # Replacing duplicates with random individuals from parent_d_2
    asnumpy_parent_d_2 = cp.asnumpy(parent_d_2) # copy parent_d_1 to host
    repeats = 0
    while x.shape[0] < popsize:
        if repeats >= popsize-1:
            break

        rndm = random.randint(0, popsize-1)
        x = np.append(x, [asnumpy_parent_d_2[rndm,1:]], axis=0)
        # x = np.unique(x, axis=0)
        repeats += 1
    # --------------------------------------------------------------------------
    x = np.insert(x, 0, count, axis=1)
    pop_d = cp.array(x)
    # --------------------------------------------------------------------------
    # Picking best solution
    old_cost = minimum_cost
    # best_sol = pop_d[0,:]
    best_sol = pop_d[pop_d[:,-1].argmin()]
    minimum_cost = best_sol[-1]

    worst_sol = pop_d[pop_d[:,-1].argmax()]
    worst_cost = worst_sol[-1]

    delta = worst_cost-minimum_cost
    average = cp.average(pop_d[:,-1])


    if minimum_cost == old_cost: # To calculate for how long the quality did not improve
        count_index += 1
    else:
        count_index = 0

    # Shuffle the population after a certain number of generations without improvement
    assign_child_1 = False

    if count == 1:
        print('At first generation, Best: %d,'%minimum_cost, 'Worst: %d'%worst_cost, \
            'delta: %d'%delta, 'Avg: %.2f'%average)
        # print('POP:', pop_d, end='\n-----------\n')
    elif (count+1)%100 == 0:
        print('After %d generations, Best: %d,'%(count+1, minimum_cost), 'Worst: %d'%worst_cost, \
            'delta: %d'%delta, 'Avg: %.2f'%average)
    count += 1

current_time = timer()
total_time = float('{0:.4f}'.format((current_time - old_time)))
time_per_loop = float('{0:.4f}'.format((current_time - old_time)/(count-1)))


best_sol = cp.subtract(best_sol, cp.ones_like(best_sol))
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

current_time = timer()
total_time = float('{0:.4f}'.format((current_time - old_time)))

del data_d
del cost_table_d
del pop_d
del missing_d
del fitness_val_d

del candid_d_1
del candid_d_2
del candid_d_3
del candid_d_4

del parent_d_1
del parent_d_2

del child_d_1
del child_d_2

del cut_idx_d