# -------- Start of the importing part -----------
import warnings
from numba.core.errors import NumbaPerformanceWarning
from numba import cuda, jit, int32, float32, int64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32
import cupy as cp
from math import pow, hypot, ceil
import numpy as np
import time
import random
from timeit import default_timer as timer
import sys
from google.colab import drive
np.set_printoptions(threshold=sys.maxsize)
# -------- End of the importing part -----------

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

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
def calc_cost_gpu(data_d, cost_table_d):

    row, col = cuda.grid(2)  # Obtaining two indices, row and col, which represent positions in the distance table.

    # Checking if the index exceeds the dimensions
    if row < data_d.shape[0] and col < data_d.shape[0]:
        # When the thread calculates the index, it uses it to access data_d and retrieve the coordinates for points i and j.
        x1, y1 = data_d[row, 2], data_d[row, 3]
        x2, y2 = data_d[col, 2], data_d[col, 3]

        # Calculating distance
        cost_table_d[row, col] = round(hypot(x2 - x1, y2 - y1))
# ------------------------- End calculating the cost table ----------------------------------------

# ------------------------- Start fitness calculation ---------------------------------------------
@cuda.jit
def fitness_gpu_old(cost_table_d, pop, fitness_val_d):
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
def fitness_gpu(cost_table_d, pop, fitness_val_d):
    # Get thread ID
    row = cuda.grid(1)
    stride = cuda.gridsize(1)

    for r in range(row, pop.shape[0], stride):
        # Reset fitness value
        fitness_val_d[r, 0] = 0
        pop[r, -1] = 1

        # Compute fitness value for this row
        fitness = 0
        for i in range(pop.shape[1] - 2):
            fitness += cost_table_d[pop[r, i] - 1, pop[r, i + 1] - 1]

        # Save fitness value
        fitness_val_d[r, 0] = fitness
        pop[r, -1] = fitness

# ------------------------- End fitness calculation ---------------------------------------------

# ------------------------- Start adjusting individuals ---------------------------------------------
@cuda.jit
def find_duplicates_old(pop, r_flag):

    # receives (100,12) i r_flag = 9999
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
def find_duplicates(pop, r_flag):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        # Let each thread operate on its assigned row
        for i in range(2, pop.shape[1] - 1):
          if pop[row, i] == 1:
            pop[row, i] = r_flag
          for j in range(i + 1, pop.shape[1] - 1):  # Avoid comparing (i, i)
            if pop[row, i] != r_flag and pop[row, i] == pop[row, j]:
                pop[row, j] = r_flag

@cuda.jit
def shift_r_flag_old(r_flag, pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        # Shift all r_flag values to the end of the list:
        for i in range(2, pop.shape[1]-2):
            if pop[row,i] == r_flag:
                k = i
                while pop[row,k] == r_flag:
                    k += 1
                if k < pop.shape[1]-1:
                    pop[row,i], pop[row,k] = pop[row,k], pop[row,i]

@cuda.jit
def shift_r_flag(r_flag, pop):
    # Use a 1D grid: one thread per row.
    row = cuda.grid(1)

    if row < pop.shape[0]:
        # Shift all r_flag values to the end of the list:
        for i in range(2, pop.shape[1] - 2):
            if pop[row, i] == r_flag:
                k = i
                # Advance k while we are within bounds and still seeing r_flag
                while k < pop.shape[1] and pop[row, k] == r_flag:
                    k += 1
                # If we found a valid element later in the row, swap it in.
                if k < pop.shape[1] - 1:
                    temp = pop[row, i]
                    pop[row, i] = pop[row, k]
                    pop[row, k] = temp

@cuda.jit
def find_missing_nodes(data_d, missing_d, pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
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
def cap_adjust_new(r_flag, vrp_capacity, data_d, pop):
    # Get the thread's row index
    threadId_row = cuda.grid(1)  # Use grid(1) since we're processing rows, not columns

    # Ensure the thread is within bounds (to avoid accessing out-of-bounds memory)
    if threadId_row < pop.shape[0]:
        reqcap = 0.0
        i = 1
        while pop[threadId_row, i] != r_flag:
            i += 1
            if pop[threadId_row, i] == r_flag:
                break

            if pop[threadId_row, i] != 1:
                reqcap += data_d[pop[threadId_row, i]-1, 1]
                if reqcap > vrp_capacity:
                    reqcap = 0
                    new_val = 1
                    rep_val = pop[threadId_row, i]
                    for j in range(i, pop.shape[1] - 2):
                        pop[threadId_row, j] = new_val
                        new_val = rep_val
                        rep_val = pop[threadId_row, j + 1]
            else:
                reqcap = 0.0
    cuda.syncthreads()

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
def initializePop_gpu(data_d, pop_d):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    if threadId_row < pop_d.shape[0] and threadId_col < data_d.shape[0]:
        pop_d[threadId_row, threadId_col + 1] = data_d[threadId_col, 0]

    if threadId_row < pop_d.shape[0] and threadId_col == 0:
        pop_d[threadId_row, 0], pop_d[threadId_row, 1] = 1, 1
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
# pop_d = np.array([[58, 8, 98],
#                   [0, 93, 83],
#                   [45, 34, 26],
#                   [30, 82, 65]])

# random_arr_d = np.array([[0, 1, 2, 3],
#                          [1, 2, 3, 0],
#                          [2, 3, 0, 1],
#                          [3, 0, 1, 2]])

# candid_d_1:
# [[58, 8, 98],
#  [0, 93, 83],
#  [45, 34, 26],
#  [30, 82, 65]]

# candid_d_2:
# [[0, 93, 83],
#  [45, 34, 26],
#  [30, 82, 65],
#  [58, 8, 98]]

# candid_d_3:
# [[45, 34, 26],
#  [30, 82, 65],
#  [58, 8, 98],
#  [0, 93, 83]]

# candid_d_4:
# [[30, 82, 65],
#  [58, 8, 98],
#  [0, 93, 83],
#  [45, 34, 26]]

# Using random selection for candidates increases diversity
# If assign_child is true, the diversity is reduced by placing the fixed first element of the population (the best one)

    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):
        for col in range(threadId_col, pop_d.shape[1], stride_y):
            if assign_child_1: # When entering the while loop for the first time, this is false.
            #   First individual in pop_d must be selected:
                candid_d_1[row, col] = pop_d[0, col]
                candid_d_2[row, col] = pop_d[random_arr_d[row, 1], col]
                candid_d_3[row, col] = pop_d[random_arr_d[row, 2], col]
                candid_d_4[row, col] = pop_d[random_arr_d[row, 3], col]
            else:
            #   Create a pool of 4 randomly selected individuals: svi kandidati će biti odabrani prema nasumično odabranim indeksima u random_arr_d
                candid_d_1[row, col] = pop_d[random_arr_d[row, 0], col]
                candid_d_2[row, col] = pop_d[random_arr_d[row, 1], col]
                candid_d_3[row, col] = pop_d[random_arr_d[row, 2], col]
                candid_d_4[row, col] = pop_d[random_arr_d[row, 3], col]

    cuda.syncthreads()

@cuda.jit
def select_parents(pop_d, candid_d_1, candid_d_2, candid_d_3, candid_d_4, parent_d_1, parent_d_2):
# candid_d_1:
# [[58, 8, 98],
#  [0, 93, 83],
#  [45, 34, 26], 
#  [30, 82, 65]] 

# candid_d_2:
# [[0, 93, 83], 
#  [45, 34, 26], 
#  [30, 82, 65],
#  [58, 8, 98]]

# parent_d_1
# [[0, 93, 83],
#  [45, 34, 26],
#  [45, 34, 26],
#  [30, 82, 65]]

# isto i za parent2

    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):
        for col in range(threadId_col, pop_d.shape[1], stride_y):
        # Selecting 2 parents with binary tournament
        # ----------------------------1st Parent--------------------------------------------------
            if candid_d_1[row, -1] < candid_d_2[row, -1]:  
                parent_d_1[row, col] = candid_d_1[row, col]
            else: #
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

    # All values in the candidate matrices are set to 1
    for row in range(threadId_row, candid_d_1.shape[0], stride_x):
        for col in range(threadId_col, candid_d_1.shape[1], stride_y):
            candid_d_1[row, col] = 1
            candid_d_2[row, col] = 1
            candid_d_3[row, col] = 1
            candid_d_4[row, col] = 1

        # Calculate the actual length of parents
        if threadId_col == 15:
            for i in range(0, candid_d_1.shape[1]-2): # It iterates through all columns of the parents except for the last two, because the last two are values like 1, 245, etc., which are not nodes but just the fitness function.
                if not (parent_d_1[row, i] == 1 and parent_d_1[row, i+1] == 1): # Each time a pair of elements is not (1,1), the value is incremented by 1
                    candid_d_1[row, 2] += 1

                if not (parent_d_2[row, i] == 1 and parent_d_2[row, i+1] == 1):
                    candid_d_2[row, 2] += 1

            # Minimum length of the two parents
            candid_d_1[row, 3] = \
            min(candid_d_1[row, 2], candid_d_2[row, 2])

            # Number of cutting points = (n/5 - 2)
            # candid_d_1[row, 4] = candid_d_1[row, 3]//20 - 2
            n_points = max(min_n, (count % (max_n * 4000)) // 4000) # the n_points increases one every 5000 iterations till 20 then resets to 2 and so on
            candid_d_1[row, 4] = n_points

    cuda.syncthreads()

@cuda.jit
def number_cut_points_new(candid_d_1, candid_d_2, candid_d_3, candid_d_4, parent_d_1, parent_d_2, count, min_n, max_n):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, candid_d_1.shape[0], stride_x):
        if threadId_col < candid_d_1.shape[1]:
            # Initialize candidate matrices
            candid_d_1[row, threadId_col] = 1
            candid_d_2[row, threadId_col] = 1
            candid_d_3[row, threadId_col] = 1
            candid_d_4[row, threadId_col] = 1

        # Only one thread per row calculates the length of parents and cut points
        if threadId_col == 0:
            length_1 = 0
            length_2 = 0
            for i in range(candid_d_1.shape[1] - 1):
                if not (parent_d_1[row, i] == 1 and parent_d_1[row, i+1] == 1):
                    length_1 += 1
                if not (parent_d_2[row, i] == 1 and parent_d_2[row, i+1] == 1):
                    length_2 += 1

            candid_d_1[row, 2] = length_1
            candid_d_2[row, 2] = length_2
            candid_d_1[row, 3] = min(length_1, length_2)

            n_points = max(min_n, (count % (max_n * 4000)) // 4000)
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
                        rnd = xoroshiro128p_uniform_float32(rng_states, row*candid_d_1.shape[1])\
                            *(candid_d_1[row, 3] - 2) + 2
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
def add_cut_points_new(candid_d_1, candid_d_2, rng_states):
    threadId_row = cuda.grid(1)

    if threadId_row < candid_d_1.shape[0]:
        no_cuts = candid_d_1[threadId_row, 4]  # Get number of cuts (2)
        sol_length = candid_d_1[threadId_row, 3]  # Get solution length (25/26)

        # Generate cut points
        for i in range(1, no_cuts + 1):
            unique = False
            candidate = 0
            while not unique:
                rnd = xoroshiro128p_uniform_float32(rng_states, threadId_row) * (sol_length - 2) + 2
                candidate = int(rnd) + 2


                unique = True  # Assume candidate is unique until we find a duplicate.
                # Check only the already–generated cut points.
                # For the first cut (i == 1), there are no previous points.
                # For subsequent ones, check those stored at positions 2 to i+1.
                for j in range(2, i + 1):
                    if candid_d_2[threadId_row, j] == candidate:
                        unique = False
                        break


            # Store the unique candidate at index (i+1) (since cut points start at column 2)
            candid_d_2[threadId_row, i + 1] = candidate

        # Sort cut points
        for i in range(2, no_cuts+2):
            for j in range(i+1, no_cuts+2):
                if candid_d_2[threadId_row, j] < candid_d_2[threadId_row, i]:
                    # Swap points
                    candid_d_2[threadId_row, i], candid_d_2[threadId_row, j] = \
                    candid_d_2[threadId_row, j], candid_d_2[threadId_row, i]

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
                no_cuts = candid_d_1[row, 4] # definiše koliko presjeka će biti
                if col < candid_d_2[row, 2]: # Swap from first element to first cut point
                    child_d_1[row, col], child_d_2[row, col] =\
                    child_d_2[row, col], child_d_1[row, col]

                if no_cuts%2 == 0: # For even number of cuts, swap from the last cut point to the end
                    if col > candid_d_2[row, no_cuts+1] and col < child_d_1.shape[1]-1:
                        child_d_1[row, col], child_d_2[row, col] =\
                        child_d_2[row, col], child_d_1[row, col]

                for j in range(2, no_cuts+1):
                    cut_idx = candid_d_2[row, j]
                    if j % 2 == 1 and col >= cut_idx and j + 1 <= no_cuts and col < candid_d_2[row, j + 1]:
                            child_d_1[row, col], child_d_2[row, col] = child_d_2[row, col], child_d_1[row, col]

                    elif no_cuts%2 == 1:
                        if j%2==1 and col>=cut_idx and col < candid_d_2[row, j+1]:
                            child_d_1[row, col], child_d_2[row, col] =\
                            child_d_2[row, col], child_d_1[row, col]
    cuda.syncthreads()
# ------------------------------------Mutation part -----------------------------------------------
@cuda.jit
def mutate_old(rng_states, child_d_1, child_d_2):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, child_d_1.shape[0], stride_x):
    # Swap two positions in the children, with 1:40 probability
        if threadId_col == 15:
            mutation_prob = 15
            rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_1.shape[1])\
                *(mutation_prob - 1) + 1
            rnd_val = int(rnd)+2
            if rnd_val == 1:
                i1 = 1

                # Repeat random selection if depot was selected:
                while child_d_1[row, i1] == 1:
                    rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_1.shape[1])\
                        *(child_d_1.shape[1] - 4) + 2
                    i1 = int(rnd)+2

                i2 = 1
                while child_d_1[row, i2] == 1:
                    rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_1.shape[1])\
                        *(child_d_1.shape[1] - 4) + 2
                    i2 = int(rnd)+2

                child_d_1[row, i1], child_d_1[row, i2] = \
                child_d_1[row, i2], child_d_1[row, i1]

            # Repeat for the second child:
                i1 = 1
                # Repeat random selection if depot was selected:
                while child_d_2[row, i1] == 1:
                    rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_2.shape[1])\
                        *(child_d_2.shape[1] - 4) + 2
                    i1 = int(rnd)+2

                i2 = 1
                while child_d_2[row, i2] == 1:
                    rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_2.shape[1])\
                        *(child_d_2.shape[1] - 4) + 2
                    i2 = int(rnd)+2

                child_d_2[row, i1], child_d_1[row, i2] = \
                child_d_2[row, i2], child_d_1[row, i1]

        cuda.syncthreads()

@cuda.jit
def mutate_aida(rng_states, child_d_1, child_d_2):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, child_d_1.shape[0], stride_x):
        for col in range(threadId_col, child_d_1.shape[1], stride_y):
            # Mutation probability (e.g., 1 in 40)
            mutation_prob = 80
            rnd = xoroshiro128p_uniform_float32(rng_states, row * child_d_1.shape[1] + col)

            if rnd < 1.0 / mutation_prob:  # Perform mutation
                i1 = int(xoroshiro128p_uniform_float32(rng_states, row) * (child_d_1.shape[1] - 2)) + 1
                i2 = int(xoroshiro128p_uniform_float32(rng_states, row) * (child_d_1.shape[1] - 2)) + 1

                # Ensure indices are different and avoid depot (index 1)
                while i1 == i2 or child_d_1[row, i1] == 1 or child_d_1[row, i2] == 1:
                    i1 = int(xoroshiro128p_uniform_float32(rng_states, row) * (child_d_1.shape[1] - 2)) + 1
                    i2 = int(xoroshiro128p_uniform_float32(rng_states, row) * (child_d_1.shape[1] - 2)) + 1

                # Swap values in child 1
                temp = child_d_1[row, i1]
                child_d_1[row, i1] = child_d_1[row, i2]
                child_d_1[row, i2] = temp

                # Repeat for child 2
                temp = child_d_2[row, i1]
                child_d_2[row, i1] = child_d_2[row, i2]
                child_d_2[row, i2] = temp

    cuda.syncthreads()

@cuda.jit
def mutate(rng_states, child_d_1, child_d_2):
    # Use a 1D grid: one thread per candidate (row)
    row = cuda.grid(1)
    ncols = child_d_1.shape[1]  # assume both children have same number of columns

    if row < child_d_1.shape[0]:
        # Use a simple probability test: if a random float in [0,1) is less than (1/40),
        # then perform mutation (i.e. a ~2.5% chance per candidate).
        if xoroshiro128p_uniform_float32(rng_states, row) < 1.0 / 40.0:
            # Mutate first child: swap two positions that are not depots (value 1)
            i1 = 1
            # Find a valid position for i1
            while child_d_1[row, i1] == 1:
                # Generate a random index in valid range [2, ncols-2)
                r = xoroshiro128p_uniform_float32(rng_states, row * ncols)
                i1 = int(r * (ncols - 4)) + 2

            i2 = 1
            # Find a valid position for i2
            while child_d_1[row, i2] == 1:
                r = xoroshiro128p_uniform_float32(rng_states, row * ncols)
                i2 = int(r * (ncols - 4)) + 2

            # Swap the two positions in child_d_1
            temp = child_d_1[row, i1]
            child_d_1[row, i1] = child_d_1[row, i2]
            child_d_1[row, i2] = temp

            # Mutate second child: swap two positions that are not depots
            i1 = 1
            while child_d_2[row, i1] == 1:
                r = xoroshiro128p_uniform_float32(rng_states, row * ncols)
                i1 = int(r * (ncols - 4)) + 2

            i2 = 1
            while child_d_2[row, i2] == 1:
                r = xoroshiro128p_uniform_float32(rng_states, row * ncols)
                i2 = int(r * (ncols - 4)) + 2

            # Swap the two positions in child_d_2
            temp = child_d_2[row, i1]
            child_d_2[row, i1] = child_d_2[row, i2]
            child_d_2[row, i2] = temp

# ------------------------- Definition of CPU functions ----------------------------------------------
def select_bests_old(parent_d_1, parent_d_2, child_d_1, child_d_2, pop_d, popsize):
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

def select_bests(parent_d_1, parent_d_2, child_d_1, child_d_2, pop_d, popsize):
    # Select the best 5% from paernt 1 & parent 2:
    pool = parent_d_1[parent_d_1[:,-1].argsort()][0:0.05*popsize,:]
    pool = cp.concatenate((pool, parent_d_2[parent_d_2[:,-1].argsort()][0:0.05*popsize,:]))
    pool = pool[pool[:,-1].argsort()]


    # Sort child 1 & child 2:
    child_d_1 = child_d_1[child_d_1[:,-1].argsort()]
    child_d_2 = child_d_2[child_d_2[:,-1].argsort()]

    i1 = round(0.05 * popsize)
    i2 = round(0.53 * popsize)


    pop_d[:i1, :] = pool[:i1, :]
    pop_d[i1:i2, :] = child_d_1[:(i2 - i1), :]
    pop_d[i2:, :] = child_d_2[:(popsize - i2), :]

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

# It won't compute because 4x4x1x1 = 16, but the matrix has 20 elements.
threads_per_block = (16, 16)
# blocks_no = 5
# TEST 1
# data = np.array([
#     [1, 0, 30, 40],
#     [2, 19, 37, 52],
#     [3, 30, 49, 49],
#     [4, 16, 52, 64],
#     [5, 2, 10, 69]
# ], dtype=np.float32)

# print("Input data (host):")
# print(data)
# print("\nData.shape[0]:")
# print(data.shape[0])

data_d = cuda.to_device(data)
cost_table_d = cuda.device_array(shape=(data.shape[0], data.shape[0]), dtype=np.int32)

# blocks = (blocks_no, blocks_no)

N = data.shape[0] # 151

# Blocks per grid = N / number of threads
# By adding thread_per_block[0] - 1 before the division, we ensure that all elements will be processed even when N is not divisible by threads_per_block[0]
# For example, blocks_per_grid_x = (20 + 16 - 1) / 16 = 2.1875, which means the GPU will use 2 blocks in the x dimension to cover all 20 elements, with each block having 16 threads

blocks_per_grid = (N + threads_per_block[0] - 1) // threads_per_block[0]
blocks = (blocks_per_grid, blocks_per_grid)

# --------------Calculate the cost table----------------------------------------------
start_event = cuda.event()
end_event = cuda.event()
start_event.record()
calc_cost_gpu[blocks, threads_per_block](data_d, cost_table_d)
end_event.record()
end_event.synchronize()
elapsed_time = cuda.event_elapsed_time(start_event, end_event)
print(f"calc_cost_gpu execution time: {elapsed_time:.3f} ms")

# cost_table_host = cost_table_d.copy_to_host()
# Print the calculated cost table
# print("\nCost table (host):")
# print(cost_table_host)

pop_d = cp.ones((popsize, int(2*data.shape[0])+2), dtype=np.int32)
# (100, 12) 
missing_d = cp.zeros(shape=(popsize, pop_d.shape[1]), dtype=np.int32)
# (100,12)
missing = np.ones(shape=(popsize,1), dtype=bool)
# (100,1) 
missing_elements = cuda.to_device(missing)

fitness_val = np.zeros(shape=(popsize,1), dtype=np.int32)
fitness_val_d = cuda.to_device(fitness_val)

# --------------Initialize population----------------------------------------------
rng_states = create_xoroshiro128p_states(threads_per_block[0]**2 * blocks[0]**2, seed=random.randint(2,2*10**5))

blocks = (
    (popsize + threads_per_block[0] - 1) // threads_per_block[0],
     (N + threads_per_block[0] - 1) // threads_per_block[0]
)
start_event = cuda.event()
end_event = cuda.event()
start_event.record()
initializePop_gpu[blocks, threads_per_block](data_d, pop_d)
end_event.record()
end_event.synchronize()
elapsed_time = cuda.event_elapsed_time(start_event, end_event)
print(f"initializePop_gpu execution time: {elapsed_time:.3f} ms")
# print("\nPop table after init:")
# print(pop_d.get())

for individual in pop_d:
    cp.random.shuffle(individual[2:-1])

# pop_d = cp.array([
#  [1, 1, 1, 5, 1, 1, 1, 4, 2, 1, 3, 1], # 1 red 1 
#  [1, 1, 4, 1, 1, 3, 1, 1, 5, 2, 1, 1], # block 1
#  [1, 1, 1, 1, 5, 1, 1, 4, 3, 2, 1, 1], # block 1
#  [1, 1, 4, 1, 1, 1, 2, 1, 3, 5, 1, 1], # block 1
#  [1, 1, 4, 3, 1, 2, 1, 1, 1, 1, 5, 1], # block 1
#  [1, 1, 1, 4, 2, 1, 3, 5, 1, 1, 1, 1], # block 1
#  [1, 1, 3, 4, 1, 1, 1, 1, 5, 2, 1, 1], # block 1
#  [1, 1, 4, 2, 1, 1, 1, 3, 1, 5, 1, 1], # block 1
#  [1, 1, 1, 3, 1, 2, 5, 1, 1, 1, 4, 1], # block 1
#  [1, 1, 1, 5, 1, 4, 3, 1, 1, 1, 2, 1], # block 1
#  [1, 1, 1, 3, 4, 1, 2, 5, 1, 1, 1, 1],
#  [1, 1, 4, 3, 2, 1, 5, 1, 1, 1, 1, 1],
#  [1, 1, 3, 1, 5, 1, 2, 1, 1, 4, 1, 1],
#  [1, 1, 1, 1, 2, 4, 1, 1, 5, 3, 1, 1],
#  [1, 1, 1, 1, 4, 3, 1, 5, 2, 1, 1, 1],
#  [1, 1, 5, 1, 1, 3, 1, 1, 4, 2, 1, 1],
#  [1, 1, 1, 1, 2, 4, 3, 1, 1, 1, 5, 1],
#  [1, 1, 1, 5, 4, 3, 1, 1, 2, 1, 1, 1],
#  [1, 1, 4, 1, 1, 5, 1, 1, 3, 1, 2, 1],
#  [1, 1, 1, 4, 2, 3, 1, 5, 1, 1, 1, 1]
# ])
# print("\nPop table before find duplicates:")
# print(pop_d.get())

blocks = (
    (popsize + threads_per_block[0] - 1) // threads_per_block[0], 1
)
start_event = cuda.event()
end_event = cuda.event()
start_event.record()
find_duplicates[blocks, threads_per_block](pop_d, r_flag) # gets (100,12) i r_flag = 9999
end_event.record()
end_event.synchronize()
elapsed_time = cuda.event_elapsed_time(start_event, end_event)
print(f"find_duplicates execution time: {elapsed_time:.3f} ms")
# print("\nPop table after find duplicates:")
# print(pop_d.get())

start_event = cuda.event()
end_event = cuda.event()
start_event.record()
shift_r_flag[ceil(pop_d.shape[0] / 128), 128](r_flag, pop_d)
end_event.record()
end_event.synchronize()
elapsed_time = cuda.event_elapsed_time(start_event, end_event)
print(f"shift_r_flag execution time: {elapsed_time:.3f} ms")
# print("\nPop table after shift_r_flag:")
# print(pop_d.get())

# 128 je 1D thread block block
#threads_per_block = 128
# (130 + (128 - 1)) // 128 = (130 + 127) // 128 = 257 // 128 = 2 blocks, while without threads_per_block - 1 it would be 1 block, meaning 2 elements would not be processed
# blocks_per_grid = (pop_d.shape[0] + (threads_per_block - 1)) // threads_per_block
#cap_adjust_new[blocks_per_grid, threads_per_block](r_flag, vrp_capacity, data_d, pop_d)

start_event = cuda.event()
end_event = cuda.event()
start_event.record()
cap_adjust[blocks_per_grid, threads_per_block](r_flag, vrp_capacity, data_d, pop_d)
end_event.record()
end_event.synchronize()
elapsed_time = cuda.event_elapsed_time(start_event, end_event)
print(f"cap_adjust execution time: {elapsed_time:.3f} ms")
# print("After cap_adjust pop_d:")
# print(pop_d.get())

blockspergrid_x = (pop_d.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
blockspergrid_y = (pop_d.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
# print("Before cleanup_r pop_d:")
# print(pop_d.get())
start_event = cuda.event()
end_event = cuda.event()
start_event.record()
cleanup_r_flag[(blockspergrid_x, blockspergrid_y), threads_per_block](r_flag, pop_d)
end_event.record()
end_event.synchronize()
elapsed_time = cuda.event_elapsed_time(start_event, end_event)
print(f"cleanup_r_flag execution time: {elapsed_time:.3f} ms")
# print("After cleanup_r pop_d:")
# print(pop_d.get())

# pop_d2 = cuda.to_device(pop_d.get()) put in new fitness for comparison
# blocks_no = 5
# blocks = (blocks_no, blocks_no)
# fitness_gpu_old[blocks, threads_per_block](cost_table_d, pop_d, fitness_val_d)
start_event = cuda.event()
end_event = cuda.event()
start_event.record()
fitness_gpu[(pop_d.shape[0] + 128 - 1) // 128, 128](cost_table_d, pop_d, fitness_val_d)
end_event.record()
end_event.synchronize()
elapsed_time = cuda.event_elapsed_time(start_event, end_event)
print(f"fitness_gpu execution time: {elapsed_time:.3f} ms")
# pop_h1 = pop_d.get()  # Converts CuPy array to NumPy array
# pop_h2 = pop_d2.copy_to_host()  # Converts CuPy array to NumPy array
# # Compare `pop_d` arrays
# if np.array_equal(pop_h1, pop_h2):
#     print("Both kernels produce the same result on `pop_d`.")
# else:
#     print("The results differ for `pop_d` between the kernels.")
#     diff_indices = np.where(pop_h1 != pop_h2)
#     print("Mismatched indices:", diff_indices)
#     print("Kernel 1 `pop` values:", pop_h1[diff_indices])
#     print("Kernel 2 `pop` values:", pop_h2[diff_indices])

pop_d = pop_d[pop_d[:,-1].argsort()] # Sort the population to get the best later
asnumpy_first_pop = cp.asnumpy(pop_d) # Converts to NumPy for CPU processing

# ------------------------- End Main ------------------------------------------------------------

# --------------Evolve population for some generations----------------------------------------------

# Create the pool of 6 arrays of the same length
# Potencijalni kandidati roditelji i djeca, iste dužine kao pop_d.shape[1]
candid_d_1 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)
candid_d_2 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)
candid_d_3 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)
candid_d_4 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)

parent_d_1 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)
parent_d_2 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)

child_d_1 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)
child_d_2 = cp.ones((popsize, pop_d.shape[1]), dtype=np.int32)

# Creates an array filled with ones, with the number of elements corresponding to the number of columns in pop_d
cut_idx = np.ones(shape=(pop_d.shape[1]), dtype=np.int32)
# Copies an array from CPU memory (using numpy) to GPU memory.
cut_idx_d = cuda.to_device(cut_idx)

minimum_cost = float('Inf') 
old_time = timer()
count = 0
count_index = 0
best_sol = 0
assign_child_1 = False
last_shuffle = 10000

while count <= generations:
    if minimum_cost <= opt: # If the objective function found is less than or equal to the defined optimal, the work is terminated.
        break

    # Creates an array of indices for all individuals in the population, then converts them into a column, which is repeated 4 times to form a matrix because there are 4 candidates.
    # Example given below for a population of 5 individuals:
    # `random_arr =
    # [[0, 0, 0, 0],
    # [1, 1, 1, 1],
    # [2, 2, 2, 2],
    # [3, 3, 3, 3],
    # [4, 4, 4, 4],
    # [5, 5, 5, 5]]`
    # Population size = 5

    random_arr = np.arange(popsize, dtype=np.int32).reshape((popsize,1))
    random_arr = np.repeat(random_arr, 4, axis=1)

    # Randomly shuffles the rows within the columns.
    random.shuffle(random_arr[:,0])
    random.shuffle(random_arr[:,1])
    random.shuffle(random_arr[:,2])
    random.shuffle(random_arr[:,3])
    # print("Random Array:")
    # print(np.sort(random_arr,axis=0))

    # Copy back to GPU memory
    random_arr_d = cuda.to_device(random_arr)

    # print("Initial Population (pop_d):")
    # print(pop_d)

    # # Check if assign_child_1 is True or False
    # if assign_child_1:
    #     print("Selecting first individual in pop_d for candidates:")
    # else:
    #     print("Selecting 4 randomly chosen individuals for candidates:")

    # print("Random Array (random_arr):")
    # print(random_arr)

    threads_per_block = (16, 16)
    blocks_per_grid_x = ceil(pop_d.shape[0] / threads_per_block[0])
    blocks_per_grid_y = ceil(pop_d.shape[1] / threads_per_block[1])
    blocks = (blocks_per_grid_x, blocks_per_grid_y)

    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()
    select_candidates[blocks, threads_per_block]\
                    (pop_d, random_arr_d, candid_d_1, candid_d_2, candid_d_3, candid_d_4, assign_child_1)

    end_event.record()
    end_event.synchronize()
    elapsed_time = cuda.event_elapsed_time(start_event, end_event)
    if count == 0:
      print(f"select_candidates execution time: {elapsed_time:.3f} ms")


    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()
    select_parents[blocks, threads_per_block]\
                (pop_d, candid_d_1, candid_d_2, candid_d_3, candid_d_4, parent_d_1, parent_d_2)
    end_event.record()
    end_event.synchronize()
    elapsed_time = cuda.event_elapsed_time(start_event, end_event)
    if count == 0:
      print(f"select_parents execution time: {elapsed_time:.3f} ms")

    # print("Selected Candidate 1:")
    # print(candid_d_1)

    # print("Selected Candidate 2:")
    # print(candid_d_2)

    # print("Selected Candidate 3:")
    # print(candid_d_3)

    # print("Selected Candidate 4:")
    # print(candid_d_4)

    # print("Parent 1:")
    # print(parent_d_1)

    # print("Parent 2:")
    # print(parent_d_2)

    ##Define fixed parents with additional rows
    # parent_d_1 = np.array([
    #     [1, 1, 12, 10, 15, 1, 11, 14, 8, 1, 7, 1, 5, 1, 4, 1, 3, 1, 6, 1, 9, 1, 13, 2, 1, 16, 1, 1, 1, 1, 1, 1, 1, 595]
    # ] * 10)  # Replicate the row 10 times

    # parent_d_2 = np.array([
    #     [1, 1, 6, 4, 12, 1, 8, 1, 3, 1, 14, 9, 1, 10, 16, 1, 5, 11, 1, 7, 1, 15, 13, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 574]
    # ] * 10)  # Replicate the row 10 times

    # # Initialize candidate arrays (required for the function call)
    # candid_d_1 = np.zeros_like(parent_d_1)
    # candid_d_2 = np.zeros_like(parent_d_1)
    # candid_d_3 = np.zeros_like(parent_d_1)
    # candid_d_4 = np.zeros_like(parent_d_1)

    # child_d_1 = np.ones_like(parent_d_1)
    # child_d_2 = np.ones_like(parent_d_2)

    threads_per_block = (16, 16)
    blocks = (ceil(candid_d_1.shape[0] / threads_per_block[0]), ceil(candid_d_1.shape[1] / threads_per_block[1]))

    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()
    number_cut_points_new[blocks, threads_per_block](candid_d_1, candid_d_2, \
                            candid_d_3, candid_d_4, parent_d_1, parent_d_2, count, min_n, max_n)
    end_event.record()
    end_event.synchronize()
    elapsed_time = cuda.event_elapsed_time(start_event, end_event)
    if count == 0:
      print(f"number_cut_points_new execution time: {elapsed_time:.3f} ms")


    warp_size = 32
    max_threads_per_block = 512

    threads_per_block = min(
        ((popsize + warp_size - 1) // warp_size) * warp_size,
        max_threads_per_block
    )

    blocks_per_grid_x = ceil(candid_d_1.shape[0] / threads_per_block)
    blocks_per_grid_y = ceil(candid_d_1.shape[1] / threads_per_block)
    blocks = (blocks_per_grid_x, blocks_per_grid_y)

    rng_states = create_xoroshiro128p_states(popsize*pop_d.shape[1], seed=random.randint(2,2*10**5))
    #rng_states = create_xoroshiro128p_states(popsize*pop_d.shape[1], seed=12345)
    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()
    add_cut_points_new[blocks, threads_per_block](candid_d_1, candid_d_2, rng_states)
    end_event.record()
    end_event.synchronize()
    elapsed_time = cuda.event_elapsed_time(start_event, end_event)
    if count == 0:
      print(f"add_cut_points_new execution time: {elapsed_time:.3f} ms")

    # print("Selected Candidate 1 sent:")
    # print(candid_d_1)

    # print("Selected Candidate 2 sent:")
    # print(candid_d_2)

    # print("Selected Parent 1 sent:")
    # print(parent_d_1)

    # print("Selected Parent 2 sent:")
    # print(parent_d_2)

    # print("Selected Child 1 sent:")
    # print(child_d_1)

    # print("Selected Child 2 sent:")
    # print(child_d_2)

    threads_per_block = (16, 16)
    blocks_per_grid_x = ceil(candid_d_1.shape[0] / threads_per_block[0])
    blocks_per_grid_y = ceil(candid_d_1.shape[1] / threads_per_block[1])
    blocks = (blocks_per_grid_x, blocks_per_grid_y)

    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()
    cross_over_gpu[blocks, threads_per_block](candid_d_1, candid_d_2, child_d_1, child_d_2, parent_d_1, parent_d_2)
    end_event.record()
    end_event.synchronize()
    elapsed_time = cuda.event_elapsed_time(start_event, end_event)
    if count == 0:
      print(f"cross_over_gpu execution time: {elapsed_time:.3f} ms")

    rng_states = create_xoroshiro128p_states(popsize*child_d_1.shape[1], seed=random.randint(2,2*10**5))
    #rng_states = create_xoroshiro128p_states(popsize*child_d_1.shape[1], seed=12345)

    threads_per_block = (16, 16) 
    blocks_per_grid_x = (child_d_1.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (child_d_1.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks = (blocks_per_grid_x, blocks_per_grid_y)
    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()
    mutate_aida[blocks, threads_per_block](rng_states, child_d_1, child_d_2)
    end_event.record()
    end_event.synchronize()
    elapsed_time = cuda.event_elapsed_time(start_event, end_event)
    if count == 0:
      print(f"mutate_aida execution time: {elapsed_time:.3f} ms")

    # mutate[(N + 256 - 1) // 256, 256](rng_states, child_d_1, child_d_2)

    threads_per_block = (16, 16)
    blocks = ( 
        (popsize + threads_per_block[0] - 1) // threads_per_block[0], 1
    )

    # Adjusting child_1 array
    find_duplicates[blocks, threads_per_block](child_d_1, r_flag)

    blocks_per_grid_x = ceil(pop_d.shape[0] / threads_per_block[0])
    blocks_per_grid_y = ceil(pop_d.shape[1] / threads_per_block[1])
    blocks = (blocks_per_grid_x, blocks_per_grid_y)

    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()
    find_missing_nodes[blocks, threads_per_block](data_d, missing_d, child_d_1)
    end_event.record()
    end_event.synchronize()
    elapsed_time = cuda.event_elapsed_time(start_event, end_event)
    if count == 0:
      print(f"find_missing_nodes execution time: {elapsed_time:.3f} ms")

    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()
    add_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, child_d_1)
    end_event.record()
    end_event.synchronize()
    elapsed_time = cuda.event_elapsed_time(start_event, end_event)
    if count == 0:
      print(f"add_missing_nodes execution time: {elapsed_time:.3f} ms")

    shift_r_flag[ceil(pop_d.shape[0] / 128), 128](r_flag, child_d_1)

    blocks_per_grid = (N + threads_per_block[0] - 1) // threads_per_block[0]
    cap_adjust[blocks_per_grid, threads_per_block](r_flag, vrp_capacity, data_d, child_d_1)

    blockspergrid_x = (pop_d.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blockspergrid_y = (pop_d.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    cleanup_r_flag[(blockspergrid_x, blockspergrid_y), threads_per_block](r_flag, child_d_1)


    # Adjusting child_2 array
    threads_per_block = (16, 16)
    blocks = (
        (popsize + threads_per_block[0] - 1) // threads_per_block[0], 1
    )
    find_duplicates[blocks, threads_per_block](child_d_2, r_flag)

    blocks_per_grid_x = ceil(pop_d.shape[0] / threads_per_block[0])
    blocks_per_grid_y = ceil(pop_d.shape[1] / threads_per_block[1])
    blocks = (blocks_per_grid_x, blocks_per_grid_y)
    find_missing_nodes[blocks, threads_per_block](data_d, missing_d, child_d_2)
    add_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, child_d_2)

    shift_r_flag[ceil(pop_d.shape[0] / 128), 128](r_flag, child_d_2)

    blocks_per_grid = (N + threads_per_block[0] - 1) // threads_per_block[0]
    cap_adjust[blocks_per_grid, threads_per_block](r_flag, vrp_capacity, data_d, child_d_2)

    blockspergrid_x = (pop_d.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blockspergrid_y = (pop_d.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    cleanup_r_flag[(blockspergrid_x, blockspergrid_y), threads_per_block](r_flag, child_d_2)

    # --------------------------------------------------------------------------
    #Performing the two-opt optimization and Calculating fitness for child_1 array

    blocks = (
        (pop_d.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
        (pop_d.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    )
    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()
    two_opt[blocks, threads_per_block](child_d_1, cost_table_d, candid_d_3)
    end_event.record()
    end_event.synchronize()
    elapsed_time = cuda.event_elapsed_time(start_event, end_event)
    if count == 0:
      print(f"two_opt execution time: {elapsed_time:.3f} ms")

    # individual = [1, 14, 77, 59, 34, 80, 12, 26, 33, 1, 23, 20, 76, 7, 70, 47, 16, 67, 1, 52, 30, 27, 13, 48, 19, 71, 38, 17, 61, 1, 18, 28, 60, 25, 64, 24, 1, 55, 51, 74, 78, 50, 50, 36, 73, 72, 79, 15, 68, 43, 1, 53, 44, 9, 40, 2, 4, 1, 54, 21, 49, 56, 37, 41, 8, 1, 3, 11, 22, 75, 62, 35, 1, 39, 63, 69, 31, 66, 6, 58, 32, 1, 57, 10, 5, 65, 42, 29, 46, 1]

    # fitness = 0
    # for i in range(len(individual) - 1):
    #     fitness += cost_table_d[individual[i] - 1, individual[i + 1] - 1]

    # print(fitness)

    fitness_gpu[(pop_d.shape[0] + 128 - 1) // 128, 128](cost_table_d, child_d_1, fitness_val_d)

    # --------------------------------------------------------------------------
    # Performing the two-opt optimization and Calculating fitness for child_2 array

    threads_per_block = (16, 16)  # 16x16 threads per block

    blocks = (
        (candid_d_3.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
        (candid_d_3.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    )

    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()
    reset_to_ones[blocks, threads_per_block](candid_d_3)
    end_event.record()
    end_event.synchronize()
    elapsed_time = cuda.event_elapsed_time(start_event, end_event)
    if count == 0:
      print(f"reset_to_ones execution time: {elapsed_time:.3f} ms")

    blocks = (
            (pop_d.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (pop_d.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        )

    two_opt[blocks, threads_per_block](child_d_2, cost_table_d, candid_d_3)

    fitness_gpu[(pop_d.shape[0] + 128 - 1) // 128, 128](cost_table_d, child_d_2, fitness_val_d)


    # check_list = set(range(1, 81))  # Define the set of elements to check
    # # Check if each row in child_d_1 contains all elements from check_list
    # for i, row in enumerate(child_d_1):
    #     row_set = set(row.tolist())  # Convert the row to a set
    #     missing_elements = check_list - row_set  # Find missing elements
    #     if missing_elements:
    #         print(f"Row {i+1} is missing the following elements: {sorted(missing_elements)}")
    #     else:
    #         print(f"Row {i+1} contains all elements from check_list.")

    # --------------------------------------------------------------------------
    # Creating the new population from parents and children
    select_bests(parent_d_1, parent_d_2, child_d_1, child_d_2, pop_d, popsize)
    # --------------------------------------------------------------------------

    # Replacing duplicates with random individuals from child_d_1
    asnumpy_pop_d = cp.asnumpy(pop_d) # copy pop_d to host, HOWEVER, it throws cudaErrorIllegalAddress in >= 800 nodes
    asnumpy_child_d_1 = cp.asnumpy(child_d_1) # copy child_d_1 to host
    repeats = 0

    x = np.unique(asnumpy_pop_d[:,1:], axis=0)
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

# Safeguard against division by zero
if count > 1:
    time_per_loop = float('{0:.4f}'.format((current_time - old_time) / (count - 1)))
else:
    time_per_loop = 0.0  # Default value for the first loop or invalid count

best_sol = cp.subtract(best_sol, cp.ones_like(best_sol))
best_sol[0] = best_sol[0] + 1

print('---------\nProblem:', sys.argv[1], ', Best known:', opt)
print('Time elapsed:', total_time, 'secs', 'Time per loop:', time_per_loop, 'secs', end = '\n---------\n')
print('Stopped at generation %d, Best cost: %d, from Generation: %d'\
      %(count-1, best_sol[-1], best_sol[0]), end = '\n---------\n')
print('Best solution:', best_sol, end = '\n---------\n')

lo = minimum_cost  # Best solution obtained
l_star = opt
percent_gap = ((lo - l_star) / l_star) * 100 if l_star != 0 else float('inf')  # Avoid division by zero
print(f"Percentage Gap (%Gap): {percent_gap:.2f}%")

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
