import numpy as np
import csv
import heapq
import time
import random
import matplotlib.pyplot as plt

class world:
    Actions = ['l','r','t','d']

    def __init__(self, dimension, transition_prob, living_reward, hole_reward, goal_reward, discount_factor, file_name):
        self.dimension = dimension
        self.transition_prob = transition_prob
        self.living_reward = living_reward
        self.hole_reward = hole_reward
        self.goal_reward = goal_reward
        self.discount_factor = discount_factor
        self.file_name = file_name
        self.grid = []
        self.new_grid = []
        self.map_val = []
        self.policy = []
        self.pq = []
        self.visited = []
        self.start_x = 0
        self.start_y = 0

    def initialize_map(self):
        file = open(self.file_name)

        csv_reader = csv.reader(file)
        line = []
        for line in csv_reader:
            self.map_val.append(line)

        file.close()

    def initialize_grid(self):
        for i in range(self.dimension):
            self.grid.append(np.array([0.001]*self.dimension))
            self.policy.append(np.array(['x']*self.dimension))

        for x in range(self.dimension):
            for y in range(self.dimension):
                if self.map_val[x][y]=='G' or self.map_val[x][y]=='H':
                    self.grid[x][y] = 0
                else:
                    i = random.randint(0,3)
                    self.policy[x][y] = world.Actions[i]
                    if self.map_val[x][y]=='S':
                        self.start_x = x
                        self.start_y = y
        self.grid = np.asarray(self.grid)
        self.policy = np.asarray(self.policy)

    def initialize_new_grid(self):
        for i in range(self.dimension):
            self.new_grid.append(np.array([0.001]*self.dimension))

        for x in range(self.dimension):
            for y in range(self.dimension):
                if self.map_val[x][y]=='G' or self.map_val[x][y]=='H':
                    self.new_grid[x][y] = 0
        self.new_grid = np.asarray(self.new_grid)

    def initialize_priority_queue(self):
        for x in range(self.dimension):
            for y in range(self.dimension):
                if self.map_val[x][y]!='G' and self.map_val[x][y]!='H':
                    heapq.heappush(self.pq, [0, [x,y]])

    def initialize_visited(self):
        for i in range(self.dimension):
            self.visited.append(np.array([False]*self.dimension))

        self.visited = np.asarray(self.visited)

    def new_reward(self,x,y):
        reward = 0
        if self.map_val[x][y]=='G':
            reward = self.goal_reward
        elif self.map_val[x][y]=='H':
            reward = self.hole_reward
        else:
            reward = self.living_reward

        return reward

    def synchronous_update(self,x,y):
        maxi = -9999999
        for action in world.Actions:
            prob = 0
            reward = 0
            sum = 0

            #left
            temp_y = max(0,y-1)
            reward = self.new_reward(x,temp_y)
            
            if action=='l':
                prob = self.transition_prob
            else:
                prob = (1-self.transition_prob)/3

            sum += prob*(reward + self.discount_factor*self.grid[x][temp_y])

            
            #right
            temp_y = min(y+1,self.dimension-1)
            reward = self.new_reward(x,temp_y)

            if action=='r':
                prob = self.transition_prob
            else:
                prob = (1-self.transition_prob)/3

            sum += prob*(reward + self.discount_factor*self.grid[x][temp_y])
            

            #top
            temp_x = max(0,x-1)
            reward = self.new_reward(temp_x,y)

            if action=='t':
                prob = self.transition_prob
            else:
                prob = (1-self.transition_prob)/3

            sum += prob*(reward + self.discount_factor*self.grid[temp_x][y])


            #down
            temp_x = min(self.dimension-1,x+1)
            reward = self.new_reward(temp_x,y)

            if action=='d':
                prob = self.transition_prob
            else:
                prob = (1-self.transition_prob)/3

            sum += prob*(reward + self.discount_factor*self.grid[temp_x][y])

            if sum > maxi:
                maxi = sum
                self.policy[x][y] = action

        self.new_grid[x][y] = maxi
        return self.new_grid[x][y] - self.grid[x][y]

    def asynchronous_update(self,x,y):
        maxi = -9999999
        for action in world.Actions:
            prob = 0
            reward = 0
            sum = 0

            #left
            temp_y = max(0,y-1)
            reward = self.new_reward(x,temp_y)
            
            if action=='l':
                prob = self.transition_prob
            else:
                prob = (1-self.transition_prob)/3

            sum += prob*(reward + self.discount_factor*self.grid[x][temp_y])

            
            #right
            temp_y = min(y+1,self.dimension-1)
            reward = self.new_reward(x,temp_y)

            if action=='r':
                prob = self.transition_prob
            else:
                prob = (1-self.transition_prob)/3

            sum += prob*(reward + self.discount_factor*self.grid[x][temp_y])
            

            #top
            temp_x = max(0,x-1)
            reward = self.new_reward(temp_x,y)

            if action=='t':
                prob = self.transition_prob
            else:
                prob = (1-self.transition_prob)/3

            sum += prob*(reward + self.discount_factor*self.grid[temp_x][y])


            #down
            temp_x = min(self.dimension-1,x+1)
            reward = self.new_reward(temp_x,y)

            if action=='d':
                prob = self.transition_prob
            else:
                prob = (1-self.transition_prob)/3

            sum += prob*(reward + self.discount_factor*self.grid[temp_x][y])

            if sum > maxi:
                maxi = sum
                self.policy[x][y] = action

        val = self.grid[x][y]
        self.grid[x][y] = maxi
        return self.grid[x][y] - val

    def prioritized_sweep(self,x,y,epsilon):
        #left
        temp_y = max(0,y-1)
        if self.map_val[x][temp_y]!='G' and self.map_val[x][temp_y]!='H':
            bellman_error = abs(self.asynchronous_update(x,temp_y))
            if bellman_error > epsilon:
                heapq.heappush(self.pq, [-1*bellman_error, [x,temp_y]])
                self.visited[x][temp_y] = False
        
        #right
        temp_y = min(y+1,self.dimension-1)
        if self.map_val[x][temp_y]!='G' and self.map_val[x][temp_y]!='H':
            bellman_error = abs(self.asynchronous_update(x,temp_y))
            if bellman_error > epsilon:
                heapq.heappush(self.pq, [-1*bellman_error, [x,temp_y]])
                self.visited[x][temp_y] = False

        #top
        temp_x = max(0,x-1)
        if self.map_val[temp_x][y]!='G' and self.map_val[temp_x][y]!='H':
            bellman_error = abs(self.asynchronous_update(temp_x,y))
            if bellman_error > epsilon:
                heapq.heappush(self.pq, [-1*bellman_error, [temp_x,y]])
                self.visited[temp_x][y] = False

        #down
        temp_x = min(x+1,self.dimension-1)
        if self.map_val[temp_x][y]!='G' and self.map_val[temp_x][y]!='H':
            bellman_error = abs(self.asynchronous_update(temp_x,y))
            if bellman_error > epsilon:
                heapq.heappush(self.pq, [-1*bellman_error, [temp_x,y]])
                self.visited[temp_x][y] = False

    def policy_evaluation(self,x,y):
        prob = 0
        reward = 0
        sum = 0

        #left
        temp_y = max(0,y-1)
        reward = self.new_reward(x,temp_y)
        
        if self.policy[x][y]=='l':
            prob = self.transition_prob
        else:
            prob = (1-self.transition_prob)/3

        sum += prob*(reward + self.discount_factor*self.grid[x][temp_y])

        
        #right
        temp_y = min(y+1,self.dimension-1)
        reward = self.new_reward(x,temp_y)

        if self.policy[x][y]=='r':
            prob = self.transition_prob
        else:
            prob = (1-self.transition_prob)/3

        sum += prob*(reward + self.discount_factor*self.grid[x][temp_y])
        

        #top
        temp_x = max(0,x-1)
        reward = self.new_reward(temp_x,y)

        if self.policy[x][y]=='t':
            prob = self.transition_prob
        else:
            prob = (1-self.transition_prob)/3

        sum += prob*(reward + self.discount_factor*self.grid[temp_x][y])


        #down
        temp_x = min(self.dimension-1,x+1)
        reward = self.new_reward(temp_x,y)

        if self.policy[x][y]=='d':
            prob = self.transition_prob
        else:
            prob = (1-self.transition_prob)/3

        sum += prob*(reward + self.discount_factor*self.grid[temp_x][y])

        val = self.grid[x][y]
        self.grid[x][y] = sum
        return self.grid[x][y] - val

    def policy_improvement(self,x,y):
        maxi = -9999999
        best_action = 'x'
        for action in world.Actions:
            prob = 0
            reward = 0
            sum = 0

            #left
            temp_y = max(0,y-1)
            reward = self.new_reward(x,temp_y)
            
            if action=='l':
                prob = self.transition_prob
            else:
                prob = (1-self.transition_prob)/3

            sum += prob*(reward + self.discount_factor*self.grid[x][temp_y])

            
            #right
            temp_y = min(y+1,self.dimension-1)
            reward = self.new_reward(x,temp_y)

            if action=='r':
                prob = self.transition_prob
            else:
                prob = (1-self.transition_prob)/3

            sum += prob*(reward + self.discount_factor*self.grid[x][temp_y])
            

            #top
            temp_x = max(0,x-1)
            reward = self.new_reward(temp_x,y)

            if action=='t':
                prob = self.transition_prob
            else:
                prob = (1-self.transition_prob)/3

            sum += prob*(reward + self.discount_factor*self.grid[temp_x][y])


            #down
            temp_x = min(self.dimension-1,x+1)
            reward = self.new_reward(temp_x,y)

            if action=='d':
                prob = self.transition_prob
            else:
                prob = (1-self.transition_prob)/3

            sum += prob*(reward + self.discount_factor*self.grid[temp_x][y])

            if sum > maxi:
                maxi = sum
                best_action = action
        
        if self.policy[x][y]==best_action:
            return True
        self.policy[x][y] = best_action
        return False

# Vanilla value iteration
a = world(50, 0.8, 0, 0, 1, 0.9, "large_map.csv")
a.initialize_map()
a.initialize_grid()
a.initialize_new_grid()
epsilon = 0.0000001
it = 1
xpoints = []
ypoints = []
while(True):
    max_residual = -9999999
    for x in range(a.dimension):
        for y in range(a.dimension):
            if a.map_val[x][y]=='G' or a.map_val[x][y]=='H':
                continue
            residual = a.synchronous_update(x,y)
            max_residual = max(max_residual, residual)

    if max_residual < epsilon*(1-a.discount_factor)/a.discount_factor:
        break
    a.grid = a.new_grid.copy()
    xpoints.append(it)
    ypoints.append(a.grid[a.start_x][a.start_y])
    it += 1

xpoints = np.asarray(xpoints)
ypoints = np.asarray(ypoints)

plt.imshow(a.new_grid)
plt.title("Vanilla Heat Map")
plt.show()

#for x in range(a.dimension):
#    for y in range(a.dimension):
#        print(a.policy[x][y], end="")
#    print("")
#For small_map, epsilon = 1.7


# Row-major sweep asynchronous value iteration
b = world(50, 0.8, 0, 0, 1, 0.9, "large_map.csv")
b.initialize_map()
b.initialize_grid()
epsilon = 0.0000001
it = 1
xpoints1 = []
ypoints1 = []
while(True):
    max_residual = -9999999
    for x in range(b.dimension):
        for y in range(b.dimension):
            if b.map_val[x][y]=='G' or b.map_val[x][y]=='H':
                continue
            residual = b.asynchronous_update(x,y)
            max_residual = max(max_residual, residual)

    if max_residual < epsilon*(1-b.discount_factor)/b.discount_factor:
        break
    xpoints1.append(it)
    ypoints1.append(b.grid[b.start_x][b.start_y])
    it += 1

xpoints1 = np.asarray(xpoints1)
ypoints1 = np.asarray(ypoints1)

#for x in range(b.dimension):
#    for y in range(b.dimension):
#        print(b.policy[x][y], end="")
#    print("")


# Prioritized sweep asynchronous value iteration
c = world(50, 0.8, 0, 0, 1, 0.9, "large_map.csv")
c.initialize_map()
c.initialize_grid()
c.initialize_priority_queue()
c.initialize_visited()
epsilon = 0.0000001
it = 1
xpoints2 = []
ypoints2 = []
while(len(c.pq)!=0):
    s = heapq.heappop(c.pq)
    x,y = s[1]
    if c.visited[x][y]==True:
        continue
    c.visited[x][y] = True
    c.prioritized_sweep(x,y,epsilon*(1-c.discount_factor)/c.discount_factor)
    xpoints2.append(it)
    ypoints2.append(c.grid[c.start_x][c.start_y])
    it += 1
    
xpoints2 = np.asarray(xpoints2)
ypoints2 = np.asarray(ypoints2)

#for x in range(c.dimension):
#    for y in range(c.dimension):
#        print(c.policy[x][y], end="")
#    print("")


# Modified Policy Iteration (Uses simplified value iteration to get an approximate evaluation of a policy)
d = world(50, 0.8, 0, 0, 1, 0.9, "large_map.csv")
d.initialize_map()
d.initialize_grid()
epsilon = 0.01
it = 1
xpoints3 = []
ypoints3 = []
while(True):
    while(True):
        max_residual = -9999999
        for x in range(d.dimension):
            for y in range(d.dimension):
                if d.map_val[x][y]=='G' or d.map_val[x][y]=='H':
                    continue
                residual = d.policy_evaluation(x,y)
                max_residual = max(max_residual,residual)

        if max_residual < epsilon*(1-d.discount_factor)/d.discount_factor:
            break

    converged = True
    for x in range(d.dimension):
        for y in range(d.dimension):
            if d.map_val[x][y]=='G' or d.map_val[x][y]=='H':
                continue
            check = d.policy_improvement(x,y)
            if converged==True and check==False:
                converged = False

    if converged==True:
        break
    xpoints3.append(it)
    ypoints3.append(d.grid[d.start_x][d.start_y])
    it += 1

xpoints3 = np.asarray(xpoints3)
ypoints3 = np.asarray(ypoints3)

#for x in range(d.dimension):
    #for y in range(d.dimension):
    #    print(d.policy[x][y], end="")
    #print("")


#plt.plot(xpoints, ypoints, color='r', label='vanilla')
#plt.plot(xpoints1, ypoints1, color='g', label='row-major sweep')
#plt.plot(xpoints2, ypoints2, color='b', label='prioritized sweep')
#plt.plot(xpoints3, ypoints3, color='y', label='policy iteration')
#plt.show()


# For large_map.csv : 
#       vanilla - time = 3.53 sec, iterations = 134
#       row-major sweep - time = 3.07 sec, iterations = 117 
#       prioritized sweep - time = 4.22 sec, iterations = 58696
#       policy iteration - time = 3.34 sec, iterations = 71

#For small_map.csv : 
#       vanilla - time = 2.96 ms, iterations = 7
#       row-major sweep - time = 2.89 ms, iterations = 7 
#       prioritized sweep - time = 2.96 ms, iterations = 25
#       policy iteration - time = 2.9 ms, iterations = 3

