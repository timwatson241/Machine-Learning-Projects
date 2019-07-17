
import numpy as np

# This implementation searches for the goal during the exploration phase and stops once it has found it before proceeding to the second run
class Robot(object):
    def __init__(self, maze_dim, number):
        
        # Record the maze dimensions
        self.dimensions = maze_dim
        
        # Initialize the wall map. Start with no walls in each square 
        self.wall_map = [[15 for a in range(maze_dim)] for b in range(maze_dim)]

        # Initiate an array to keep track of which cells contain walls that have been updated
        self.updated_the_walls = [[0 for a in range(maze_dim)] for b in range(maze_dim)]
        
        # Initiate an array to keep track of which cells that the robot has visited
        self.visited = [[0 for a in range(maze_dim)] for b in range(maze_dim)]
        
        # Define the goal cell for each maze
        if number == 1:
            self.goal = [6, 6]
        elif number == 2:
            self.goal = [6, 7]
        elif number == 3:
            self.goal = [8, 7]
        else:
            self.goal = [self.dimensions/2 - 1, self.dimensions/2]
        
        # Define the possible moves (West, South, East, North)
        self.moves = [[-1,0],
                    [0,-1],
                    [1,0],
                    [0,1]]
        
        # Define the initial robot settings
        self.location = [0, 0]
        self.heading = 'up'                    
        self.found_goal = False
        self.ready_to_reset = False
        self.first_run_complete = False
        
        # Record that the robot has visited its start location
        self.visited[self.location[0]][self.location[1]] = 1
    
    # This function is to verify whether a cell location falls within the maze
    def isxy_inmaze(self, point):
        if int(point[0]) >=0 and int(point[0]) < self.dimensions and int(point[1]) >=0 and int(point[1]) < self.dimensions:
            return True
        else:
            return False

    # This block of code is for the flood fill algorithm. It takes as input any cell and calculates the distance to that cell from every other cell based on our knowledge of walls and then returns an array called dist_to_g
    def flood_algorithm(self,x,y):
                
        # Initialize the distance map. Start with all distances = 99        
        self.dist_to_g = [[99 for i in range(self.dimensions)] for j in range(self.dimensions)]
        
        # Set the distance from the target cell to 0
        self.dist_to_g[x][y] = 0

        # Initialize an array to keep track of which cells we have updated in this round of the flood fill algorithm
        updated = [[0 for a in range(self.dimensions)] for b in range(self.dimensions)]
        updated[x][y] = 1
        
        # Create a list containing all cells we have already updated
        updated_cells = [[x,y]]

        while len(updated_cells)>0:
            # Choose a cell we have updated already
            now = updated_cells.pop(0)
            # Convert the wall_map number for the cell to a binary number
            bin_numb = np.binary_repr(self.wall_map[now[0]][now[1]],width=4)
            # Go through the binary number, digit by digit
            for i in range(len(bin_numb)):
                # If no wall exists on that side (i.e. if the digit is a 1)
                if int(bin_numb[i]) == 1:
                    # Add the cell on that side into the next_cell list
                    next_cell = [(now[0]+self.moves[i][0]),(now[1]+self.moves[i][1])]
                    # Check if the cell that we are looking at exists (i.e. is in the grid)
                    if self.isxy_inmaze([next_cell[0],next_cell[1]]):
                        # If we haven't already updated this cell
                        if (updated[next_cell[0]][next_cell[1]] != 1):
                            # Set the distance to go in that cell to 1 + the distance to go of the cell we are currently in
                            self.dist_to_g[next_cell[0]][next_cell[1]] = self.dist_to_g[now[0]][now[1]] + 1
                            # Add this new cell location to the updated_cells
                            updated_cells.append(next_cell)
                            # Record that we have updated the new cell
                            updated[next_cell[0]][next_cell[1]] = 1

    # this function takes the direction and the values from the sensor and updates the wall_map with walls corresponding to those sensor measurments
    def look_for_walls(self, direction, sense):
    
                            
        price = {'up': [8,1,2],
                            'down': [2,4,8],
                            'left': [4,8,1],
                            'right': [1,2,4]
                            }
        
        h = {'up': [-1,0,1,0,1,0,'down',1,0],
                            'down': [1,0,-1,0,-1,0,'up',-1,0],
                            'left': [0,-1,0,-1,0,1,'right',0,1],
                            'right':[0,1,0,1,0,-1,'left',0,-1]
                            }
                            
        index = {'up': [0,3,2],
                            'down': [2,1,0],
                            'left': [1,0,3],
                            'right':  [3,2,1]}

        # Save location of the grid cell in which a wall is being sensed
        location_left_sensor = [self.location[0]+h[direction][0]*sense[0],self.location[1]+h[direction][3]*sense[0]]
        location_forward_sensor = [self.location[0]+h[direction][1]*sense[1],self.location[1]+h[direction][4]*sense[1]]
        location_right_sensor = [self.location[0]+h[direction][2]*sense[2],self.location[1]+h[direction][5]*sense[2]]
        
        # Save the wall_map value of the grid cells being sensed
        left_sensed_wall = self.wall_map[self.location[0]+h[direction][0]*sense[0]][self.location[1]+h[direction][3]*sense[0]]
        forward_sensed_wall = self.wall_map[self.location[0]+h[direction][1]*sense[1]][self.location[1]+h[direction][4]*sense[1]]
        right_sensed_wall = self.wall_map[self.location[0]+h[direction][2]*sense[2]][self.location[1]+h[direction][5]*sense[2]]

        # Save the location of the grid cell next to the one we are sensing in the direction of sensing
        next_left = [self.location[0]+h[direction][0]*sense[0]-h[direction][7], self.location[1]+h[direction][3]*sense[0]-h[direction][8]]
        next_forwards = [self.location[0]+h[direction][1]*sense[1]-h[direction][8],self.location[1]+h[direction][4]*sense[1]+h[direction][7]]
        next_right = [self.location[0]+h[direction][2]*sense[2]+h[direction][7],self.location[1]+h[direction][5]*sense[2]+h[direction][8]]

        # Here, we update the wall that we sense to our left
        # First convert the wall number to binary and check if there is a wall in the appropriate position already
        if int(np.binary_repr(left_sensed_wall, width=4)[index[direction][0]]) == 1:
            # If no wall, adjust the wall number to add the appropriate wall
            self.wall_map[location_left_sensor[0]][location_left_sensor[1]] -= price[direction][0]
            # Keep track of the fact we have updated the cell
            self.updated_the_walls[location_left_sensor[0]][location_left_sensor[1]] = 1
            # Check to see if the next cell in the direction of sensing actually exists
            if self.isxy_inmaze(next_left):
                # Add the appropriate wall (opposite compared to above)
                self.wall_map[next_left[0]][next_left[1]] -= price[h[direction][6]][0]
                # Record that we know something about that cell
                self.updated_the_walls[next_left[0]][next_left[1]] = 1

        # Here, we update the wall that we sense in front
        if int(np.binary_repr(forward_sensed_wall, width=4)[index[direction][1]]) == 1:
            self.wall_map[location_forward_sensor[0]][location_forward_sensor[1]] -= price[direction][1]
            self.updated_the_walls[location_forward_sensor[0]][location_forward_sensor[1]] = 1
            if self.isxy_inmaze(next_forwards):
                self.wall_map[next_forwards[0]][next_forwards[1]] -= price[h[direction][6]][1]
                self.updated_the_walls[next_forwards[0]][next_forwards[1]] = 1

        # Here, we update the wall that we sense to the right
        if int(np.binary_repr(right_sensed_wall, width=4)[index[direction][2]]) == 1:
            self.wall_map[location_right_sensor[0]][location_right_sensor[1]] -= price[direction][2]
            self.updated_the_walls[location_right_sensor[0]][location_right_sensor[1]] = 1
            if self.isxy_inmaze(next_right):
                self.wall_map[next_right[0]][next_right[1]] -= price[h[direction][6]][2]
                self.updated_the_walls[next_right[0]][next_right[1]] = 1

    # This function tells the robot where to move next by returning movement and rotation information
    def next_move(self, sensors):
        
        # Indicate that we are at the start of a new move
        move_complete = False
        
        ## Check to see if we have reached the goal or not
        if self.location == self.goal:
            self.found_goal = True
    
        # Update the wall_map array
        self.look_for_walls(self.heading, sensors)
        
        # Keep track of the cells we know nothing about         
        no_knowledge = []
        
        # Initialize an array to keep track of the cells that we know something about
        info_on =[[0 for a in range(self.dimensions)] for b in range(self.dimensions)]
        
        # If we have visited or updated the walls of a cell, record that we know something about that cell
        for a in range(self.dimensions):
                for b in range(self.dimensions):
                    if int(self.visited[a][b]) == 1 or int(self.updated_the_walls[a][b]) == 1:
                        info_on[a][b] += 1
                        
        # If we have not found the goal, perform the flood fill algorithm
        if self.found_goal == False:
            self.flood_algorithm(self.goal[0], self.goal[1])
        # If we have found the goal, block all cell that we know nothing about
        elif self.found_goal == True and self.ready_to_reset == False:
            for a in range(self.dimensions):
                for b in range(self.dimensions):
                    if info_on[a][b] == 0:
                        # Keep track of the cells we know nothing about
                        no_knowledge.append([a,b])
            if len(no_knowledge) > 0:
                for item in no_knowledge:
                    # Block off all cells we have no knowledge about
                    self.wall_map[item[0]][item[1]] = 0
                    # Now add adjacent walls in the adjacent cells
                    for a in range(len(self.wall_map)):
                        for b in range(len(self.wall_map)):
                            # If we haven't been there (if we had, we would know all about it and the walls would be there already)
                            if self.visited[a][b] == 0 :
                                if self.isxy_inmaze([a-1,b]):
                                    # Add the appropriate wall if the appropriate wall doesnt already exist
                                    if int(np.binary_repr(self.wall_map[a-1][b],width=4)[2]) == 1:
                                        self.wall_map[a-1][b] -= 2
                                if self.isxy_inmaze([a,b-1]):
                                    if int(np.binary_repr(self.wall_map[a][b-1],width=4)[3]) == 1:
                                        self.wall_map[a][b-1] -= 1
                                if self.isxy_inmaze([a+1,b]):
                                    if int(np.binary_repr(self.wall_map[a+1][b],width=4)[0]) == 1:
                                        self.wall_map[a+1][b] -= 8
                                if self.isxy_inmaze([a,b+1]):
                                    if int(np.binary_repr(self.wall_map[a][b+1],width=4)[1]) == 1:
                                        self.wall_map[a][b+1] -= 4
                    ready_to_reset = True 
                             
            # Perform the necessary task to reset the robot
            if self.first_run_complete==False:
                print "Reset in Progress"
                
                rotation = 'Reset'
                movement = 'Reset'
                
                # Reset Robot
                the_move = [0,0]
                self.heading = 'up'

                print "Final distances:"
                for item in self.dist_to_g:
                    print item
                print "Final wall map:"
                for item in self.wall_map:
                    print item
                print "The_move set"
                
                # Reset distances to goal
                self.flood_algorithm(self.goal[0], self.goal[1])
                
                # Raise flag to indicate that the first run is complete
                self.first_run_complete = True
      
        ######################################################################################      
        # This next block of code controls the movement implementation for the EXPLORATORY RUN
        
        # Keep track of the potential moves
        move_list = []
        
        # Keep track of the distance to goal for the cells reached by the potential moves
        dist_list = []
        
        # Only do this if we haven't already completed the first run
        if self.first_run_complete == False:
            # Convert wall number in current location to a binary number
            binary = np.binary_repr(self.wall_map[self.location[0]][self.location[1]], width=4)
            # Go through each direction
            for a in range(4):
                # If no wall exists in that direction
                if int(binary[a]) == 1:
                    # Save the cell one away in that direction as a possible next move
                    new_loc = [self.location[0]+self.moves[a][0],self.location[1]+self.moves[a][1]]
                    # Check if the possible move exists
                    if self.isxy_inmaze(new_loc):
                        # Save possible cells we can move to
                        move_list.append([new_loc[0],new_loc[1]])
                        # Keep track of the distance to goal of these possible cells
                        dist_list.append(self.dist_to_g[new_loc[0]][new_loc[1]])
            if len(move_list) > 0:
                # Pick the move that yields the smallest distance to goal
                chosen_move = move_list[np.argmin(dist_list)]
                
            # Determine the change along each direction     
            diff_x = chosen_move[0]-self.location[0]
            diff_y = chosen_move[1]-self.location[1]
           
            # Determine rotation and movement based on the chosen next cell
            if self.heading == 'up':
                    if diff_x == 0 and diff_y == 1:
                        rotation = 0
                        movement = 1
                    if diff_x == 1 and diff_y == 0:
                        rotation = 90
                        movement = 1
                        self.heading = 'right'
                    if diff_x == 0 and diff_y == -1:
                        rotation = 0
                        movement = -1
                        self.heading = 'up'  
                    if diff_x == -1 and diff_y == 0:
                        rotation = -90
                        movement = 1
                        self.heading = 'left'

            elif self.heading == 'down':
                    if diff_x == 0 and diff_y == 1:
                        rotation = 0
                        movement = -1
                        self.heading = 'down'
                    if diff_x == 1 and diff_y == 0:
                        rotation = -90
                        movement = 1
                        self.heading = 'right'
                    if diff_x == 0 and diff_y == -1:
                        rotation = 0
                        movement = 1
                    if diff_x == -1 and diff_y == 0:
                        rotation = 90
                        movement = 1
                        self.heading = 'left'

            elif self.heading == 'left':

                    if diff_x == 0 and diff_y == 1:
                        rotation = 90
                        movement = 1
                        self.heading = 'up'
                    if diff_x == 1 and diff_y == 0:
                        rotation = 0
                        movement = -1
                        self.heading = 'left'
                    if diff_x == 0 and diff_y == -1:
                        rotation = -90
                        movement = 1
                        self.heading = 'down'
                    if diff_x == -1 and diff_y == 0:
                        rotation = 0
                        movement = 1

            elif self.heading == 'right':

                    if diff_x == 0 and diff_y == 1:
                        rotation = -90
                        movement = 1
                        self.heading = 'up'
                    if diff_x == 1 and diff_y == 0:
                        rotation = 0
                        movement = 1
                    if diff_x == 0 and diff_y == -1:
                        rotation = 90
                        movement = 1
                        self.heading = 'down'
                    if diff_x == -1 and diff_y == 0:
                        rotation = 0
                        movement = -1
                        self.heading = 'right'

        #################################################################################      
        # This next block of code controls the movement implementation for the SECOND RUN
        
        # Only perform this part if the first run is complete
        if self.first_run_complete == True:

            # Define possible moves for each heading
            moves = { 'up': [[-1,0],
                                     [0,1],
                                     [1,0],
                                     ],
                               'down': [[1,0],
                                       [0,-1],
                                       [-1,0]
                                       ],
                               'left': [[0,-1],
                                       [-1,0],
                                       [0,1],
                                       ],
                               'right': [[0,1],
                                        [1,0],
                                        [0,-1],
                                        ],
                                }

            # Keep track of the potential cells to move to
            potential_move_list = []
            
            # If we see a distance of more than 3, change the distance to 3 since we can only move 3 anyways
            for i in range(len(sensors)):
                if sensors[i] > 3:
                    sensors[i] = 3
            
            # Iterate through the sensor directions 0,1 and 2
            for i in range(3):
                # Set a flag to indicate we haven't yet found a possible move in that direction
                found = False
                # Iterate from sensor reading to 0
                for a in reversed(range(0,sensors[i]+1)):
                    # Only consider directions in which the sensor does not show 0
                    if sensors[i] != 0:
                        # Check to see if we have found a possible move in that direction
                        if found == False:
                            # Save the cell that is (a) moves away in the (i) direction
                            check_point = [self.location[0]+ a * moves[self.heading][i][0],self.location[1]+ a * moves[self.heading][i][1]]
                            # Check if every step brings us closer and check to make sure the point is in the maze
                            # Do not consider self.moves with a distance of 0
                            if a != 0:
                                if self.dist_to_g[check_point[0]][check_point[1]] + a == self.dist_to_g[self.location[0]][self.location[1]] and self.isxy_inmaze([check_point[0],check_point[1]]) == True:
                                    # Add the move to the list of possible self.moves
                                    potential_move_list.append([check_point[0],check_point[1],a])
                                    # Flag to say that we found a move in that direction so that we can move on to looking at the next direction
                                    found = True
            
            
            if len(potential_move_list) >0 and self.location != self.goal:
                # Sort the potential_move_list starting with the moves with the lowest distance
                potential_move_list.sort(key=lambda x: int(x[2]))
                # Reverse the list
                potential_move_list.reverse()
                # Pick a move with the highest distance
                the_move = potential_move_list.pop(0)
                 
                # Now we need to return the rotation and movement based on the move selected
                # Calculate the distance to move in each direction
                dx = the_move[0]-self.location[0]
                dy = the_move[1]-self.location[1]
                
                # Set default rotation              
                rotation = 0
                
                if self.heading == 'up':             
                    if dx == 0 and dy > 0:
                        movement = the_move[2]
                    elif dx > 0 and dy == 0:
                        rotation = 90
                        movement = the_move[2]
                        self.heading = 'right'
                    elif dx < 0 and dy == 0:
                        rotation = -90
                        movement = the_move[2]
                        self.heading = 'left'
            
                elif self.heading == 'down': 
                    if dx > 0 and dy == 0:
                        rotation = -90
                        movement = the_move[2]
                        self.heading = 'right'                
                    elif dx == 0 and dy < 0:
                        movement = the_move[2]                   
                    elif dx < 0 and dy == 0:
                        rotation = 90
                        movement = the_move[2]
                        self.heading = 'left'   
                    
                elif self.heading == 'left':
                    if dx == 0 and dy > 0:
                        rotation = 90
                        movement = the_move[2]
                        self.heading = 'up'                
                    elif dx == 0 and dy < 0:
                        rotation = -90
                        movement = the_move[2]
                        self.heading = 'down'                   
                    elif dx < 0 and dy == 0:
                        movement = the_move[2]
                  

                elif self.heading == 'right' :              
                    if dx == 0 and dy > 0:
                        rotation = -90
                        movement = the_move[2]
                        self.heading = 'up'               
                    elif dx > 0 and dy == 0:
                        movement = the_move[2]               
                    elif dx == 0 and dy < 0:
                        rotation = 90
                        movement = the_move[2]
                        self.heading = 'down'

        # This is where we actually make the robot move
        if self.first_run_complete == False:
            self.location = chosen_move
        else:
            self.location = the_move
            
        # Record that we have visited the current location
        self.visited[self.location[0]][self.location[1]] = 1        
        
        # Return the movement and rotation specifications
        return rotation, movement

