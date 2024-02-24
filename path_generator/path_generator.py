import yaml
import numpy as np
from imageio.v2 import imread
import heapq

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Function to convert world coordinates to map coordinates
def world_to_map(pose, origin, resolution):
    map_x = int((pose[0] - origin[0]) / resolution)
    map_y = int((pose[1] - origin[1]) / resolution)
    return (map_x, map_y)

# Function to convert map coordinates to world coordinates
def map_to_world(map_coords, origin, resolution):
    world_x = (map_coords[0] * resolution) + origin[0]
    world_y = (map_coords[1] * resolution) + origin[1]
    return (world_x, world_y)

# Modify the plot function to use world coordinates
def plot_path_on_map(map_img, path, origin, resolution):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(map_img, cmap='gray', origin='lower')
    
    # Convert path to array for easier manipulation
    path = np.array(path)
    
    if path.size > 0:
        # Scale the path according to the resolution and shift by the origin
        path_scaled = (path - np.array(origin[:2])) / resolution
        y, x = path_scaled.T
        ax.plot(x, y, 'r-', linewidth=2)
        ax.plot(x[0], y[0], 'go')  # Start in green
        ax.plot(x[-1], y[-1], 'bo')  # Goal in blue
    else:
        print("Invalid or too short path provided for plotting")

    ax.set_title("Path Planning")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis('equal')
    plt.show()

# Function to read YAML configuration file
def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to read PGM map file
def read_pgm_map(file_path):
    return imread(file_path, pilmode='L')

# Function to convert map pixels to occupancy grid
def map_to_occupancy_grid(map_img, occupied_thresh, free_thresh):
    # Normalize pixel values to [0, 1]
    normalized_map = map_img / 255.0
    # Occupancy grid initialization
    occupancy_grid = np.zeros_like(normalized_map, dtype=np.int8)
    # Free space
    occupancy_grid[normalized_map >= free_thresh] = -1
    # Occupied space
    occupancy_grid[normalized_map <= occupied_thresh] = 1
    return occupancy_grid

# Dijkstra's algorithm for path finding
def dijkstra(occupancy_grid, start, goal):
    # Priority queue for open set
    open_set = [(0, start)]
    # Dictionary to store the cost to reach all visited nodes
    g_cost = {start: 0}
    # Dictionary to store the path
    came_from = {}
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] # 4-directional movement

    while open_set:
        current_cost, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < occupancy_grid.shape[0] and 0 <= neighbor[1] < occupancy_grid.shape[1]:
                if occupancy_grid[neighbor] == 1: # Check for obstacle
                    continue
                new_cost = g_cost[current] + 1
                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    priority = new_cost
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current
    return []

# Function to reconstruct path from came_from dictionary
def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current) # Add the start position
    path.reverse() # Reverse the path to start->goal
    return path


# Modify the main function to convert the start and goal poses
def find_path(config_file, start_pose, goal_pose):
    config = read_yaml_config(config_file)
    origin = config['origin'][:2]  # We only need the X,Y components
    resolution = config['resolution']
    
    map_img = read_pgm_map(config['image'])
    occupancy_grid = map_to_occupancy_grid(map_img, config['occupied_thresh'], config['free_thresh'])
    
    # Convert world poses to map coordinates
    start_map = world_to_map(start_pose, origin, resolution)
    goal_map = world_to_map(goal_pose, origin, resolution)

    # Find the path in map coordinates
    path_map = dijkstra(occupancy_grid, start_map, goal_map)
    
    # Convert the path back to world coordinates
    path_world = [map_to_world(pose, origin, resolution) for pose in path_map]
    
    return path_world


if __name__ == "__main__":
    config_file_path = '302_3f_room_and_hallway_slam.yaml'
    start_pose = (-20, -10) # Example start position
    goal_pose = (3, 3) # Example goal position
    path = find_path(config_file_path, start_pose, goal_pose)

    config = read_yaml_config(config_file_path)
    origin = config['origin'][:2]  # We only need the X,Y components
    resolution = config['resolution']
    map_img = read_pgm_map(config['image'])
    plot_path_on_map(map_img, path, origin, resolution)
    print(path)