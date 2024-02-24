import yaml
import numpy as np
from imageio.v2 import imread
import heapq

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_path_on_map(map_img, path):
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the size to fit your map dimensions
    # Show the map image
    ax.imshow(map_img, cmap='gray', origin='lower')

    # Check if the path is valid and has more than one point
    if path and len(path) > 1:
        # Extract X and Y coordinates from the path
        y, x = zip(*path)
        ax.plot(x, y, 'r-', linewidth=2)  # Red line for the path
        ax.plot(x[0], y[0], 'go')  # Green dot for the start
        ax.plot(x[-1], y[-1], 'bo')  # Blue dot for the goal
    else:
        print("Invalid or too short path provided for plotting")

    ax.set_title("Path Planning")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('image')  # Ensures the aspect ratio is equal and axes are scaled correctly
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

# Main function to read the config and map, and find the path
def find_path(config_file, start, goal):
    config = read_yaml_config(config_file)
    map_img = read_pgm_map(config['image'])
    occupancy_grid = map_to_occupancy_grid(map_img, config['occupied_thresh'], config['free_thresh'])
    path = dijkstra(occupancy_grid, start, goal)
    return path

if __name__ == "__main__":
    config_file_path = '302_3f_room_and_hallway_slam.yaml'
    start_pose = (100, 100) # Example start position
    goal_pose = (500, 500) # Example goal position
    path = find_path(config_file_path, start_pose, goal_pose)

    config = read_yaml_config(config_file_path)
    map_img = read_pgm_map(config['image'])
    plot_path_on_map(map_img, path)
    print(path)