import yaml
import numpy as np
from imageio.v2 import imread
import heapq

import matplotlib.pyplot as plt

import json

def save_path_to_json(path, filename):
    # Convert any non-serializable items (like tuples) in the path to serializable formats (like lists)
    serializable_path = [list(point) for point in path]
    with open(filename, 'w') as json_file:
        json.dump(serializable_path, json_file, indent=4)

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

    print("normalized_map shape", normalized_map.shape)
    # print("normalized_map", normalized_map[100])
    # for i in range(600):
    #     print(normalized_map[400][i])
    # print("free_thresh", free_thresh)
    # print("occupied_thresh", occupied_thresh)
    
    # Occupancy grid initialization
    occupancy_grid = np.zeros_like(normalized_map, dtype=np.int8)
    # Free space
    occupancy_grid[normalized_map >= free_thresh] = -1
    # Occupied space
    occupancy_grid[normalized_map <= occupied_thresh] = 1

    # for i in range(600):
    #     print(occupancy_grid[400][i])
    return occupancy_grid

# Function to check line of sight using Bresenham's Line Algorithm
def line_of_sight(occupancy_grid, start, end):
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if occupancy_grid[x0, y0] == 1:  # Check for obstacle
            return False
        if (x0, y0) == (x1, y1):
            return True
        e2 = 2 * err
        if e2 >= dy:
            if x0 == x1:
                break
            err += dy
            x0 += sx
        if e2 <= dx:
            if y0 == y1:
                break
            err += dx
            y0 += sy

# Theta* algorithm for path finding
def theta_star(occupancy_grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    g_cost = {start: 0}
    f_cost = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == start:
            came_from[current] = None
        elif current == goal:
            break  # Found the goal

        for neighbor in get_neighbors(current, occupancy_grid):
            tentative_g_cost = g_cost[current] + distance(current, neighbor)
            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                if came_from[current] is None or line_of_sight(occupancy_grid, came_from[current], neighbor):
                    came_from[neighbor] = current
                else:
                    came_from[neighbor] = came_from[current]
                g_cost[neighbor] = tentative_g_cost
                f_cost[neighbor] = tentative_g_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_cost[neighbor], neighbor))
    return reconstruct_thetastar_path(came_from, goal)

# Heuristic function for path scoring (Euclidean distance)
def heuristic(start, goal):
    return np.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)

# Function to get neighbors for the current node
def get_neighbors(node, grid):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), # 4-directional
                  (-1, -1), (-1, 1), (1, -1), (1, 1)] # Diagonals
    neighbors = []
    for dx, dy in directions:
        x, y = node[0] + dx, node[1] + dy
        if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x, y] != 1:
            neighbors.append((x, y))
    return neighbors

# Function to calculate distance between two points (Euclidean distance)
def distance(start, end):
    return np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

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
            return reconstruct_dijkstra_path(came_from, current)

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
def reconstruct_thetastar_path(came_from, current):
    path = []
    while current is not None:  # Skip if current is None
        path.append(current)
        current = came_from[current]
    path.reverse()  # Reverse the path to start->goal
    return path


def reconstruct_dijkstra_path(came_from, current):
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
    # path_map = dijkstra(occupancy_grid, start_map, goal_map)
    path_map = theta_star(occupancy_grid, start_map, goal_map)
    print("path_map: ", path_map)
    
    # Convert the path back to world coordinates
    path_world = [map_to_world(pose, origin, resolution) for pose in path_map]
    
    return path_world


if __name__ == "__main__":
    config_file_path = '302_3f_room_and_hallway_slam.yaml'
    start_pose = (-20, -10) # Example start position
    goal_pose = (0, 0) # Example goal position
    path = find_path(config_file_path, start_pose, goal_pose)
    file_name = f'path_{start_pose[0]}_{start_pose[1]}_to_{goal_pose[0]}_{goal_pose[0]}.json'
    save_path_to_json(path, file_name)

    config = read_yaml_config(config_file_path)
    origin = config['origin'][:2]  # We only need the X,Y components
    resolution = config['resolution']
    map_img = read_pgm_map(config['image'])
    plot_path_on_map(map_img, path, origin, resolution)
    print(path)