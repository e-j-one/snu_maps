from path_generator import find_path, read_yaml_config, read_pgm_map
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_planned_paths_on_map(map_img, planned_paths, origin, resolution):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(map_img, cmap='gray', origin='lower')
    
    # Convert path to array for easier manipulation
    for plan in planned_paths:
        path = np.array(plan["waypoints"])

        if path.size > 0:
            # Scale the path according to the resolution and shift by the origin
            path_scaled = (path - np.array(origin[:2])) / resolution
            x, y = path_scaled.T
            y = map_img.shape[0] - 1 - y
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


def save_data_to_json(data, filename):
    # Convert any non-serializable items (like tuples) in the path to serializable formats (like lists)
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        
if __name__ == "__main__":

    planned_paths = []

    start_poses = [(-13, 8), (-1, 9), (8, 8), (15, 8), (48, -1), (33, -8), (20, -8), (-25, -8)]
    goal_poses = [(-13, 8), (-1, 9), (8, 8), (15, 8), (48, -1), (33, -8), (20, -8), (-25, -8)]

    map_name = '302_3f'
    costmap_config_file_path = '302_3f_room_and_hallway_slam_adj_costmap.yaml'

    for start_idx in range(len(start_poses)):
        start_pose = start_poses[start_idx]
        for goal_idx in range(len(goal_poses)):
            goal_pose = goal_poses[goal_idx]
            if start_pose == goal_pose:
                continue
            print(f'generate {start_idx}-({start_pose}) to {goal_idx}-({goal_pose}) {start_idx*len(goal_poses)+goal_idx+1}/{len(start_poses)*len(goal_poses)}')
            path = find_path(costmap_config_file_path, start_pose, goal_pose, True)
            planned_paths.append({
                "waypoints": path,
                "start": start_idx,
                "end": goal_idx
            })
    
    file_name = f'paths/{map_name}/traj_candidate_{len(start_poses)}x{len(goal_poses)}.json'
    save_data_to_json(planned_paths, file_name)

    gridmap_config_file_path = '302_3f_room_and_hallway_slam_adj.yaml'
    config = read_yaml_config(gridmap_config_file_path)
    origin = config['origin'][:2]  # We only need the X,Y components
    resolution = config['resolution']
    map_img = read_pgm_map(config['image'])
    plot_planned_paths_on_map(map_img, planned_paths, origin, resolution)
