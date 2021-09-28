# Bayes Filter for 1D Localization Problem
# Eko Rudiawan Jamzuri @ 2021

import matplotlib.pyplot as plt
import numpy as np
import time
from filterpy.discrete_bayes import predict, update

def draw_robot(ax, x_pos, y_pos=5):
    robot = ax.scatter([x_pos], [y_pos], s=200)
    return robot

def move_robot(robot, x_vel):
    current_pos = robot.get_offsets()
    x = current_pos[0,0] + x_vel
    y = current_pos[0,1]
    robot.set_offsets([[x, y]])

def draw_doors(ax, list_x_pos, y_pos=30):
    list_doors = []
    for x_pos in list_x_pos:
        door = ax.scatter([x_pos], [y_pos], s=500, marker='s', color='r')
        list_doors.append(door)
    return list_doors

def check_sensor(robot, list_doors):
    robot_pos = robot.get_offsets()
    hit_obstacle = False
    for door in list_doors:
        door_pos = door.get_offsets()
        if robot_pos[0,0] == door_pos[0,0]:
            hit_obstacle = True
            break
        else:
            hit_obstacle = False
    return hit_obstacle

def main():
    # for delay
    dt = 0.1
    # vx, assume 1 unit per second
    vx = 1
    plt.ion()
    fig, ax = plt.subplots(3)
    ax[0].set_xlim(0, 100)
    ax[0].set_ylim(0, 50)
    ax[1].set_xlim(0, 100)
    ax[1].set_ylim(0, 1)
    ax[2].set_xlim(0, 100)
    ax[2].set_ylim(0, 1)

    # Set initial belief and likelihood uniformly
    belief = np.array([1/100]*100)
    likelihood = np.array([1/100]*100)
    line_belief = ax[1].plot(np.arange(belief.shape[0]), belief)
    line_likelihood = ax[2].plot(np.arange(likelihood.shape[0]), likelihood)
    
    # Set initial robot location at x=0
    robot = draw_robot(ax[0], 0)

    # Set door location
    door_locations = [10, 25, 50]
    list_doors = draw_doors(ax[0], door_locations)

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(dt)
    for i in range(100):
        move_robot(robot, vx)

        # Prediction step
        belief_hat = predict(belief, offset=vx, kernel=[0.0, 1.0, 0.0])
        
        # Observation from sensor
        if check_sensor(robot, list_doors):
            # Update belief
            print("Detect door")
            likelihood = np.zeros(100)
            likelihood[door_locations] = 1/len(door_locations)
        else:
            print("Detect no door")
            likelihood = np.ones(100)
            likelihood[door_locations] = 0
            likelihood /= 100-len(door_locations)
        print("Likelihood ", likelihood)

        # Correction step
        belief = update(likelihood, belief_hat)
        line_belief[0].set_ydata(belief)
        line_likelihood[0].set_ydata(likelihood)
        fig.canvas.draw()
        fig.canvas.flush_events()
        robot_pos = robot.get_offsets()
        print("Predicted location ", np.argmax(belief))
        print("Actual location ", robot_pos[0,0])
        print("Error ", np.argmax(belief)-robot_pos[0,0])
        print("==============================")
        time.sleep(dt)

if __name__ == "__main__":
    main()