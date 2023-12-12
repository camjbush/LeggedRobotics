# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

""" Run CPG """
import time
import numpy as np
import matplotlib as plt

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

def find_local_minima(arr):
    minima_indices = []

    # Check each element in the array except the first and last
    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            minima_indices.append(i)

    return minima_indices

def calculate_COT(torque_matrix, joint_matrix, forward_velocity, mass, gravity, time_step):
    num_motors, num_time_steps = torque_matrix.shape
    power = np.zeros([num_motors, num_time_steps])
    cot = np.zeros([num_motors, num_time_steps])
    average_cot = 0 
    for i in range(num_motors):
        torque_i = torque_matrix[i, :]
        joint_i = joint_matrix[i, :]
        joint_i = np.concatenate([joint_i, [joint_i[-1]]])
        # Adjust the shape of joint_i to match torque_i
        power[i,:] = np.abs((torque_i*np.diff(joint_i))/time_step)
        cot[i,:] = power[i,:]/(mass*gravity*forward_velocity)

           
    average_cot = np.average(cot)
    return average_cot

ADD_CARTESIAN_PD = True
PLOT = False
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=False,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    record_video=False
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP
hopf_vars = np.zeros((8,TEST_STEPS))
hopf_vars_dot = np.zeros((8,TEST_STEPS))
torques = np.zeros((12,TEST_STEPS))
joint_pos = np.zeros((12,TEST_STEPS))
rob_vel = np.zeros((3,TEST_STEPS))
start_time = time.time()

# [TODO] initialize data structures to save CPG and robot states


############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])*3
kd=np.array([2,2,2])*3
# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)

#Setup of tracked states
des_foot_pos = np.zeros([3,TEST_STEPS])
actual_foot_pos = np.zeros([3,TEST_STEPS])
des_joint_pos = np.zeros([3,TEST_STEPS])
actual_joint_pos = np.zeros([3,TEST_STEPS])


for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 
  # get desired foot positions from CPG 
  xs,zs = cpg.update()
  des_foot_pos[1,:] = -foot_y
  des_foot_pos[0,j],des_foot_pos[2,j] = xs[0],zs[0]
  _, actual_foot_pos[:,j] = env.robot.ComputeJacobianAndPosition(0)

  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py____ DONE
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()
  actual_joint_pos[:,j] = q[0:3]
  des_joint_pos[:,j] = env.robot.ComputeInverseKinematics(0,np.array([xs[0],sideSign[0] * foot_y,zs[0]]))

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)
    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(i,leg_xyz) # [TODO] 
    # Add joint PD contribution to tau for leg i (Equation 4)
    dq_des = np.zeros(3)
    tau += kp*(leg_q-q[i*3:i*3+3])+kd*(dq_des-dq[i*3:i*3+3]) # [TODO]

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      J , foot_pos = env.robot.ComputeJacobianAndPosition(i)
      # Get current foot velocity in leg frame (Equation 2)
      vd = np.zeros(3) 
      # foot velocity in leg frame i (Equation 2)
      v = np.dot(J,dq[i*3:i*3+3])
      #desired foot position
      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau += J.T @ (kpCartesian @ (leg_xyz - foot_pos) + kdCartesian @ (vd - v)) # [TODO]

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau
    torques[3*i:3*i+3,j] = tau

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

   # save any CPG or robot states
  joint_pos[:,j] = env.robot.GetMotorAngles()
  hopf_vars[:,j] = cpg.X[0:2,:].flatten('F')
  hopf_vars_dot[:,j] = cpg.X_dot[0:2,:].flatten('F')
  rob_vel[:,j] = env.robot.GetBaseLinearVelocity()
  average_body_vel = np.average(rob_vel[0,:])






  # time.sleep(0.01)

step_times = find_local_minima(hopf_vars[1,:])
step_duration = np.zeros(len(step_times)-1)
for i in range(len(step_times)-1):
   step_duration = step_times[i+1]-step_times[i]
total_time = time.time() - start_time
print('TOTAL TIME', total_time) 
low_bound = step_times[10]
up_bound = step_times[15]

# Condition: Select indices where leg is swinging
condition = np.logical_and(0 <= hopf_vars[1, :], hopf_vars[1, :] <= np.pi)
# Use np.where to get the indices that satisfy the condition
swing_indices = np.where(condition)[0]
original_step_indices = np.where(np.diff(swing_indices) > 1, )[0]
# Convert original step indices to corresponding indices in hopf_vars
step_indices = swing_indices[original_step_indices]
# Count the number of steps
step_count = len(step_indices)

period_start_times = []
period_end_times = []

# Iterate through swing_indices to determine start and end times
start_time = swing_indices[0]
for i in range(1, len(swing_indices)):
    if swing_indices[i] != swing_indices[i - 1] + 1:
        # If the current index is not consecutive to the previous one, it's the end of a period
        end_time = swing_indices[i - 1]
        period_start_times.append(start_time)
        period_end_times.append(end_time)
        start_time = swing_indices[i]

# The last period might continue until the end of the vector
end_time = swing_indices[-1]
period_start_times.append(start_time)
period_end_times.append(end_time)

# Create a new vector containing start and end times
periods_vector = list(zip(period_start_times, period_end_times))
period_durations = [end - start + 1 for start, end in periods_vector]
average_duration = np.average(period_durations)/1000
# Calculate the average gait cycle duration
gait_cycle_durations = [periods_vector[i+1][0] - periods_vector[i][0] for i in range(len(periods_vector)-1)]
average_gait_cycle_duration = np.average(gait_cycle_durations)/1000
# Calculate the duty cycle as the average length of swing divided by the gait duration
duty_cycle_percentage = (average_duration / average_gait_cycle_duration) * 100

#Find Cost of transportation
total_mass = sum(env.robot.GetTotalMassFromURDF())
gravity = 9.81
cot = calculate_COT(torques, joint_pos, rob_vel[0,:], total_mass, gravity, TIME_STEP)


##################################################### 
# PLOTS
#####################################################
if PLOT:
  # fig = plt.figure()

  # Base velocity 
  '''
  plt.figure()
  plt.plot(t,rob_vel[0,:], label='x velocity')
  plt.plot(t,rob_vel[1,:], label='y velocity')
  plt.plot(t,rob_vel[2,:], label='z velocity')
  plt.title('Global Base Velocity')
  plt.legend()

  # Joint angles 
  plt.figure()
  plt.subplot(2, 1, 1)
  plt.plot(t,joint_pos[1,:], label='FR Hip Joint Angles')
  plt.plot(t,joint_pos[2,:], label='FR Knee Joint Angles')
  plt.grid()

  plt.subplot(2, 1, 2)
  plt.plot(t,joint_pos[7,:], label='RR Hip Joint Angles')
  plt.plot(t,joint_pos[8,:], label='RR knee Joint Angles')
  plt.grid(which='both')
  plt.legend()
  '''


  #####################################################
  #Plot of CPG states for one leg 
  plt.figure()
  plt.subplot(2, 2, 1)
  plt.plot(t[low_bound:up_bound],hopf_vars[0,low_bound:up_bound], label='FR r')
  plt.plot(t[low_bound:up_bound],hopf_vars[1,low_bound:up_bound], label='FR phi')
  plt.plot(t[low_bound:up_bound],hopf_vars_dot[0,low_bound:up_bound], label='FR r_dot')
  plt.plot(t[low_bound:up_bound],hopf_vars_dot[1,low_bound:up_bound], label='FR phi_dot')
  plt.grid(which='both')
  plt.legend()

  plt.subplot(2, 2, 2)
  plt.plot(t[low_bound:up_bound],hopf_vars[2,low_bound:up_bound], label='FL r')
  plt.plot(t[low_bound:up_bound],hopf_vars[3,low_bound:up_bound], label='FL phi')
  plt.plot(t[low_bound:up_bound],hopf_vars_dot[2,low_bound:up_bound], label='FL r_dot')
  plt.plot(t[low_bound:up_bound],hopf_vars_dot[3,low_bound:up_bound], label='FL phi_dot')
  plt.grid(which='both')
  plt.legend()

  plt.subplot(2, 2, 3)
  plt.plot(t[low_bound:up_bound],hopf_vars[4,low_bound:up_bound], label='RR r')
  plt.plot(t[low_bound:up_bound],hopf_vars[5,low_bound:up_bound], label='RR phi')
  plt.plot(t[low_bound:up_bound],hopf_vars_dot[4,low_bound:up_bound], label='FR r_dot')
  plt.plot(t[low_bound:up_bound],hopf_vars_dot[5,low_bound:up_bound], label='FR phi_dot')
  plt.grid(which='both')
  plt.legend()

  plt.subplot(2, 2, 4)
  plt.plot(t[low_bound:up_bound],hopf_vars[6,low_bound:up_bound], label='RL r')
  plt.plot(t[low_bound:up_bound],hopf_vars[7,low_bound:up_bound], label='RL phi')
  plt.plot(t[low_bound:up_bound],hopf_vars_dot[6,low_bound:up_bound], label='RL r_dot')
  plt.plot(t[low_bound:up_bound],hopf_vars_dot[7,low_bound:up_bound], label='RL phi_dot')
  plt.grid(which='both')
  plt.legend()
  plt.show()

  #3D Plot
  '''plt.figure()
  ax = plt.axes(projection='3d')
  # Plotting points in 3D
  ax.scatter(des_foot_pos[0], des_foot_pos[1], des_foot_pos[2], c='blue', marker='o', label='Desired Foot Position')
  ax.scatter(actual_foot_pos[0], actual_foot_pos[1], actual_foot_pos[2], c='red', marker='o', label='Actual Foot Position')


  # Connecting points with lines
  ax.plot(des_foot_pos[0, :], des_foot_pos[1, :], des_foot_pos[2, :], color='blue', linestyle='-', linewidth=1)
  ax.plot(actual_foot_pos[0, :], actual_foot_pos[1, :], actual_foot_pos[2, :], color='red', linestyle='-', linewidth=1)


  ax.set_xlabel('X Axis')
  ax.set_ylabel('Y Axis')
  ax.set_zlabel('Z Axis')

  plt.legend()
  plt.show()'''

  #Plot of actual foot position vs desired foot position for one leg
  plt.figure()
  plt.subplot(3,1,1)
  plt.plot(t,des_foot_pos[0],label='Desired x position')
  plt.plot(t,actual_foot_pos[0],label='Actual x position')
  plt.title('Desired vs Actual Foot position in X')
  plt.legend()

  plt.subplot(3,1,2)
  plt.plot(t,des_foot_pos[1],label='Desired y position')
  plt.plot(t,actual_foot_pos[1],label='Actual y position')
  plt.title('Desired vs Actual Foot position in Y')
  plt.legend()

  plt.subplot(3,1,3)
  plt.plot(t,des_foot_pos[2],label='Desired z position')
  plt.plot(t,actual_foot_pos[2],label='Actual z position')
  plt.title('Desired vs Actual Foot position in Z')
  plt.legend()
  plt.tight_layout()
  plt.show()


  #Plot of Actual Joint angles vs Desired Joint angles for one leg
  plt.figure()
  plt.subplot(3,1,1)
  plt.plot(t,des_joint_pos[0],label='Desired hip position')
  plt.plot(t,actual_joint_pos[0],label='Actual hip position')
  plt.title('Actual vs Desired Hip Positions')
  plt.legend()

  plt.subplot(3,1,2)
  plt.plot(t,des_joint_pos[1],label='Desired Thigh position')
  plt.plot(t,actual_joint_pos[1],label='Actual Thigh position')
  plt.title('Actual vs Desired Thigh Positions')
  plt.legend()

  plt.subplot(3,1,3)
  plt.plot(t,des_joint_pos[2],label='Desired Calf position')
  plt.plot(t,actual_joint_pos[2],label='Actual Calf position')
  plt.title('Desired vs Actual Calf Position')
  plt.legend()
  plt.tight_layout()
  plt.show()

#Printing performance characteristics 
print('Average forward body velocity: ',average_body_vel)
print('Cost of Transport: ',cot)
print("Average duration of swing:", average_duration)
print('Average Gait Duration:', average_gait_cycle_duration)
print('Swing Duty cycle:', duty_cycle_percentage)


