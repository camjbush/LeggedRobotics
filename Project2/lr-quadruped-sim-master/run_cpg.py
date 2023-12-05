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

ADD_CARTESIAN_PD = False
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
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP
hopf_vars = np.zeros((8,TEST_STEPS))
hopf_vars_dot = np.zeros((8,TEST_STEPS))
joint_pos = np.zeros((12,TEST_STEPS))
rob_vel = np.zeros((3,TEST_STEPS))
start_time = time.time()

# [TODO] initialize data structures to save CPG and robot states


############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])
# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)
des_foot_pos = np.zeros([3,TEST_STEPS])
actual_foot_pos = np.zeros([3,TEST_STEPS])
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

    # add Cartesian PD contribution (HAVENT COMPLETED YET)
    if ADD_CARTESIAN_PD:
      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      J , foot_pos = env.robot.ComputeJacobianAndPosition(i)
      # Get current foot velocity in leg frame (Equation 2)
      vd = np.zeros(3) 
      # foot velocity in leg frame i (Equation 2)
      v = np.dot(J[i,:],dq[i*3:i*3+3])
      #desired foot position
      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau += np.multiply(J.T, (kpCartesian @ (leg_xyz - foot_pos) + kdCartesian @ (vd - v))) # [TODO]

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

   # save any CPG or robot states
  joint_pos[:,j] = env.robot.GetMotorAngles()
  hopf_vars[:,j] = cpg.X[0:2,:].flatten('F')
  hopf_vars_dot[:,j] = cpg.X_dot[0:2,:].flatten('F')
  rob_vel[:,j] = env.robot.GetBaseLinearVelocity()
  average_body_vel = np.average(rob_vel)
  # time.sleep(0.01)

step_times = find_local_minima(hopf_vars[1,:])
print('TOTAL TIME', time.time() - start_time) 
low_bound = step_times[10]
up_bound = step_times[15]


##################################################### 
# PLOTS
#####################################################
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
plt.plot(t[low_bound:up_bound],hopf_vars_dot[2,low_bound:up_bound], label='FR r_dot')
plt.plot(t[low_bound:up_bound],hopf_vars_dot[3,low_bound:up_bound], label='FR phi_dot')
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
plt.plot(t[low_bound:up_bound],hopf_vars[6,low_bound:up_bound], label='RR r')
plt.plot(t[low_bound:up_bound],hopf_vars[7,low_bound:up_bound], label='RR phi')
plt.plot(t[low_bound:up_bound],hopf_vars_dot[6,low_bound:up_bound], label='FR r_dot')
plt.plot(t[low_bound:up_bound],hopf_vars_dot[7,low_bound:up_bound], label='FR phi_dot')
plt.grid(which='both')
plt.legend()
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.plot(t,des_foot_pos[0],label='x position of foot')
plt.plot(t,des_foot_pos[1],label='y position of foot')
plt.plot(t,des_foot_pos[2],label='z position of foot')
plt.subplot(1,2,2)
plt.plot(t,actual_foot_pos[0],label='x position of foot')
plt.plot(t,actual_foot_pos[1],label='y position of foot')
plt.plot(t,actual_foot_pos[2],label='z position of foot')
plt.legend()
plt.show()


print('Desired foot position: ',des_foot_pos)
print('Actual foot position: ',actual_foot_pos)




plt.figure()
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
plt.show()