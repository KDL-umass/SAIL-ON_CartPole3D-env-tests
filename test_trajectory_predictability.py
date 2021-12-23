"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/book/code/pole.c
"""

import os
import sys
import time

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import pybullet as p2
from pybullet_utils import bullet_client as bc
import math

import json
from glob import glob
import copy

import random
random.seed(123)

import warnings
warnings.filterwarnings("ignore", message="Box bound precision lowered by casting to float32")
warnings.filterwarnings("ignore", category=UserWarning)


class CartPoleBulletEnv_077(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, path, use_img=False, renders=False, discrete_actions=True):
        # start the bullet physics server
        self._renders = renders
        self._discrete_actions = discrete_actions
        self._render_height = 480
        self._render_width = 640
        self._physics_client_id = -1
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 0.4  # 2.4
        self.use_img = use_img
        high = np.array([self.x_threshold * 2, np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2, np.finfo(np.float32).max])

        # Environmental params
        self.force_mag = 10
        self.timeStep = 1.0 / 50.0
        self.angle_limit = 10
        self.actions = ['left', 'right', 'forward', 'backward', 'nothing']

        # Internal params
        self.path = path
        self.tick_limit = 200
        self.tick = 0
        self.time = None

        ## Param for setting initial conditions to zero
        self.init_zero = True

        # Object definitions
        self.nb_blocks = None
        self.cartpole = -10
        self.ground = None
        self.blocks = list()
        self.walls = None
        self.state = None

        if self._discrete_actions:
            self.action_space = spaces.Discrete(5)
        else:
            action_dim = 1
            action_high = np.array([self.force_mag] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self._configure()

        return None

    def _configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        p = self._p

        # Convert from string to int
        if action == 'nothing':
            action = 0
        elif action == 'left':
            action = 1
        elif action == 'right':
            action = 2
        elif action == 'forward':
            action = 3
        elif action == 'backward':
            action = 4

        # Handle math first then direction
        cart_deg_angle = self.quaternion_to_euler(*p.getLinkState(self.cartpole, 0)[1])[2]
        cart_angle = (cart_deg_angle) * np.pi / 180

        # Adjust forces so it always apply in reference to world frame
        fx = self.force_mag * np.cos(cart_angle)
        fy = self.force_mag * np.sin(cart_angle) * -1

        # based on action decide the x and y forces
        if action == 0:
            fx = 0.0
            fy = 0.0
        elif action == 1:
            fx = fx
            fy = fy
        elif action == 2:
            fx = -fx
            fy = - fy
        elif action == 3:
            tmp = fx
            fx = -fy
            fy = tmp
        elif action == 4:
            tmp = fx
            fx = fy
            fy = -tmp
        else:
            raise Exception("unknown discrete action [%s]" % action)

        # Apply correccted forces
        p.applyExternalForce(self.cartpole, 0, (fx, fy, 0.0), (0, 0, 0), p.LINK_FRAME)

        # Apply anti-gravity to blocks
        for i in self.blocks:
            p.applyExternalForce(i, -1, (0, 0, 9.8), (0, 0, 0), p.LINK_FRAME)

        p.stepSimulation()

        done = self.is_done()
        reward = self.get_reward()

        self.tick = self.tick + 1

        return self.get_state(), reward, done, {}

    # Check if is done
    def is_done(self):
        # Check tick limit condition
        if self.tick >= self.tick_limit:
            return True

        # Check pole angle condition
        p = self._p
        pos, vel, jRF, aJMT = p.getJointStateMultiDof(self.cartpole, 1)
        pos = self.quaternion_to_euler(*pos)
        x_angle = abs(pos[0])
        y_angle = abs(pos[1])

        if x_angle < self.angle_limit and y_angle < self.angle_limit:
            return False
        else:
            return True

        return None

    def get_reward(self):
        return self.tick / self.tick_limit

    def get_time(self):
        return self.time + self.tick * self.timeStep

    def get_actions(self):
        return self.actions

    def reset(self):
        # self.close()
        # Set time paremeter for sensor value
        self.time = time.time()

        # Create client if it doesnt exist
        if self._physics_client_id < 0:
            self.generate_world()

        self.tick = 0
        self.reset_world()

        # Run for one step to get everything going
        self.step(0)

        return self.get_state(initial=True)

    # Used to generate the initial world state
    def generate_world(self):
        # Create bullet physics client
        if self._renders:
        #if True:
            self._p = bc.BulletClient(connection_mode=p2.GUI)
        else:
            self._p = bc.BulletClient(connection_mode=p2.DIRECT)
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K") # Clear to the end of line

        # Client id link, for closing or checking if running
        self._physics_client_id = self._p._client

        # Load world simulation
        p = self._p
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)

        # Load world objects
        self.cartpole = p.loadURDF(os.path.join(self.path, 'ground_cart.urdf'))
        self.walls = p.loadURDF(os.path.join(self.path, 'walls.urdf'))

        # Set walls to be bouncy
        for joint_nb in range(-1, 6):
            p.changeDynamics(self.walls, joint_nb, restitution=1.0, lateralFriction=0.0,
                             rollingFriction=0.0, spinningFriction=0.0)

        return None

    def reset_world(self):
        # Reset world (assume is created)
        p = self._p

        # Delete cartpole
        if self.cartpole == -10:
            self.cartpole = p.loadURDF(os.path.join(self.path, 'ground_cart.urdf'))
        else:
            p.removeBody(self.cartpole)
            self.cartpole = p.loadURDF(os.path.join(self.path, 'ground_cart.urdf'))

        # This big line sets the spehrical joint on the pole to loose
        p.setJointMotorControlMultiDof(self.cartpole, 1, p.POSITION_CONTROL, targetPosition=[0, 0, 0, 1],
                                       targetVelocity=[0, 0, 0], positionGain=0, velocityGain=0.1,
                                       force=[0, 0, 0])

        # Reset cart (technicaly ground object)
        if self.init_zero:
            cart_pos = list(self.np_random.uniform(low=0, high=0, size=(2,))) + [0]
            cart_vel = list(self.np_random.uniform(low=0, high=0, size=(2,))) + [100]
        else:
            cart_pos = list(self.np_random.uniform(low=-3, high=3, size=(2,))) + [0]
            cart_vel = list(self.np_random.uniform(low=-1, high=1, size=(2,))) + [100]
        p.resetBasePositionAndOrientation(self.cartpole, cart_pos, [0, 0, 0, 1])
        p.applyExternalForce(self.cartpole, 0, cart_vel, (0, 0, 0), p.WORLD_FRAME)

        # Reset pole
        if self.init_zero:
            randstate = list(self.np_random.uniform(low=0, high=0, size=(6,)))
        else:
            randstate = list(self.np_random.uniform(low=-0.01, high=0.01, size=(6,)))
        pole_pos = randstate[0:3] + [1]
        # zero so it doesnt spin like a top :)
        pole_ori = list(randstate[3:5]) + [0]
        p.resetJointStateMultiDof(self.cartpole, 1, targetValue=pole_pos, targetVelocity=pole_ori)

        # Delete old blocks
        for i in self.blocks:
            p.removeBody(i)

        # Load blocks in
        self.nb_blocks = np.random.randint(3) + 2
        self.blocks = [None] * self.nb_blocks
        for i in range(self.nb_blocks):
            self.blocks[i] = p.loadURDF(os.path.join(self.path, 'block.urdf'))

        # Set blocks to be bouncy
        for i in self.blocks:
            p.changeDynamics(i, -1, restitution=1.0, lateralFriction=0.0,
                             rollingFriction=0.0, spinningFriction=0.0)

        # Set block posistions
        min_dist = 1
        cart_pos, _ = p.getBasePositionAndOrientation(self.cartpole)
        cart_pos = np.asarray(cart_pos)
        for i in self.blocks:
            pos = self.np_random.uniform(low=-4.0, high=4.0, size=(3,))
            pos[2] = pos[2] + 5.0
            while np.linalg.norm(cart_pos[0:2] - pos[0:2]) < min_dist:
                pos = self.np_random.uniform(low=-4.0, high=4.0, size=(3,))
                # Z is not centered at 0.0
                pos[2] = pos[2] + 5.0
            p.resetBasePositionAndOrientation(i, pos, [0, 0, 1, 0])

        # Set block velocities
        for i in self.blocks:
            vel = self.np_random.uniform(low=6.0, high=10.0, size=(3,))
            for ind, val in enumerate(vel):
                if np.random.rand() < 0.5:
                    vel[ind] = val * -1

            p.resetBaseVelocity(i, vel, [0, 0, 0])

        return None

    def set_world(self, state):
        # Reset world (assume is created)
        p = self._p

        start_state = state
        cart_pos = [start_state['cart']['x_position'],
                            start_state['cart']['y_position'],
                            start_state['cart']['z_position']]
        cart_vel = [start_state['cart']['x_velocity'],
                    start_state['cart']['y_velocity'],
                    start_state['cart']['z_velocity']]
        pole_ori = [start_state['pole']['x_quaternion'],
                    start_state['pole']['y_quaternion'],
                    start_state['pole']['z_quaternion'],
                    start_state['pole']['w_quaternion']]
        pole_vel = [start_state['pole']['x_velocity'],
                    start_state['pole']['y_velocity'],
                    start_state['pole']['z_velocity']]

        cart_pos = [cart_pos[0], cart_pos[1], cart_pos[2]]
        cart_vel = [cart_vel[2], cart_vel[1], cart_vel[0]]
        pole_ori = [pole_ori[1], pole_ori[0], pole_ori[2], pole_ori[3]]
        pole_vel = [pole_vel[1], pole_vel[0], pole_vel[2]]

        # Delete cartpole
        if self.cartpole == -10:
            pass
        else:
            p.removeBody(self.cartpole, physicsClientId=self._physics_client_id)

        # correctly sets cart position via the basePosition of the cartpole object
        self.cartpole = p.loadURDF(os.path.join(self.path, 'ground_cart.urdf'),
              basePosition=cart_pos,
              baseOrientation=[0, 0, 0, 1],
              physicsClientId=self._physics_client_id)

        # This big line sets the spherical joint on the pole to loose ONLY
        # no affect on cartpole state position or velocity values
        p.setJointMotorControlMultiDof(self.cartpole, 1, p.POSITION_CONTROL, targetPosition=[0, 0, 0, 1],
                                       targetVelocity=[0, 0, 0], positionGain=0, velocityGain=0.1,
                                       force=[0, 0, 0],
                                       physicsClientId=self._physics_client_id)

        # correctly sets the cart velocity via the basePosition of the cartpole object
        p.resetBasePositionAndOrientation(self.cartpole, cart_pos, [0, 0, 0, 1])
        p.resetJointStateMultiDof(self.cartpole, 0, targetValue=[0, 0, 0], targetVelocity=cart_vel, physicsClientId=self._physics_client_id)
        # correctly sets the pole velocity and position

        p.resetJointStateMultiDof(self.cartpole, 1, targetValue=pole_ori, targetVelocity=pole_vel,
                                  physicsClientId=self._physics_client_id)

        p.setGravity(0, 0, -9.8, physicsClientId=self._physics_client_id)

        for i in self.blocks:
            p.removeBody(i, physicsClientId=self._physics_client_id)
        self.blocks = []

        blocks = start_state['blocks']
        self.nb_blocks = len(blocks)

        self.blocks = [p.loadURDF(os.path.join(self.path, 'block.urdf'),
                                  physicsClientId=self._physics_client_id) for _ in range(self.nb_blocks)]

        self.nb_blocks = len(self.blocks)

        assert self.nb_blocks == len(self.blocks)

        # Set blocks to be bounce off walls
        for i in self.blocks:
            p.changeDynamics(i, -1, restitution=1.0, lateralFriction=0.0,
                             rollingFriction=0.0, spinningFriction=0.0,
                             physicsClientId=self._physics_client_id)


        for i, b in enumerate(self.blocks):
            bb = blocks[i]
            block_pos = [bb['x_position'], bb['y_position'], bb['z_position']]
            block_vel = [bb['x_velocity'], bb['y_velocity'], bb['z_velocity']]
            p.resetBasePositionAndOrientation(b, block_pos, [0, 0, 1, 0],
                                              physicsClientId=self._physics_client_id)
            p.resetBaseVelocity(b, block_vel, [0,0,0],
                                physicsClientId=self._physics_client_id)

    # Unified function for getting state information
    def get_state(self, initial=False):
        p = self._p
        world_state = dict()
        round_amount = 6

        # Get cart info ============================================
        state = dict()

        # Handle pos, ori
        _, vel, _, _ = p.getJointStateMultiDof(self.cartpole, 0)
        pos, _, _, _, _, _ = p.getLinkState(self.cartpole, 0)
        state['x_position'] = round(pos[0], round_amount)
        state['y_position'] = round(pos[1], round_amount)
        state['z_position'] = round(pos[2], round_amount)

        # Handle velocity
        state['x_velocity'] = round(vel[2], round_amount)
        state['y_velocity'] = round(vel[1], round_amount)
        state['z_velocity'] = round(vel[0], round_amount)

        world_state['cart'] = state

        # Get pole info =============================================
        state = dict()
        use_euler = False

        # Position and orientation, the other two not used
        pos, vel, jRF, aJMT = p.getJointStateMultiDof(self.cartpole, 1)

        # Convert quats to eulers
        eulers = self.quaternion_to_euler(*pos)

        # Position
        if use_euler:
            state['x_position'] = round(eulers[0], round_amount)
            state['y_position'] = round(eulers[1], round_amount)
            state['z_position'] = round(eulers[2], round_amount)
        else:
            state['x_quaternion'] = round(pos[1], round_amount)
            state['y_quaternion'] = round(pos[0], round_amount)
            state['z_quaternion'] = round(pos[2], round_amount)
            state['w_quaternion'] = round(pos[3], round_amount)

        # Velocity
        state['x_velocity'] = round(vel[1], round_amount)
        state['y_velocity'] = round(vel[0], round_amount)
        state['z_velocity'] = round(vel[2], round_amount)

        world_state['pole'] = state

        # get block info ====================================
        block_state = list()
        for ind, val in enumerate(self.blocks):
            state = dict()
            state['id'] = val

            pos, _ = p.getBasePositionAndOrientation(val)
            state['x_position'] = round(pos[0], round_amount)
            state['y_position'] = round(pos[1], round_amount)
            state['z_position'] = round(pos[2], round_amount)

            vel, _ = p.getBaseVelocity(val)
            state['x_velocity'] = round(vel[0], round_amount)
            state['y_velocity'] = round(vel[1], round_amount)
            state['z_velocity'] = round(vel[2], round_amount)

            block_state.append(state)

        world_state['blocks'] = block_state

        # Get wall info ======================================
        # Hardcoded cause I don't know how to get the info :(
        if initial:
            state = list()
            state.append([-5, -5, 0])
            state.append([5, -5, 0])
            state.append([5, 5, 0])
            state.append([-5, 5, 0])

            state.append([-5, -5, 10])
            state.append([5, -5, 10])
            state.append([5, 5, 10])
            state.append([-5, 5, 10])

            world_state['walls'] = state

        return world_state

    def get_image(self):
        if self.use_img:
            return self.render()
        else:
            return None

    def render(self, mode='human', close=False, dist='close'):
        if mode == "human":
            self._renders = True

        if dist == 'far':
            base_pos = [4.45, 4.45, 9.8]
            cam_dist = 0.1
            cam_pitch = -45.0
            cam_yaw = 45.0 + 90
            cam_roll = 0.0
            fov = 100

        elif dist == 'close':
            base_pos = [4.45, 4.45, 2.0]
            cam_dist = 0.1
            cam_pitch = -15.0
            cam_yaw = 45.0 + 90
            cam_roll = 0.0
            fov = 60

        elif dist == 'follow':
            base_pose, _ = self._p.getBasePositionAndOrientation(self.cartpole)
            pos, vel, jRF, aJMT = self._p.getJointStateMultiDof(self.cartpole, 0)

            x = pos[0] + base_pose[0]
            y = pos[1] + base_pose[1]

            base_pos = [x, y, 2.0]
            cam_dist = 0.1
            cam_pitch = -15.0
            cam_yaw = 45.0 + 90
            cam_roll = 0.0
            fov = 60

        if self._physics_client_id >= 0:
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=cam_dist,
                yaw=cam_yaw,
                pitch=cam_pitch,
                roll=cam_roll,
                upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=fov,
                                                             aspect=float(self._render_width) /
                                                                    self._render_height,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(
                width=self._render_width,
                height=self._render_height,
                renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix)
        else:
            px = np.array([[[255, 255, 255, 255]] * self._render_width] * self._render_height, dtype=np.uint8)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def configure(self, args):
        pass

    def eulerToQuaternion(self, yaw, pitch, roll):
        qx = np.sin(yaw / 2) * np.sin(pitch / 2) * np.cos(roll / 2) + np.cos(yaw / 2) * np.cos(pitch / 2) * np.sin(
            roll / 2)
        qy = np.sin(yaw / 2) * np.cos(pitch / 2) * np.cos(roll / 2) + np.cos(yaw / 2) * np.sin(pitch / 2) * np.sin(
            roll / 2)
        qz = np.cos(yaw / 2) * np.sin(pitch / 2) * np.cos(roll / 2) - np.sin(yaw / 2) * np.cos(pitch / 2) * np.sin(
            roll / 2)
        qw = np.cos(yaw / 2) * np.cos(pitch / 2) * np.cos(roll / 2) - np.sin(yaw / 2) * np.sin(pitch / 2) * np.sin(
            roll / 2)

        return (qx, qy, qz, qw)

    def quaternion_to_euler(self, x, y, z, w):
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = np.degrees(np.arctan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = np.where(t2 > +1.0, +1.0, t2)
        # t2 = +1.0 if t2 > +1.0 else t2

        t2 = np.where(t2 < -1.0, -1.0, t2)
        # t2 = -1.0 if t2 < -1.0 else t2
        Y = np.degrees(np.arcsin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = np.degrees(np.arctan2(t3, t4))

        return (X, Y, Z)

    def close(self):
        if self._physics_client_id >= 0:
            self._p.disconnect()
        self._physics_client_id = -1

class CartPoleBulletEnv_078(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, params: dict = None):
        # start the bullet physics server
        self._render_height = 480
        self._render_width = 640
        self._physics_client_id = -1
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 0.4  # 2.4
        high = np.array([self.x_threshold * 2, np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2, np.finfo(np.float32).max])
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Environmental params
        self.force_mag = 10
        self.timeStep = 1.0 / 50.0
        self.angle_limit = 10.0 * np.pi / 180.0 # 10 degrees in radians
        self.actions = ['right', 'left', 'forward', 'backward', 'nothing']
        self.tick_limit = 200

        # Internal params
        self.params = params
        self.path = self.params['path']
        self._renders = self.params['use_gui']
        self.tick = 0
        self.time = None
        self.np_random = None
        self.use_img = self.params['use_img']
        self._p = None

        # Params
        self.init_zero = False
        self.config = self.params['config']

        # Object definitions
        self.nb_blocks = None
        self.cartpole = -10
        self.ground = None
        self.blocks = list()
        self.walls = None
        self.state = None
        self.origin = None

        # Functions to be run directly after init
        self.seed(self.params['seed'])

        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return None

    def step(self, action):
        p = self._p

        # Convert from string to int
        if action == 'nothing':
            action = 0
        elif action == 'right':
            action = 1
        elif action == 'left':
            action = 2
        elif action == 'forward':
            action = 3
        elif action == 'backward':
            action = 4

        # Adjust forces so they always apply in reference to world frame
        _, ori, _, _, _, _ = p.getLinkState(self.cartpole, 0)
        cart_angle = p.getEulerFromQuaternion(ori)[2] # yaw
        fx = self.force_mag * np.cos(cart_angle)
        fy = self.force_mag * np.sin(cart_angle) * -1

        # based on action decide the x and y forces
        if action == 0:
            fx = 0.0
            fy = 0.0
        elif action == 1:
            fx = fx
            fy = fy
        elif action == 2:
            fx = -fx
            fy = - fy
        elif action == 3:
            tmp = fx
            fx = -fy
            fy = tmp
        elif action == 4:
            tmp = fx
            fx = fy
            fy = -tmp
        else:
            raise Exception("unknown discrete action [%s]" % action)

        # Apply correccted forces
        p.applyExternalForce(self.cartpole, 0, (fx, fy, 0.0), (0, 0, 0), p.LINK_FRAME)

        # Apply anti-gravity to blocks
        for i in self.blocks:
            p.applyExternalForce(i, -1, (0, 0, 9.8), (0, 0, 0), p.LINK_FRAME)

        p.stepSimulation()

        done = self.is_done()
        reward = self.get_reward()

        self.tick = self.tick + 1

        return self.get_state(), reward, done, {}

    # Check if is done
    def is_done(self):
        # Check tick limit condition
        if self.tick >= self.tick_limit:
            return True

        # Check pole angle condition
        p = self._p
        _, _, _, _, _, ori, _, _ = p.getLinkState(self.cartpole, 1, 1)
        eulers = p.getEulerFromQuaternion(ori)
        x_angle, y_angle = eulers[0], eulers[1]

        if abs(x_angle) > self.angle_limit or abs(y_angle) > self.angle_limit:
            return True
        else:
            return False

        return None

    def get_reward(self):
        return self.tick / self.tick_limit

    def get_time(self):
        return self.time + self.tick * self.timeStep

    def get_actions(self):
        return self.actions

    def reset(self):
        # Set time paremeter for sensor value
        self.time = time.time()

        # Create client if it doesnt exist
        if self._physics_client_id < 0:
            self.generate_world()

        self.tick = 0
        self.reset_world()

        # Run for one step to get everything going
        self.step(0)

        return self.get_state(initial=True)

    # Used to generate the initial world state
    def generate_world(self):
        # Read user config here
        if self.config is not None:
            if 'start_zeroed_out' in self.config:
                self.init_zero = self.config['start_zeroed_out']
            if 'episode_seed' in self.config:
                self.seed(self.config['episode_seed'])
            if 'start_world_state' in self.config:
                self.set_world(self.config['start_world_state'])

        # Create bullet physics client
        if self._renders:
            self._p = bc.BulletClient(connection_mode=p2.GUI)
        else:
            self._p = bc.BulletClient(connection_mode=p2.DIRECT)
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K") # Clear to the end of line

        # Client id link, for closing or checking if running
        self._physics_client_id = self._p._client

        # Load world simulation
        p = self._p
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)

        # Load world objects
        self.cartpole = p.loadURDF(os.path.join(self.path, 'ground_cart.urdf'))
        self.walls = p.loadURDF(os.path.join(self.path, 'walls.urdf'))
        self.origin = p.loadURDF(os.path.join(self.path, 'origin.urdf'))

        # Set walls to be bouncy
        for joint_nb in range(-1, 6):
            p.changeDynamics(self.walls, joint_nb, restitution=1.0, lateralFriction=0.0,
                             rollingFriction=0.0, spinningFriction=0.0)

        return None

    def reset_world(self):
        # Reset world (assume is created)
        p = self._p

        # Delete cartpole
        if self.cartpole == -10:
            self.cartpole = p.loadURDF(os.path.join(self.path, 'ground_cart.urdf'))
        else:
            p.removeBody(self.cartpole)
            self.cartpole = p.loadURDF(os.path.join(self.path, 'ground_cart.urdf'))

        # This big line sets the spehrical joint on the pole to loose
        p.setJointMotorControlMultiDof(self.cartpole, 1, p.POSITION_CONTROL, targetPosition=[0, 0, 0, 1],
                                       targetVelocity=[0, 0, 0], positionGain=0, velocityGain=0.0,
                                       force=[0, 0, 0])

        # Reset cart (technicaly ground object)
        if self.init_zero:
            cart_pos = list(self.np_random.uniform(low=0, high=0, size=(2,))) + [0]
            cart_vel = list(self.np_random.uniform(low=0, high=0, size=(2,))) + [0]
        else:
            cart_pos = list(self.np_random.uniform(low=-3, high=3, size=(2,))) + [0]
            cart_vel = list(self.np_random.uniform(low=-1, high=1, size=(2,))) + [0]

        p.resetBasePositionAndOrientation(self.cartpole, cart_pos, [0, 0, 0, 1])
        p.applyExternalForce(self.cartpole, 0, cart_vel, (0, 0, 0), p.LINK_FRAME)

        # Reset pole
        if self.init_zero:
            randstate = list(self.np_random.uniform(low=0, high=0, size=(6,)))
        else:
            randstate = list(self.np_random.uniform(low=-0.01, high=0.01, size=(6,)))

        pole_pos = randstate[0:3] + [1]
        # zero so it doesnt spin like a top :)
        pole_ori = list(randstate[3:5]) + [0]
        p.resetJointStateMultiDof(self.cartpole, 1, targetValue=pole_pos, targetVelocity=pole_ori)

        # Delete old blocks
        for i in self.blocks:
            p.removeBody(i)

        # Load blocks in
        self.nb_blocks = np.random.randint(3) + 2
        self.blocks = [None] * self.nb_blocks
        for i in range(self.nb_blocks):
            self.blocks[i] = p.loadURDF(os.path.join(self.path, 'block.urdf'))

        # Set blocks to be bouncy
        for i in self.blocks:
            p.changeDynamics(i, -1, restitution=1.0, lateralFriction=0.0,
                             rollingFriction=0.0, spinningFriction=0.0)

        # Set block posistions
        min_dist = 1
        cart_pos, _ = p.getBasePositionAndOrientation(self.cartpole)
        cart_pos = np.asarray(cart_pos)
        for i in self.blocks:
            pos = self.np_random.uniform(low=-4.0, high=4.0, size=(3,))
            pos[2] = pos[2] + 5.0
            while np.linalg.norm(cart_pos[0:2] - pos[0:2]) < min_dist:
                pos = self.np_random.uniform(low=-4.0, high=4.0, size=(3,))
                # Z is not centered at 0.0
                pos[2] = pos[2] + 5.0
            p.resetBasePositionAndOrientation(i, pos, [0, 0, 1, 0])

        # Set block velocities
        for i in self.blocks:
            vel = self.np_random.uniform(low=6.0, high=10.0, size=(3,))
            for ind, val in enumerate(vel):
                if np.random.rand() < 0.5:
                    vel[ind] = val * -1

            p.resetBaseVelocity(i, vel, [0, 0, 0])

        return None

    # Terry's implementation
    def set_world(self, state):
        # print('Set World only approximately implemented :(')
        p = self._p
        cart_position = [state["cart"]["x_position"],  state["cart"]["y_position"],state["cart"]["z_position"]]
        # we swap x and z for velocity interface to pybullet
        cart_velocity = [state["cart"]["z_velocity"], state["cart"]["y_velocity"],state["cart"]["x_velocity"]]

        p.resetBasePositionAndOrientation(self.cartpole, cart_position, [0, 0, 0, 1])
        p.resetJointStateMultiDof(self.cartpole, 0, targetValue=[0,0,0], targetVelocity=cart_velocity)

        # Reset pole
        pole_position = [state["pole"]["x_quaternion"],state["pole"]["y_quaternion"],state["pole"]["z_quaternion"],state["pole"]["w_quaternion"]]
        pole_velocity = [state["pole"]["x_velocity"],state["pole"]["y_velocity"],state["pole"]["z_velocity"]*0]
        p.resetJointStateMultiDof(self.cartpole, 1, targetValue=pole_position, targetVelocity=pole_velocity)

        # Delete old blocks if number is different
        if(len(state['blocks']) != self.nb_blocks):
            for i in self.blocks:
                p.removeBody(i)

            self.nb_blocks = len(state['blocks'])
            self.blocks = [None] * self.nb_blocks
            for i in range(self.nb_blocks):
                self.blocks[i] = p.loadURDF(os.path.join(self.path, 'block.urdf'))

        i=0
        for block in state["blocks"]:
            pos = [block["x_position"], block["y_position"],block["z_position"]]
            vel = [block["x_velocity"], block["y_velocity"],block["z_velocity"]]
            p.resetBasePositionAndOrientation(self.blocks[i], pos, [0, 0, 1, 0])
            p.resetBaseVelocity(self.blocks[i], vel, [0, 0, 0])
            i = i+1

        return None

    # UMass implementation
    def set_world_umass(self, obs):
        # Reset world (assume is created)
        p = self._p

        start_state = obs
        cart_pos = [start_state['cart']['x_position'],
                            start_state['cart']['y_position'],
                            start_state['cart']['z_position']]
        cart_vel = [start_state['cart']['x_velocity'],
                    start_state['cart']['y_velocity'],
                    start_state['cart']['z_velocity']]
        pole_ori = [start_state['pole']['x_quaternion'],
                    start_state['pole']['y_quaternion'],
                    start_state['pole']['z_quaternion'],
                    start_state['pole']['w_quaternion']]
        pole_vel = [start_state['pole']['x_velocity'],
                    start_state['pole']['y_velocity'],
                    start_state['pole']['z_velocity']]

        # Note from UCCS TB:
        # we swap x and z for velocity interface to pybullet
        cart_vel = [cart_vel[2], cart_vel[1], cart_vel[0]]

        # Delete cartpole
        if self.cartpole == -10:
            pass
        else:
            p.removeBody(self.cartpole, physicsClientId=self._physics_client_id)

        # correctly sets cart position via the basePosition of the cartpole object
        self.cartpole = p.loadURDF(os.path.join(self.path, 'ground_cart.urdf'),
              basePosition=cart_pos,
              baseOrientation=[0, 0, 0, 1],
              physicsClientId=self._physics_client_id)

        # This big line sets the spherical joint on the pole to loose ONLY
        # no affect on cartpole state position or velocity values
        p.setJointMotorControlMultiDof(self.cartpole, 1, p.POSITION_CONTROL, targetPosition=[0, 0, 0, 1],
                                       targetVelocity=[0, 0, 0], positionGain=0, velocityGain=0,
                                       force=[0, 0, 0],
                                       physicsClientId=self._physics_client_id)

        # correctly sets the cart velocity via the basePosition of the cartpole object
        p.resetBasePositionAndOrientation(self.cartpole, cart_pos, [0, 0, 0, 1], physicsClientId=self._physics_client_id)
        p.resetJointStateMultiDof(self.cartpole, 0, targetValue=[0, 0, 0], targetVelocity=cart_vel, physicsClientId=self._physics_client_id)
        # correctly sets the pole velocity and position

        p.resetJointStateMultiDof(self.cartpole, 1, targetValue=pole_ori, targetVelocity=pole_vel,
                                  physicsClientId=self._physics_client_id)

        p.setGravity(0, 0, -9.8, physicsClientId=self._physics_client_id)

        for i in self.blocks:
            p.removeBody(i, physicsClientId=self._physics_client_id)
        self.blocks = []

        blocks = start_state['blocks']
        self.nb_blocks = len(blocks)

        self.blocks = [p.loadURDF(os.path.join(self.path, 'block.urdf'),
                                  physicsClientId=self._physics_client_id) for _ in range(self.nb_blocks)]

        self.nb_blocks = len(self.blocks)

        assert self.nb_blocks == len(self.blocks)

        # Set blocks to be bounce off walls
        for i in self.blocks:
            p.changeDynamics(i, -1, restitution=1.0, lateralFriction=0.0,
                             rollingFriction=0.0, spinningFriction=0.0,
                             physicsClientId=self._physics_client_id)


        for i, b in enumerate(self.blocks):
            bb = blocks[i]
            block_pos = [bb['x_position'], bb['y_position'], bb['z_position']]
            block_vel = [bb['x_velocity'], bb['y_velocity'], bb['z_velocity']]
            p.resetBasePositionAndOrientation(b, block_pos, [0, 0, 1, 0],
                                              physicsClientId=self._physics_client_id)
            p.resetBaseVelocity(b, block_vel, [0,0,0],
                                physicsClientId=self._physics_client_id)

        #cpos, _, _, _, _, _ = p.getLinkState(self.cartpole, 0, physicsClientId=self._physics_client_id)
        # print('requested cart position: ', cart_pos, 'set to: ', cpos)
        #
        #_, vel, _, _ = p.getJointStateMultiDof(self.cartpole, 0, physicsClientId=self._physics_client_id)
        # print('requested cart velocity: ', cart_vel, 'set to: ', vel)
        # # Convert quats to eulers
        # pos, vel, jRF, aJMT = p.getJointStateMultiDof(self.cartpole, 1, physicsClientId=self._physics_client_id)
        # print('requested pole position: ', pole_ori, 'set to: ', pos)
        # print('requested pole velocity: ', pole_vel, 'set to: ', vel)


    # Unified function for getting state information
    def get_state(self, initial=False):
        p = self._p
        world_state = dict()
        round_amount = 6

        # Get cart info ============================================
        state = dict()

        # Handle pos, vel
        pos, _, _, _, _, _ = p.getLinkState(self.cartpole, 0)
        state['x_position'] = round(pos[0], round_amount)
        state['y_position'] = round(pos[1], round_amount)
        state['z_position'] = round(pos[2], round_amount)

        # Cart velocity from planar joint (buggy in PyBullet; thus reverse order)
        # _, vel, _, _ = p.getJointStateMultiDof(self.cartpole, 0)
        # state['x_velocity'] = round(vel[2], round_amount)
        # state['y_velocity'] = round(vel[1], round_amount)
        # state['z_velocity'] = round(vel[0], round_amount)

        # Cart velocity from cart
        _, _, _, _, _, _, vel, _ = p.getLinkState(self.cartpole, 0, 1)
        state['x_velocity'] = round(vel[0], round_amount)
        state['y_velocity'] = round(vel[1], round_amount)
        state['z_velocity'] = round(vel[2], round_amount)

        # Set world state of cart
        world_state['cart'] = state

        # Get pole info =============================================
        state = dict()
        use_euler = False

        # Orientation and A_velocity, the others not used
        _, _, _, _, _, ori, _, vel = p.getLinkState(self.cartpole, 1, 1)

        # Orientation
        if use_euler:
            # Convert quats to eulers
            eulers = p.getEulerFromQuaternion(ori)
            state['x_euler'] = round(eulers[0], round_amount)
            state['y_euler'] = round(eulers[1], round_amount)
            state['z_euler'] = round(eulers[2], round_amount)
        else:
            state['x_quaternion'] = round(ori[0], round_amount)
            state['y_quaternion'] = round(ori[1], round_amount)
            state['z_quaternion'] = round(ori[2], round_amount)
            state['w_quaternion'] = round(ori[3], round_amount)

        # A_velocity
        state['x_velocity'] = round(vel[0], round_amount)
        state['y_velocity'] = round(vel[1], round_amount)
        state['z_velocity'] = round(vel[2], round_amount)

        world_state['pole'] = state

        # get block info ====================================
        block_state = list()
        for ind, val in enumerate(self.blocks):
            state = dict()
            state['id'] = val

            pos, _ = p.getBasePositionAndOrientation(val)
            state['x_position'] = round(pos[0], round_amount)
            state['y_position'] = round(pos[1], round_amount)
            state['z_position'] = round(pos[2], round_amount)

            vel, _ = p.getBaseVelocity(val)
            state['x_velocity'] = round(vel[0], round_amount)
            state['y_velocity'] = round(vel[1], round_amount)
            state['z_velocity'] = round(vel[2], round_amount)

            block_state.append(state)

        world_state['blocks'] = block_state

        # Get wall info ======================================
        if initial:
            state = list()
            state.append([-5, -5, 0])
            state.append([5, -5, 0])
            state.append([5, 5, 0])
            state.append([-5, 5, 0])

            state.append([-5, -5, 10])
            state.append([5, -5, 10])
            state.append([5, 5, 10])
            state.append([-5, 5, 10])

            world_state['walls'] = state

        return world_state

    def get_image(self):
        if self.use_img:
            return self.render()
        else:
            return None

    def render(self, mode='human', close=False, dist='close'):
        if mode == "human":
            self._renders = True

        if dist == 'far':
            base_pos = [4.45, 4.45, 9.8]
            cam_dist = 0.1
            cam_pitch = -45.0
            cam_yaw = 45.0 + 90
            cam_roll = 0.0
            fov = 100

        elif dist == 'close':
            base_pos = [4.45, 4.45, 2.0]
            cam_dist = 0.1
            cam_pitch = -15.0
            cam_yaw = 45.0 + 90
            cam_roll = 0.0
            fov = 60

        elif dist == 'follow':
            base_pose, _ = self._p.getBasePositionAndOrientation(self.cartpole)
            pos, vel, jRF, aJMT = self._p.getJointStateMultiDof(self.cartpole, 0)

            x = pos[0] + base_pose[0]
            y = pos[1] + base_pose[1]

            base_pos = [x, y, 2.0]
            cam_dist = 0.1
            cam_pitch = -15.0
            cam_yaw = 45.0 + 90
            cam_roll = 0.0
            fov = 60

        if self._physics_client_id >= 0:
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=cam_dist,
                yaw=cam_yaw,
                pitch=cam_pitch,
                roll=cam_roll,
                upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=fov,
                                                             aspect=float(self._render_width) /
                                                                    self._render_height,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(
                width=self._render_width,
                height=self._render_height,
                renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix)
        else:
            px = np.array([[[255, 255, 255, 255]] * self._render_width] * self._render_height, dtype=np.uint8)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


euler_keys = ['x', 'y', 'z']
quaternion_keys = ['x_quaternion', 'y_quaternion', 'z_quaternion', 'w_quaternion']
euler_pos_keys = [k+'_position' for k in euler_keys]
euler_vel_keys = [k+'_velocity' for k in euler_keys]
quaternion_vel_keys = [k+'_velocity' for k in quaternion_keys]


# convert the WSU feature vector into the observation input format for the agent
def convert_features_to_obs(feature_vec):
  #print(feature_vec)
  cart_posx = feature_vec["cart"]["x_position"]
  cart_posy = feature_vec["cart"]["y_position"]
  cart_posz = feature_vec["cart"]["z_position"]

  pole_xq = feature_vec["pole"]["x_quaternion"]
  pole_yq = feature_vec["pole"]["y_quaternion"]
  pole_zq = feature_vec["pole"]["z_quaternion"]
  pole_wq = feature_vec["pole"]["w_quaternion"]

  cart_vx = feature_vec["cart"]["x_velocity"]
  cart_vy = feature_vec["cart"]["y_velocity"]
  cart_vz = feature_vec["cart"]["z_velocity"]

  pole_vx = feature_vec["pole"]["x_velocity"]
  pole_vy = feature_vec["pole"]["y_velocity"]
  pole_vz = feature_vec["pole"]["z_velocity"]

  blocks_vec = []
  blocks = feature_vec['blocks']
  for block in blocks:
    blocks_vec.append([block['x_position'], block['y_position'], block['z_position'],
                       block['x_velocity'], block['y_velocity'], block['z_velocity']])
  if 'walls' in feature_vec.keys():
    walls = feature_vec['walls']

  cart_pos = [cart_posx, cart_posy, cart_posz]
  cart_vel = [cart_vx, cart_vy, cart_vz]
  pole_ori = [pole_xq, pole_yq, pole_zq, pole_wq]
  pole_vel = [pole_vx, pole_vy, pole_vz]

  cart_vec = [cart_pos, cart_vel]
  pole_vec = [pole_ori, pole_vel]
  return [cart_vec, pole_vec, blocks_vec]

def covert_obj_var_dicts_to_tuples(obj_var_dict):
  obj_state = dict()

  # cart
  obj_state["cart"] = (
    obj_var_dict["cart"]['x_position'],
    obj_var_dict["cart"]['y_position'],
    obj_var_dict["cart"]['z_position'],
    obj_var_dict["cart"]['x_velocity'],
    obj_var_dict["cart"]['y_velocity'],
    obj_var_dict["cart"]['z_velocity']
    )

  # pole
  obj_state["pole"] = (
    obj_var_dict["pole"]['x_quaternion'],
    obj_var_dict["pole"]['y_quaternion'],
    obj_var_dict["pole"]['z_quaternion'],
    obj_var_dict["pole"]['w_quaternion'],
    obj_var_dict["pole"]['x_velocity'],
    obj_var_dict["pole"]['y_velocity'],
    obj_var_dict["pole"]['z_velocity']
  )

  # blocks
  # note that blocks are sorted in the order of insertion to the dict
  # only since python3.6
  for b, block_state in enumerate(obj_var_dict["blocks"]):
    # print(block_state["id"])
    obj_state["block-"+str(b)] = (
      block_state['x_position'],
      block_state['y_position'],
      block_state['z_position'],
      block_state['x_velocity'],
      block_state['y_velocity'],
      block_state['z_velocity']
      )

  return obj_state


def action_to_wsu_output(action: int):
  assert action < len(WSUActionEnum)
  return WSUActionEnum[action].name


def euler_to_quaternion(yaw, pitch, roll):
  qx = np.sin(yaw / 2) * np.sin(pitch / 2) * np.cos(roll / 2) + np.cos(yaw / 2) * np.cos(pitch / 2) * np.sin(
    roll / 2)
  qy = np.sin(yaw / 2) * np.cos(pitch / 2) * np.cos(roll / 2) + np.cos(yaw / 2) * np.sin(pitch / 2) * np.sin(
    roll / 2)
  qz = np.cos(yaw / 2) * np.sin(pitch / 2) * np.cos(roll / 2) - np.sin(yaw / 2) * np.cos(pitch / 2) * np.sin(
    roll / 2)
  qw = np.cos(yaw / 2) * np.cos(pitch / 2) * np.cos(roll / 2) - np.sin(yaw / 2) * np.sin(pitch / 2) * np.sin(
    roll / 2)

  return (qx, qy, qz, qw)


def quaternion_to_euler(x, y, z, w):
  ysqr = y * y

  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + ysqr)
  X = np.degrees(np.arctan2(t0, t1))

  t2 = +2.0 * (w * y - z * x)
  t2 = np.where(t2 > +1.0, +1.0, t2)
  # t2 = +1.0 if t2 > +1.0 else t2

  t2 = np.where(t2 < -1.0, -1.0, t2)
  # t2 = -1.0 if t2 < -1.0 else t2
  Y = np.degrees(np.arcsin(t2))

  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (ysqr + z * z)
  Z = np.degrees(np.arctan2(t3, t4))

  return (X, Y, Z)

def quaternion_is_norm(x, y, z, w):
  q = [x, y, z, w]
  norm = sum([c**2 for c in q])
  return np.isclose(norm, 1., atol=1e4)

def calc_state_object_differences(s_0, s_1, verbose=False):
  obj_state_0 = covert_obj_var_dicts_to_tuples(s_0)
  obj_state_1 = covert_obj_var_dicts_to_tuples(s_1)
  cdist = euler_step_distance(s_0, s_1, 'cart', euler_pos_keys)
  cvel = euler_step_distance(s_0, s_1, 'cart', euler_vel_keys)
  pdist = report_quaternion_step_differences_by_object(s_0, s_1, 'pole', quaternion_keys)

  # calculate difference in angular velocity of pole
  # convert pole velocity euler vales to quaternion
  pvq0 = euler_to_quaternion(s_0['pole'][euler_vel_keys[0]],
                             s_0['pole'][euler_vel_keys[1]],
                             s_0['pole'][euler_vel_keys[2]])  # pole velocity quaternion for s_0
  pvq1 = euler_to_quaternion(s_1['pole'][euler_vel_keys[0]],
                             s_1['pole'][euler_vel_keys[1]],
                             s_1['pole'][euler_vel_keys[2]])  # pole velocity quaternion for s_1
  for i, k in enumerate(quaternion_vel_keys):
    s_0['pole'][k] = pvq0[i]
    s_1['pole'][k] = pvq1[i]
  # compute distance between quaternions
  pvel = report_quaternion_step_differences_by_object(s_0, s_1, 'pole', quaternion_vel_keys)
  nb = len(s_0['blocks'])
  assert nb == len(s_1['blocks'])
  if nb:
    bdists = [euler_step_distance(s_0['blocks'][b], s_1['blocks'][b], 'block', euler_pos_keys)
              for b in range(nb)]
    bvels = [euler_step_distance(s_0['blocks'][b], s_1['blocks'][b], 'block', euler_vel_keys)
              for b in range(nb)]
  else:
    bdists = []
    bvels = []

  # TODO: check for additional objects in the json

  if verbose>=1:
    print("cart position dist: ", "{:.2e}".format(cdist),
      "\tenv0:", " ".join(["%6.3f"%e for e in obj_state_0["cart"][0:3]]),
      "\t        env1:", " ".join(["%6.3f"%e for e in obj_state_1["cart"][0:3]]),
      "\t              cart velocity dist: ", "{:.2e}".format(cvel),
      "\tenv0:", " ".join(["%6.3f"%e for e in obj_state_0["cart"][3:6]]),
      "\tenv1:", " ".join(["%6.3f"%e for e in obj_state_1["cart"][3:6]]))
    print("pole position dist: ", "{:.2e}".format(pdist),
      "\tenv0:", " ".join(["%6.3f"%e for e in obj_state_0["pole"][0:4]]),
      "\tenv1:", " ".join(["%6.3f"%e for e in obj_state_1["pole"][0:4]]),
      "\t      pole velocity dist: ", "{:.2e}".format(pvel),
      "\tenv0:", " ".join(["%6.3f"%e for e in obj_state_0["pole"][4:7]]),
      "\tenv1:", " ".join(["%6.3f"%e for e in obj_state_1["pole"][4:7]]))

    if verbose>=2:
      print(f'block position dist: ', '{:.2e}'.format(sum(bdists)))
      for b in range(nb):
        print(
          "  block-"+str(b), "\tstate 0:", obj_state_0['block-'+str(b)][0:3],
          "  block-"+str(b), "\tstate 1:", obj_state_1['block-'+str(b)][0:3] )
      print(f'block velocity dist: ', '{:.2e}'.format(sum(bvels)))
      for b in range(nb):
        print(
          "  block-"+str(b), "\tstate 0:", obj_state_0['block-'+str(b)][3:6],
          "  block-"+str(b), "\tstate 1:", obj_state_1['block-'+str(b)][3:6] )


  return [cdist, cvel, pdist, pvel] + bdists + bvels

def calc_state_difference(s_0, s_1, verbose=False):
  all_obj_dists = calc_state_object_differences(s_0, s_1, verbose=verbose)
  nblocks = len(s_0['blocks'])
  cdist, cvel, pdist, pvel = all_obj_dists[:4]
  bdists = all_obj_dists[4:4+nblocks]
  bvels = all_obj_dists[4+nblocks:]
  assert len(bdists) == len(bvels)

  cartpole_dist = sum(np.asarray([cdist, cvel, pdist, pvel]))
  blocks_dist = sum(np.asarray(bdists + bvels))
  if verbose:
    print('cartpole dist: ', "{:.2e}".format(cartpole_dist), '\tblocks dist: ', "{:.2e}".format(blocks_dist))
  return cartpole_dist, blocks_dist

def euler_step_distance(s_0, s_1, object_key, vector_keys):
  if object_key == 'block':
    c0 = s_0
    c1 = s_1

  elif object_key in ['cart', 'pole']:
    c0 = s_0[object_key]
    c1 = s_1[object_key]

  else:
    raise ValueError('Not recognized: ' + object_key)

  obj0 = [c0[k] for k in vector_keys]
  obj1 = [c1[k] for k in vector_keys]
  dist = np.linalg.norm(np.array(obj0) - np.array(obj1))
  return dist

def report_quaternion_step_differences_by_object(s_0, s_1, object_key, vector_keys):
  if object_key in ['pole']:
    c0 = s_0[object_key]
    c1 = s_1[object_key]
    obj0 = [c0[v] for v in vector_keys]
    obj1 = [c1[v] for v in vector_keys]

    for i, obj in enumerate([obj0, obj1]):
      if not quaternion_is_norm(obj[0], obj[1], obj[2], obj[3]):
        print(f'object {i} quaternion not normalized: ', object_key, vector_keys)

    # compute distance between quaternions
    # https://math.stackexchange.com/questions/90081/quaternion-distance
    # assumes quats are normalized
    inner_prod_quats = sum([obj0[i]*obj1[i] for i, _ in enumerate(vector_keys)])
    #dist = math.acos(2*(inner_prod_quats**2)-1) # angular distance between quaternions
    dist = 1-inner_prod_quats**2   # approximates the distance; 0 when same orientation and 1 when 180 degrees apart
    if False:
      print(dist, obj0, obj1, object_key)

  else:
    raise ValueError('Not recognized: ', object_key)

  return dist

def compare_parallel_trajectories(condition_name, env_version, actions):
    path = 'urdf-files/'
    params = {
        'path':path, 'use_gui':False, 'use_img':False,
        'config':{'episode_seed':123, 'start_zeroed_out':True}, 'seed':1
        }

    if env_version=="077":
        env0 = CartPoleBulletEnv_077(path)
        env1 = CartPoleBulletEnv_077(path)
    elif env_version=="078":
        env0 = CartPoleBulletEnv_078(params)
        env1 = CartPoleBulletEnv_078(params)


    if "yaw-mismatch" in condition_name:
        env1.path = 'urdf-files/yaw90/'
        # env1.path = 'urdf-files/yaw45/'

    print("\n======================== comparing two trajectories in env", env_version,
        "under condition", condition_name,
        " ==========================")

    env0.reset()
    env1.reset()
    env1.set_world(env0.get_state())
    initial_state1 = env0.get_state()
    initial_state2 = env1.get_state()
    print(f"#### initial state ####")
    calc_state_difference(initial_state1, initial_state2, verbose=True)
    for i in range(len(actions)):
        # print()
        action = actions[i]
        if "master-slave" in condition_name:
            env1.set_world(env0.get_state())
        next_state1, _, _, _ = env0.step(action)
        next_state2, _, _, _ = env1.step(action)
        print(f"#### step {i}, action: {action} ####")
        calc_state_difference(next_state1, next_state2, verbose=True)



if __name__ == '__main__':

    # problem 1: diminishing predictability after a large number of steps
    # n_steps = 100
    # actions = np.random.choice(
    #     ["nothing", "right", "left", "forward", "backward"], n_steps)
    # compare_parallel_trajectories("master-slave", "077", actions)
    # compare_parallel_trajectories("master-slave", "078", actions)

    # a small number of steps is sufficient for the remaining tests
    actions = ["right", "right", "right","left", "forward", "backward"]

    # compare trajectories when the cart in one of the envs has a non-zero yaw
    compare_parallel_trajectories("vanilla", "077", actions)
    compare_parallel_trajectories("vanilla", "078", actions)

    # error when one trajectory (slave) follows another trajectory (master)
    compare_parallel_trajectories("master-slave", "077", actions)
    compare_parallel_trajectories("master-slave", "078", actions)

    # compare trajectories when the cart in one of the envs has a non-zero yaw
    compare_parallel_trajectories("yaw-mismatch", "077", actions)
    compare_parallel_trajectories("yaw-mismatch", "078", actions)

    # compare trajectories when the cart in one of the envs has a non-zero yaw
    compare_parallel_trajectories("master-slave_yaw-mismatch", "077", actions)
    compare_parallel_trajectories("master-slave_yaw-mismatch", "078", actions)

