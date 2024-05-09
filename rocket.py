import numpy as np
import random
import cv2
import utils
from aerodynamics import Aerodynamics as aeroModel
import policy


class Rocket(object):
    def __init__(self, max_steps, task='launching', rocket_type='falcon-4',
                 viewport_h=768, path_to_bg_img=None, fault=False):

        self.task = task
        self.rocket_type = rocket_type
        self.G = 6.6742*10**-11; #%%Gravitational constant (SI Unit)
        self.g = 9.8
        self.current_stage = 0 #현재 rocket의 stage
        
        self.H = [50,10]  # rocket height (meters)
        self.I = 1/12*self.H*self.H  # Moment of inertia
        
        self.dt = 0.05
        
        self.R_planet = 00000
        self.mass=[10000, 2000, 100] #rocket의 stage별 질량 [초기질량, 1단 분리 이후 질량, 2단 분리 이후 인공위성 질량]
        self.CD = [0.4839, 0, 0]  #stage별 coefficient of drag 공기저항 계수입니다.
        self.num_engines = [5,3]
        self.r_engines = [3, 2] #각 단계의 엔진의 중심으로부터의 거리
        self.state = self.create_initial_state()
        self.engine_actor = policy.ActorCritic(input_dim=len(self.state), output_dim=len(2*sum(self.num_engines)))
        self.action_table = self.create_action_table()

        self.state_dims = 22 # x, z, vx, vz, theta, vtheta, phi0~4, phi5~7, f0~4, f5~7
        self.action_dims = len(self.action_table)

        '''if path_to_bg_img is None:
            path_to_bg_img = task+'.jpg'
        self.bg_img = utils.load_bg_img(path_to_bg_img, w=self.viewport_w, h=self.viewport_h)'''

        self.state_buffer = []

    def reset(self, state_dict=None):

        if state_dict is None:
            self.state = self.create_random_state()
        else:
            self.state = state_dict

        self.state_buffer = []
        self.step_id = 0
        self.already_landing = False
        cv2.destroyAllWindows()
        return self.flatten(self.state)

    def create_action_table(self):
        f0 = 0.2 * self.g  # thrust
        f1 = 1.0 * self.g
        f2 = 2 * self.g
        vphi0 = 0  # Nozzle angular velocity
        vphi1 = 30 / 180 * np.pi
        vphi2 = -30 / 180 * np.pi

        action_table = [[f0, vphi0], [f0, vphi1], [f0, vphi2],
                        [f1, vphi0], [f1, vphi1], [f1, vphi2],
                        [f2, vphi0], [f2, vphi1], [f2, vphi2]
                        ]
        return action_table

    def get_random_action(self):
        return random.randint(0, len(self.action_table)-1)

    def create_initial_state(self):
        # predefined locations
        x0 = self.R_planet 
        z0 = 0.0
        velz0 = 0.0
        velx0 = 0.0
        theta = 0
        m = self.mass[0]
        state = {
            'x': x0, 'z': z0, 'vx': velx0, 'vz': velz0,
            'theta': theta, 'vtheta': 0,
            'phi': [0,0,0,0,0,0,0,0], 'f': [0,0,0,0,0,0,0,0],
            't': 0, 'a_': 0, 'mass':m
        }
        return state
    
    def get_gravity(self):
        global Rplanet,mplanet
        x, y, z = self.state[0:3]    
        r = np.sqrt(x**2 + y**2 + z**2)
    
        if r < Rplanet:
            accelx = 0.0
            accely = 0.0
            accelz = 0.0
        else:
            accelx = -self.G*mplanet/(r**3)*x
            accely = -self.G*mplanet/(r**3)*y
            accelz = -self.G*mplanet/(r**3)*z
        
        return np.asarray([accelx,accely,accelz]),r
    
    def get_propulsion(self, action):
        #input: action
        #output: total_torque, total_thrust, mdot 
        NumofUsedEngines= sum(self.num_engines[0:self.state[5]]) #state의 5번째 값은 current_Stage에 해당됨.
        current_NumofEngines = self.num_engines[self.state[5]] 
        thrusts = action[NumofUsedEngines*3 + current_NumofEngines*2:NumofUsedEngines*3 + current_NumofEngines*3]
        angle_thrusts = self.state[6][NumofUsedEngines:NumofUsedEngines+current_NumofEngines]
        x_thrust=0
        y_thrust=0
        z_thrust=0
        
        for i in range(len(thrusts)):
            z_thrust += -thrusts[i]*np.cos(angle_thrusts[i][1])
            x_thrust += thrusts[i]*np.sin(angle_thrusts[i][1])*np.cos(angle_thrusts[i][0])
            y_thrust += thrusts[i]*np.sin(angle_thrusts[i][1])*np.sin(angle_thrusts[i][0])
        
        behavior = self.engine_actor.get_action(state)
        stage = self.current_stage
        if current_stage == 
        thrust = 
        return thrust, [x_thrust, y_thrust, z_thrust]
    
    def get_Derivatives(self, state, t):
        #input : state; x, z좌표, x, z방향 속도, 질량, 
        x = state[0]
        z = state[1]
        velx = state[2]
        velz = state[3]
        mass = state[4]
        xdot = velx
        zdot = velz
        
        #gravitational force
        g_a, r = self.get_gravity(x,z)
        gravityF = g_a*mass
        
        #dragging force
        altitude = r - Rplanet ##altitude above the surface
        rho = aeroModel.getDensity(altitude) ##air density
        V = np.sqrt(velz**2+velx**2)
        qinf = (np.pi/8.0)*rho*(D**2)*abs(V)
        aeroF = -qinf*self.CD*np.asarray([velx,velz])
        thrustF,mdot = propulsion(state)
    
    def check_crash(self, state):
        if self.task == 'hover':
            x, y = state['x'], state['y']
            theta = state['theta']
            crash = False
            if y <= self.H / 2.0:
                crash = True
            if y >= self.world_y_max - self.H / 2.0:
                crash = True
            return crash

        elif self.task == 'landing':
            x, y = state['x'], state['y']
            vx, vy = state['vx'], state['vy']
            theta = state['theta']
            vtheta = state['vtheta']
            v = (vx**2 + vy**2)**0.5

            crash = False
            if y >= self.world_y_max - self.H / 2.0:
                crash = True
            if y <= 0 + self.H / 2.0 and v >= 15.0:
                crash = True
            if y <= 0 + self.H / 2.0 and abs(x) >= self.target_r:
                crash = True
            if y <= 0 + self.H / 2.0 and abs(theta) >= 10/180*np.pi:
                crash = True
            if y <= 0 + self.H / 2.0 and abs(vtheta) >= 10/180*np.pi:
                crash = True
            return crash

    def calculate_reward(self, state):

        x_range = self.world_x_max - self.world_x_min
        y_range = self.world_y_max - self.world_y_min

        # dist between agent and target point
        dist_x = abs(state['x'] - self.target_x)
        dist_y = abs(state['y'] - self.target_y)
        dist_norm = dist_x / x_range + dist_y / y_range

        dist_reward = 0.1*(1.0 - dist_norm)

        if abs(state['theta']) <= np.pi / 6.0:
            pose_reward = 0.1
        else:
            pose_reward = abs(state['theta']) / (0.5*np.pi)
            pose_reward = 0.1 * (1.0 - pose_reward)

        reward = dist_reward + pose_reward

        if self.task == 'hover' and (dist_x**2 + dist_y**2)**0.5 <= 2*self.target_r:  # hit target
            reward = 0.25
        if self.task == 'hover' and (dist_x**2 + dist_y**2)**0.5 <= 1*self.target_r:  # hit target
            reward = 0.5
        if self.task == 'hover' and abs(state['theta']) > 90 / 180 * np.pi:
            reward = 0

        v = (state['vx'] ** 2 + state['vy'] ** 2) ** 0.5
        if self.task == 'landing' and self.already_crash:
            reward = (reward + 5*np.exp(-1*v/10.)) * (self.max_steps - self.step_id)
        if self.task == 'landing' and self.already_landing:
            reward = (1.0 + 5*np.exp(-1*v/10.))*(self.max_steps - self.step_id)

        return reward

    def step(self, action):
        stage = self.current_stage 
        for act in action:
        action

        x, y, vx, vy = self.state['x'], self.state['y'], self.state['vx'], self.state['vy']
        theta, vtheta = self.state['theta'], self.state['vtheta']
        phi = self.state['phi']

        f, vphi = self.action_table[action]

        ft, fr = -f*np.sin(phi), f*np.cos(phi)
        fx = ft*np.cos(theta) - fr*np.sin(theta)
        fy = ft*np.sin(theta) + fr*np.cos(theta)

        rho = 1 / (125/(self.g/2.0))**0.5  # suppose after 125 m free fall, then air resistance = mg
        ax, ay = fx-rho*vx, fy-self.g-rho*vy
        atheta = ft*self.H/2 / self.I

        # update agent
        if self.already_landing:
            vx, vy, ax, ay, theta, vtheta, atheta = 0, 0, 0, 0, 0, 0, 0
            phi, f = 0, 0
            action = 0

        self.step_id += 1
        x_new = x + vx*self.dt + 0.5 * ax * (self.dt**2)
        y_new = y + vy*self.dt + 0.5 * ay * (self.dt**2)
        vx_new, vy_new = vx + ax * self.dt, vy + ay * self.dt
        theta_new = theta + vtheta*self.dt + 0.5 * atheta * (self.dt**2)
        vtheta_new = vtheta + atheta * self.dt
        phi = phi + self.dt*vphi

        phi = max(phi, -20/180*3.1415926)
        phi = min(phi, 20/180*3.1415926)

        self.state = {
            'x': x_new, 'y': y_new, 'vx': vx_new, 'vy': vy_new,
            'theta': theta_new, 'vtheta': vtheta_new,
            'phi': phi, 'f': f,
            't': self.step_id, 'action_': action
        }
        self.state_buffer.append(self.state)

        self.already_landing = self.check_landing_success(self.state)
        self.already_crash = self.check_crash(self.state)
        reward = self.calculate_reward(self.state)

        if self.already_crash or self.already_landing:
            done = True
        else:
            done = False

        return self.flatten(self.state), reward, done, None
