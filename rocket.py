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
        

        #기본적인 환경 configuration
        self.G = 6.6742*10**-11;            #%%Gravitational constant (SI Unit)
        self.g = 9.8                        #지표에서의 중력가속도
        self.dt = 0.05                      # step의 시간간격
        self.R_planet = 6371                # 지구반지름 TODO: 단위가 km임

        self.max_thrust = [6804, 934]       # 최대 추력, 1단과 2단이 다름, 모든 엔진값의 합 -> 나중에 엔진당 추력으로 변환 필요

        #rocket configuration
        self.rocket_type = rocket_type
        self.D = 3.7                                 # rocket diameter (meters)
        self.H = [70,27.4,13.1]                          # rocket height (meters) 
        self.I = [[10000,1000,1000],
                 [1000,100,100],
                 [100,10,10]]                       # stage별 3축의 Moment of inertia TODO: 관성 모멘트가 나와 있지 않음
        self.mass=[549054, 2000, 22800]                # rocket의 stage별 질량 [초기질량, 1단 분리 이후 질량, 2단 분리 이후 인공위성 질량]
        self.fuel_mass=[411000,1800, 0]               # stage별 가용 연료 질량 TODO: 1단 분리 이후 질량과 연료값? / 연소시간기준 1단 162초, 2단 397초
        self.d_CM_e=[(0,0,-25),(0,0,-5),(0,0,0)]    # stage별 질량중심과 엔진사이의 거리
        self.CD = [0.4839, 0, 0]                    # stage별 coefficient of drag 공기저항 계수입니다.
        
        #rocket engine configuration
        self.num_engines = [5,3,1]           # stage별 engine의 개수
        self.r_engines = [[(1,0,0),(0,1,0),(-1,0,0),(0,-1,0),(0,0,0)], 
                          [(1,0,0),(-1/2, np.cos(np.pi/6),0),(-1/2, -np.cos(np.pi/6),0)],
                          [(0,0,0)]]     
                                            # stage의 엔진의 로켓 중심으로부터의 위치
        self.Isp = [297, 348]                   # 로켓 엔진의 specific Impulse TODO: 1단과 2단의 비추력값이 다름

        #rocket 현재 상황
        self.state = self.create_initial_state()
        self.step_id = 0
        self.state_buffer = []
        self.already_crash = False
        self.engine_actor = policy.ActorCritic(input_dim=len(self.flatten(self.state)), output_dim=len(3*sum(self.num_engines)+1))
        self.action_table = self.create_action_table()
        self.state_dims = len(self.state)
        self.action_dims = len(self.action_table)
        

    def reset(self, state_dict=None):

        if state_dict is None:
            self.state = self.create_initial_state()
        else:
            self.state = state_dict

        self.state_buffer = []
        self.step_id = 0
        self.already_crach = False
        cv2.destroyAllWindows()
        return self.state

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

    def get_aerofriction(self, distance):
        #input: distance from the centor of the Earth
        #output: aerofriction vector

        stage = self.state[5]
        velocity = self.state[1]
        altitude = distance - Rplanet ##altitude above the surface
        rho = aeroModel.getDensity(altitude) ##air density
        V = np.sqrt(velocity.dot(velocity))
        qinf = (np.pi/8.0)*rho*(self.D**2)*abs(V)
        aeroF = -qinf*self.CD*velocity
        return aeroF

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
        #output: total_torque, total_thrust, mdot, 엔진 노즐의 angular velocity pair
        stage = self.state[5]   #state의 5번째 값은 current_Stage에 해당됨.
        
        NumofUsedEngines= sum(self.num_engines[0:stage]) 
        current_NumofEngines = self.num_engines[stage] 
        
        thrusts = action[NumofUsedEngines*3 + current_NumofEngines*2:NumofUsedEngines*3 + current_NumofEngines*3]
        angle_thrusts = self.state[6][NumofUsedEngines:NumofUsedEngines+current_NumofEngines]
        
        if self.state[4] <= 0: #연료가 없는 경우
            total_torque = np.array([0,0,0])
            total_thrust = np.array([0,0,0])
            mdot = 0
        else:
            for i in range(len(thrusts)):
                x_thrust = thrusts[i]*np.sin(angle_thrusts[i][1])*np.cos(angle_thrusts[i][0])
                y_thrust = thrusts[i]*np.sin(angle_thrusts[i][1])*np.sin(angle_thrusts[i][0])
                z_thrust = -thrusts[i]*np.cos(angle_thrusts[i][1])
                thrust = np.array([x_thrust,y_thrust,z_thrust])     # 엔진별 thrust vector 생성

                total_thrust += thrust                              # thrust vector의 합                            
                total_torque += np.cross(np.array(self.d_CM_e[stage])+np.array(self.r_engines[stage][i]),thrust)        
            mdot = - sum(thrusts) / (self.g * self.Isp)             # thrust에 따른 연료소모속도
        
        angular_velocity0 = action[0:self.num_engines[0]] + action[self.num_engines[0]*3:self.num_engines[0]*3+self.num_engines[1]]
        angular_velocity1 = action[self.num_engines[0]:self.num_engines[0]*2] + action[self.num_engines[0]*3+self.num_engines[1]:self.num_engines[0]*3+self.num_engines[1]*2]
        angular_velocity_pair = np.array(list(zip(angular_velocity0, angular_velocity1)))

        return total_torque, total_thrust, mdot, angular_velocity_pair
    
    def get_New_state(self, state, acc, angular_acc, mdot, d_angVofEngines, detach):
        #input : [position, velocity, rotational angle, angular velocity, feul 질량, current stage, engine angle], acc, angular_acc, engine angular V array, detach array 
        #output : derivatives of position, velocity, rotational angle, angular velocity, feul 질량, current stage, engine angle
    
        new_position = np.add(state[0], state[1]*self.dt)
        new_velocity = np.add(state[1], acc*self.dt)
        new_rot_angle = np.add(state[2], state[3]*self.dt)
        new_rot_angV = np.add(state[3], angular_acc*self.dt)
        if state[5] >= int(detach):         #분리가 일어나지 않는 경우
            new_stage = state[5]
            new_feul = max(state[4]+ mdot*self.dt, 0)
        elif state[5] == 2:                 #분리가 최대로 일어난 경우
            new_stage = state[5]
            new_fuel = max(state[4]+ mdot*self.dt, 0)
        else:                               #분리가 일어나는 경우
            new_stage = min[int(detach),2]
            new_feul = self.fuel_mass[new_stage]
        new_engine_angle = np.add(state[6], d_angVofEngines*self.dt)
        return [new_position, new_velocity, new_rot_angle, new_rot_angV, new_fuel, new_stage, new_engine_angle] 

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

        stage = self.state[5]

        torque , thrust, mdot, d_ang_VofEngines = self.get_propulsion(action)
        g_acc,r = self.get_gravity()
        gravity = g_acc*self.mass[stage]
        aeroF = self.get_aerofriction(r)
        Forces = gravity + aeroF + thrust

        vdot = Forces/self.mass[stage]
        wdot = torque/self.I[stage]                         # 각가속도 구하는 부분인데 잘못 구현함. 이 부분도 나중에 수정 필요

        new_state = self.get_New_state(self.state, vdot, wdot,mdot, d_ang_VofEngines, action[-1])
                                                            # timestep 지난 후 변경된 새 state return
        self.step_id += 1
        
        self.state_buffer.append(self.state)                # 기존 state buffer에 넣기
        self.state = new_state                              # 새 state update 

        self.already_crash = self.check_crash(self.state)
        reward = self.calculate_reward(self.state)

        if self.already_crash or self.step_id==10000:       #문제가 생겨 더이상 진행하지 못하거나 정해진 시간이 전부 지난 경우
            done = True
        else:
            done = False

        return self.state, reward, done, None
