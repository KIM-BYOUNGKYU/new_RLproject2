import numpy as np
import random
import cv2
from aerodynamics import Aerodynamics as aeroModel
import policy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def rotation_matrix(roll, pitch, yaw):
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # Roll에 대한 회전 행렬
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    # Pitch에 대한 회전 행렬
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    # Yaw에 대한 회전 행렬
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # 회전 행렬 결합
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    return R

def transform_coordinates(coordinates, roll, pitch, yaw):
    R = rotation_matrix(roll, pitch, yaw)
    transformed_coordinates = np.dot(R, coordinates)
    return transformed_coordinates

def calculate_air_density(altitude):
    # 기본 상수
    R = 287.05  # 이상기체 상수, J/(kg·K)
    g = 9.80665  # 중력 가속도, m/s²
    T0 = 288.15  # 해수면 기준 온도, K
    P0 = 101325  # 해수면 기준 압력, Pa
    L = 0.0065   # 기온 감소율, K/m
    M = 0.0289644  # 건조 공기의 몰 질량, kg/mol

    # 고도에 따른 온도
    T = T0 - L * altitude
    
    # 고도에 따른 압력
    P = P0 * (1 - L * altitude / T0) ** (g * M / (R * L))

    # 공기 밀도
    density = P / (R * T)
    
    return density

class Rocket(object):
    def __init__(self, max_steps=10000, task='launching', rocket_type='falcon-4',
                 viewport_h=768, path_to_bg_img=None, fault=False):

        self.task = task
        

        #기본적인 환경 configuration
        self.G = 6.6742*10**-11;            #%%Gravitational constant (SI Unit)
        self.g = 9.8                        #지표에서의 중력가속도
        self.dt = 0.25                      # step의 시간간격
        self.max_step= max_steps
        self.R_planet = 6371000                # 지구반지름 TODO: 단위가 m임
        self.M_planet = 5.972*(10**(24))
        self.max_thrust = [6804000, 934000,0]       # 최대 추력, 1단과 2단이 다름, 모든 엔진값의 합 -> 나중에 엔진당 추력으로 변환 필요

        #rocket configuration
        self.rocket_type = rocket_type
        self.D = 3.7                                 # rocket diameter (meters)
        self.H = [70,27.4,13.1]                          # rocket height (meters) 
        self.I = [[1000,1000,1000000],
                 [100,100,1000],
                 [10,10,100]]                       # stage별 3축의 Moment of inertia TODO: 관성 모멘트가 나와 있지 않음
        self.mass=[549054, 2000, 22800]                # rocket의 stage별 질량 [초기질량, 1단 분리 이후 질량, 2단 분리 이후 인공위성 질량]
        self.fuel_mass=[411000, 1800, 0]               # stage별 가용 연료 질량 TODO: 1단 분리 이후 질량과 연료값? / 연소시간기준 1단 162초, 2단 397초
        self.d_CM_e=[(0,0,-25),(0,0,-5),(0,0,0)]    # stage별 질량중심과 엔진사이의 거리
        self.CD = [0.4839, 0, 0]                    # stage별 coefficient of drag 공기저항 계수입니다.
        self.current_mass = self.mass[0]
    
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

        #self.engine_actor = policy.ActorCritic(input_dim=len(self.flatten(self.state)), output_dim=len(3*sum(self.num_engines)+1))
        #self.action_table = self.create_action_table()
        self.state_dims = len(self.state)
        #self.action_dims = len(self.action_table)
        
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

    def create_action_table(self): # unused
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
        altitude = distance - self.R_planet ##altitude above the surface
        rho = calculate_air_density(altitude) ##air density
        V = np.sqrt(velocity.dot(velocity))
        qinf = (np.pi/8.0)*rho*(self.D**2)*abs(V)
        aeroF = -qinf*self.CD[self.state[5]]*velocity
        return aeroF

    def get_random_action(self):
        return random.randint(0, len(self.action_table)-1)

    def create_initial_state(self):
        # predefined locations
        position = np.array([0.0, 0.0, self.R_planet])
        velocity = np.array([0.0, 0.0, 0.0])
        angle = np.array([0.0, 0.0, 0.0])
        angular_v= np.array([0.0, 0.0, 0.0])
        state = [
            position, velocity, angle, angular_v, self.fuel_mass[0],0, np.zeros((8, 2))
        ]
        return state
    
    def get_gravity(self):
        x, y, z = self.state[0]
        r = np.sqrt(x**2 + y**2 + z**2)
    
        if r < self.R_planet:
            accelx = 0.0
            accely = 0.0
            accelz = 0.0
        else:
            accelx = -self.G*self.M_planet/(r**3)*x
            accely = -self.G*self.M_planet/(r**3)*y
            accelz = -self.G*self.M_planet/(r**3)*z
        
        return np.asarray([accelx,accely,accelz]),r
    
    def get_propulsion(self, action):
        #input: action
        #output: total_torque;nparray, total_thrust;nparray, mdot, 엔진 노즐의 angular velocity pair
        stage = self.state[5]   #state의 5번째 값은 current_Stage에 해당됨.
        
        NumofUsedEngines= sum(self.num_engines[0:stage]) 
        current_NumofEngines = self.num_engines[stage] 
        
        thrusts = action[NumofUsedEngines*3 + current_NumofEngines*2:NumofUsedEngines*3 + current_NumofEngines*3]
        angle_thrusts = self.state[6][NumofUsedEngines:NumofUsedEngines+current_NumofEngines]
        total_thrust = np.array([0.0,0.0,0.0])
        total_torque = np.array([0.0,0.0,0.0])
        mdot = 0
        if self.state[4] > 0: #연료가 있는 경우
            for i in range(len(thrusts)):
                x_thrust = -thrusts[i]*np.sin(angle_thrusts[i][0])*np.cos(angle_thrusts[i][1])
                y_thrust = -thrusts[i]*np.sin(angle_thrusts[i][0])*np.sin(angle_thrusts[i][1])
                z_thrust = thrusts[i]*np.cos(angle_thrusts[i][0])
                thrust = np.array([x_thrust,y_thrust,z_thrust])     # 엔진별 thrust vector 생성

                total_thrust +=thrust                            # thrust vector의 합                            
                total_torque +=np.cross(np.array(self.d_CM_e[stage])+np.array(self.r_engines[stage][i]),thrust)
            if self.state[5] != 2: # final stage가 아닌 경우
                mdot -= sum(thrusts) / (self.g * self.Isp[self.state[5]])# thrust에 따른 연료소모속도
        
        angular_velocity0 = action[0:self.num_engines[0]] + action[self.num_engines[0]*3:self.num_engines[0]*3+self.num_engines[1]]
        angular_velocity1 = action[self.num_engines[0]:self.num_engines[0]*2] + action[self.num_engines[0]*3+self.num_engines[1]:self.num_engines[0]*3+self.num_engines[1]*2]
        angular_velocity_pair = np.array(list(zip(angular_velocity0, angular_velocity1)))

        return total_torque, total_thrust, mdot, angular_velocity_pair
    
    def get_New_state(self, state, acc, angular_acc, mdot, d_angVofEngines, detach):
        #input : [position, velocity, rotational angle, angular velocity, feul 질량, current stage, engine angle], acc, angular_acc, engine angular V array, detach array 
        #output : derivatives of position, velocity, rotational angle, angular velocity, feul 질량, current stage, engine angle
        #유의점 : input의 position, velocity, rotational angle, angular velocity는 지구 좌표계 기준,
        #                engine angle, acc, angular_acc는 로켓 좌표계 기준으로 설정
        new_position = np.add(state[0], state[1]*self.dt)

        acc_Earth = transform_coordinates(acc,state[2][0],state[2][1],state[2][2])
        new_velocity = np.add(state[1], acc_Earth*self.dt)

        new_rot_angle = np.add(state[2], state[3]*self.dt)

        angular_acc_Earth = transform_coordinates(angular_acc,state[2][0],state[2][1],state[2][2])
        new_rot_angV = np.add(state[3], angular_acc_Earth*self.dt)

        if state[5] >= int(detach):         #분리가 일어나지 않는 경우
            new_stage = state[5]
            new_fuel = max(state[4]+ mdot*self.dt, 0)
            self.current_mass += new_fuel-state[5]
        elif state[5] == 2:                 #분리가 최대로 일어난 경우
            new_stage = state[5]
            new_fuel = max(state[4]+ mdot*self.dt, 0)
            self.current_mass += new_fuel-state[5]
        else:                               #분리가 일어나는 경우
            new_stage = min[int(detach),2]
            new_fuel = self.fuel_mass[new_stage]
            self.current_mass = self.mass[new_stage]

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
        gravity = g_acc*self.current_mass
        aeroF = self.get_aerofriction(r)
        Forces = gravity + aeroF + thrust

        vdot = Forces/self.current_mass
        wdot = np.divide(torque,self.I[stage])              

        new_state = self.get_New_state(self.state, vdot, wdot,mdot, d_ang_VofEngines, action[-1])
                                                            # timestep 지난 후 변경된 새 state return
        self.step_id += 1
        
        self.state_buffer.append(self.state)                # 기존 state buffer에 넣기
        self.state = new_state                              # 새 state update 

        self.already_crash = self.check_crash(self.state)
        #reward = self.calculate_reward(self.state)
        reward = 0
        if self.already_crash or self.step_id==self.max_step:       #문제가 생겨 더이상 진행하지 못하거나 정해진 시간이 전부 지난 경우
            done = True
        else:
            done = False

        return self.state, reward, done, None

    def show_path_from_state_buffer(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.R_planet * np.outer(np.cos(u), np.sin(v))
        y = self.R_planet * np.outer(np.sin(u), np.sin(v))
        z = self.R_planet * np.outer(np.ones(np.size(u)), np.cos(v))
    
        # 지구 표면 플롯
        #ax.plot_surface(x, y, z, color='b', alpha=0.3)

        # 스테이지별 색상 맵을 정의
        color_map = ['g', 'r', 'orange']

        # 각 상태 리스트에서 인덱스 0, 1, 2가 각각 x, y, z 좌표라고 가정
        stages = set(state[5] for state in self.state_buffer)  # 모든 스테이지 번호를 추출
        for stage in stages:
            # 해당 스테이지의 모든 상태를 추출
            stage_positions = np.array([state[0] for state in self.state_buffer if state[5] == stage])
            ax.plot(stage_positions[:, 0], stage_positions[:, 1], stage_positions[:, 2], label=f'Stage {stage}', color=color_map[stage % len(color_map)])

        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_zlabel('Z position')
        ax.legend()
        plt.show()

    

    def animate_trajectory(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        
        data = np.array([state[0] for state in self.state_buffer]).T
        line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], 'r-')

        ax.set_xlim([np.min(data[0])-1000, np.max(data[0])+1000])
        ax.set_ylim([np.min(data[1])-1000, np.max(data[1])+1000])
        ax.set_zlim([np.min(data[2])-1000, np.max(data[2])+1000])
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_zlabel('Z position')

        ani = FuncAnimation(fig, update, frames=len(self.state_buffer), fargs=(data, line), interval=0)
        plt.show()

def update(num, data, line, skip_steps=500):
    end = min(num * skip_steps + 1, data.shape[1])
    line.set_data(data[:2, :end])
    line.set_3d_properties(data[2, :end])