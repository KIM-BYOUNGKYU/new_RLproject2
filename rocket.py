import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from ambiance import Atmosphere
import gymnasium as gym
from gymnasium import spaces


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

class Rocket(object):
    def __init__(self, max_steps=100000, task='launching', rocket_type='falcon-4',
                 viewport_h=768, path_to_bg_img=None, fault=False):

        self.task = task

        #기본적인 환경 configuration
        self.G = 6.6742*10**-11;            #%%Gravitational constant (SI Unit)
        self.g = 9.8                        #지표에서의 중력가속도
        self.dt = 0.05                      # step의 시간간격
        self.max_step= max_steps
        self.R_planet = 6371000                # 지구반지름 TODO: 단위가 m임
        self.M_planet = 5.972*(10**(24))
        self.max_thrust = [6804000, 934000,0]       # 최대 추력, 1단과 2단이 다름, 모든 엔진값의 합 -> 나중에 엔진당 추력으로 변환 필요

        #rocket configuration
        self.target_p = [80000, 170000, 200000]
        self.rocket_type = rocket_type
        self.D = 3.7                                 # rocket diameter (meters)
        self.H = [70,27.4,13.1]                          # rocket height (meters) 

        self.mass=[553000, 119800, 8300]                # rocket의 stage별 질량 [초기질량, 1단 분리 이후 질량, 2단 분리 이후 인공위성 질량]
        self.fuel_mass=[411000, 107500, 0]               # stage별 가용 연료 질량 TODO: 1단 분리 이후 질량과 연료값? / 연소시간기준 1단 162초, 2단 397초
        
        self.d_CM_e=[(0,0,-25),(0,0,-5),(0,0,0)]    # stage별 질량중심과 엔진사이의 거리
        self.CD = [0.4839, 0, 0]                    # stage별 coefficient of drag 공기저항 계수입니다.
        
        #rocket engine configuration
        self.num_engines = [5,3,1]           # stage별 engine의 개수
        self.r_engines = [[(1,0,0),(0,1,0),(-1,0,0),(0,-1,0),(0,0,0)], 
                          [(1,0,0),(-1/2, np.cos(np.pi/6),0),(-1/2, -np.cos(np.pi/6),0)],
                          [(0,0,0)]]     
                                            # stage의 엔진의 로켓 중심으로부터의 위치 TODO
        self.Isp = [297, 348]                   # 로켓 엔진의 specific Impulse TODO: 1단과 2단의 비추력값이 다름 -> 처리 과정

        #rocket 현재 상황
        self.state = self.create_initial_state()
        self.current_mass = self.mass[0]
        self.I = [0.25 * self.current_mass * (self.D ** 2) + self.current_mass * (self.H[self.state[5]] ** 2) / 12, 
                  0.25 * self.current_mass * (self.D ** 2) + self.current_mass * (self.H[self.state[5]] ** 2) / 12,
                  0.5 * self.current_mass * (self.D ** 2)]                     # stage별 3축의 Moment of inertia, X축과 Y축의 관성 모멘트가 같다고 가정 TODO: mass와 관성 모멘트를 계속 업데이트 해줘야 함
        
        self.step_id = 0
        self.state_buffer = []
        self.already_crash = False

        #self.engine_actor = policy.ActorCritic(input_dim=len(self.flatten(self.state)), output_dim=len(3*sum(self.num_engines)+1))
        self.action_table = self.create_action_table()
        self.state_dims = len(Rocket.flatten(self.state))
        self.action_dims = 25
        

    def reset(self, state_dict=None):

        if state_dict is None:
            self.state = self.create_initial_state()
        else:
            self.state = state_dict
        self.state_buffer = []
        self.step_id = 0
        self.already_crash = False
        cv2.destroyAllWindows()
        return Rocket.flatten(self.state)

    def create_action_table(self): # unused
        f0 = 0.2 * self.g  # thrust
        f1 = 1.0 * self.g
        f2 = 2 * self.g
        vphi0 = 0  # Nozzle angular velocity
        vphi1 = 30 / 180 * np.pi
        vphi2 = -30 / 180 * np.pi
        detach = 0 # 분리 버튼

        action_table = [[f0, vphi0], [f0, vphi1], [f0, vphi2],
                        [f1, vphi0], [f1, vphi1], [f1, vphi2],
                        [f2, vphi0], [f2, vphi1], [f2, vphi2],
                        detach
                        ]
        return action_table

   
    def get_aerofriction(self, distance):
        #input: distance from the centor of the Earth
        #output: aerofriction vector

        velocity = self.state[1]
        altitude = distance - self.R_planet ##altitude above the surface

        if altitude > 80000:
            rho = 0
        else:
            rho = Atmosphere(altitude).density # air density

        V = np.sqrt(velocity.dot(velocity))
        qinf = (np.pi/8.0)*rho*(self.D**2)*abs(V)
        aeroF = -qinf*self.CD[self.state[5]]*velocity
        if np.isnan(aeroF).any():
            print(aeroF)
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
    
    def flatten(input_list) :
        output_list = []
        # 좌표, 속도, 회전 각도, 회전 각속도를 갖는 array의 요소를 순서대로 추가 (12개)
        for i in range(4):
            for element in input_list[i]:
                output_list.append(element)

        # 현재 연료 질량, stage 추가 (2개)
        output_list.append(input_list[4])
        output_list.append(input_list[5])

        # 노즐 각도 추가 (16개)
        for pair in input_list[6]:
            for element in pair:
                output_list.append(element)

        return output_list

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
    
    def get_propulsion(self, action): # rocket coordinate 기준으로 return
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
                thrust= np.array([x_thrust,y_thrust,z_thrust])     # 엔진별 thrust vector 생성
                
                total_thrust +=thrust                            # thrust vector의 합                            
                total_torque +=np.cross(np.array(self.d_CM_e[stage])+np.array(self.r_engines[stage][i]),thrust)
            if self.state[5] != 2: # final stage가 아닌 경우
                mdot -= sum(thrusts) / (self.g * self.Isp[self.state[5]])# thrust에 따른 연료소모속도
        
        angular_velocity0 = list(action[0:self.num_engines[0]]) + list(action[self.num_engines[0]*3:self.num_engines[0]*3+self.num_engines[1]])
        angular_velocity1 = list(action[self.num_engines[0]:self.num_engines[0]*2]) + list(action[self.num_engines[0]*3+self.num_engines[1]:self.num_engines[0]*3+self.num_engines[1]*2])
        angular_velocity_pair = np.array(list(zip(angular_velocity0, angular_velocity1)))

        return total_torque, total_thrust, mdot, angular_velocity_pair
    
    def get_New_state(self, state, acc_Earth, angular_acc, mdot, d_angVofEngines, detach):
        #input : [position, velocity, rotational angle, angular velocity, feul 질량, current stage, engine angle], acc, angular_acc, engine angular V array, detach array 
        #output : derivatives of position, velocity, rotational angle, angular velocity, feul 질량, current stage, engine angle
        
        new_position = np.add(state[0], state[1]*self.dt)

        new_velocity = np.add(state[1], acc_Earth*self.dt)
        if np.isnan(new_velocity).any():
            print(new_velocity)

        new_rot_angle = np.add(state[2], state[3]*self.dt)

        angular_acc_Earth = transform_coordinates(angular_acc,state[2][0],state[2][1],state[2][2])
        new_rot_angV = np.add(state[3], angular_acc_Earth*self.dt)

        if state[5] >= int(detach):         #분리가 일어나지 않는 경우
            new_stage = state[5]
            new_fuel = max(state[4]+ mdot*self.dt, 0)
            self.current_mass += new_fuel-state[4]
        elif state[5] == 2:                 #분리가 최대로 일어난 경우
            new_stage = state[5]
            new_fuel = max(state[4]+ mdot*self.dt, 0)
            self.current_mass += new_fuel-state[4]
        else:                               #분리가 일어나는 경우
            new_stage = min(int(detach),2)
            new_fuel = self.fuel_mass[new_stage]
            self.current_mass = self.mass[new_stage]

        new_engine_angle = np.add(state[6], d_angVofEngines*self.dt)

        return [new_position, new_velocity, new_rot_angle, new_rot_angV, new_fuel, new_stage, new_engine_angle] 
    
    def check_crash(self):
        crash = False
        x, y, z = self.state[0]
        vx,vy,vz = self.state[1]
        r = np.sqrt(x**2 + y**2 + z**2)
        if (r < self.R_planet):
            crash = True
        if vx==0 and vy==0 and vz==0:
            crash = True
            
        return crash

    def calculate_reward(self, state):
        # dist between agent and target point
        altitude = np.sqrt(state[0][0]**2+state[0][1]**2+state[0][2]**2)-self.R_planet
        dist_to_target = abs(self.target_p[state[5]]-altitude)
        reward = np.exp(-dist_to_target/1000)
        if dist_to_target<100:
            reward += 10
        
        return reward

    def step(self, action):

        stage = self.state[5]

        torque , thrust, mdot, d_ang_VofEngines = self.get_propulsion(action)
        g_acc,r = self.get_gravity()
    
        aeroF = self.get_aerofriction(r)
        
        vdot = g_acc + (aeroF + transform_coordinates(thrust, self.state[2][0], self.state[2][1], self.state[2][2]))/self.current_mass
        if np.isnan(vdot).any():
            print(vdot)
        wdot = torque /self.I
        
        new_state = self.get_New_state(self.state, vdot, wdot,mdot, d_ang_VofEngines, action[-1])
                                                            # timestep 지난 후 변경된 새 state return
        self.step_id += 1
        
        
        if np.isnan(self.state[1]).any():
            print(self.state[1])

        self.state_buffer.append(self.state)                # 기존 state buffer에 넣기
        self.state = new_state                              # 새 state update 

        self.already_crash = self.check_crash()
        reward = self.calculate_reward(self.state)
        
        if self.already_crash or self.step_id==self.max_step:       #문제가 생겨 더이상 진행하지 못하거나 정해진 시간이 전부 지난 경우
            done = True
        else:
            done = False

        return Rocket.flatten(self.state), reward, done, {}

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

    def animate_trajectory(self, skip_steps=1):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.R_planet * np.outer(np.cos(u), np.sin(v))
        y = self.R_planet * np.outer(np.sin(u), np.sin(v))
        z = self.R_planet* np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot Earth
        ax.plot_wireframe(x, y, z, color = 'g', alpha = 0.1)

        # 데이터 초기화 및 skip_steps 적용
        data = np.array([state[0] for state in self.state_buffer[::skip_steps]]).T
        line, = ax.plot([], [], [], 'r-', label="Trajectory")  # 빈 궤적 초기화
        marker, = ax.plot([], [], [], 'bo', markersize=5, label="Rocket")  # 로켓 마커 초기화

        # 설정 축 범위
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_zlabel('Z position')
        ax.legend()

        # 애니메이션 정의
        ani = FuncAnimation(fig, self.update, frames=len(data[0]), fargs=(data, line, marker, ax), interval=10) # interval 프레임간 시간 간격 ms
        plt.show()

    def update(self, num, data, line, marker, ax):
        # 마커 위치 업데이트
        if num < len(data[0]):
            marker.set_data(data[0, num], data[1, num])
            marker.set_3d_properties(data[2, num])

            # 궤적 데이터 업데이트
            line.set_data(data[0, :num+1], data[1, :num+1])
            line.set_3d_properties(data[2, :num+1])
            # 축 범위 재설정 (선택적)
            ax.set_xlim([0, np.max(data)/2])
            ax.set_ylim([0, np.max(data)/2])
            ax.set_zlim([np.max(data)/2, np.max(data)])

class RocketEnv(gym.Env):
    def __init__(self):
        super(RocketEnv, self).__init__()
        self.rocket = Rocket()
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.rocket.state_dims,), dtype=np.float32
        )

        # Action space
        low = np.array([-30] * 10 + [0] * 5 + [-30] * 6 + [0] * 3 + [0])
        high = np.array([30] * 10 + [self.rocket.max_thrust[0]] * 5 + [30] * 6 + [self.rocket.max_thrust[1]] * 3 + [3])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Gymnasium의 reset 메서드를 호출하여 seed 설정
        state = self.rocket.reset()
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        state, reward, done, info = self.rocket.step(action)
        # terminated: 에피소드 종료 여부
        # truncated: 시간 초과 등으로 에피소드가 중단되었는지 여부
        terminated = done
        truncated = self.rocket.step_id >= self.rocket.max_step
        return np.array(state, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
