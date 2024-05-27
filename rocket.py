import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from ambiance import Atmosphere


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
    def __init__(self, max_steps=400000, task='launching', rocket_type='falcon-4',
                 viewport_h=768, path_to_bg_img=None, fault=False):

        self.task = task

        #기본적인 환경 configuration
        self.G = 6.6742*10**-11;            #%%Gravitational constant (SI Unit)
        self.g = 9.8                        #지표에서의 중력가속도
        self.dt = 0.05                      # step의 시간간격
        self.max_step= max_steps
        self.R_planet = 6371000                # 지구반지름 TODO: 단위가 m임
        self.M_planet = 5.972*(10**(24))
        self.max_thrust = [1521000, 327000, 0]       # 최대 추력, 1단과 2단이 다름, 모든 엔진값의 합 -> 나중에 엔진당 추력으로 변환 필요

        #rocket configuration
        self.target_p =  800000
        self.rocket_type = rocket_type
        self.D = 3.7                                 # rocket diameter (meters)
        self.H = [70,27.4,13.1]                          # rocket height (meters) 

        self.mass=[553000, 119800, 8300]                # rocket의 stage별 질량 [초기질량, 1단 분리 이후 질량, 2단 분리 이후 인공위성 질량]
        self.fuel_mass=[411000, 107500, 0]               # stage별 가용 연료 질량 TODO: 1단 분리 이후 질량과 연료값? / 연소시간기준 1단 162초, 2단 397초
        self.engine_cutoff_fuelmass = [36000,0,-9999]
        
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

        self.state_dims = len(Rocket.flatten(self.state))
        self.action_dims = 24
        

    def reset(self, state_list=None):

        if state_list is None:
            self.state = self.create_initial_state()
        else:
            self.state = state_list
        self.state_buffer = []
        self.step_id = 0
        self.already_crash = False
        cv2.destroyAllWindows()
        return Rocket.flatten(self.state)


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

    def create_initial_state(self):
        # predefined locations
        position = np.array([0.0, 0.0, self.R_planet])
        velocity = np.array([0.0, 0.0, 0.0])
        angle = np.array([0.0, 0.0, 0.0])
        angular_v= np.array([0.0, 0.0, 0.0])
        state = [
            position, velocity, angle, angular_v, self.fuel_mass[0],0, np.zeros((8, 2)), 0
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

        # 엔진 분리상황 추가
        output_list.append(input_list[7])

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
        angle_thrusts = np.radians(self.state[6][NumofUsedEngines:NumofUsedEngines+current_NumofEngines])           # state에는 도 로 표시됨
        
        total_thrust = np.array([0.0,0.0,0.0])
        total_torque = np.array([0.0,0.0,0.0])
        mdot = 0
        if (self.state[4] > self.engine_cutoff_fuelmass[stage]) and (self.state[7] == 0):                           # 연료가 충분하고 분리작업이 진행 중이지 않은 경우
            if stage != 2: # final stage가 아닌 경우
                for i in range(len(thrusts)):
                    x_thrust = -min(thrusts[i],self.max_thrust[stage])*np.sin(angle_thrusts[i][0])*np.cos(angle_thrusts[i][1])
                    y_thrust = -min(thrusts[i],self.max_thrust[stage])*np.sin(angle_thrusts[i][0])*np.sin(angle_thrusts[i][1])
                    z_thrust = min(thrusts[i],self.max_thrust[stage])*np.cos(angle_thrusts[i][0])
                    thrust= np.array([x_thrust,y_thrust,z_thrust])     # 엔진별 thrust vector 생성
                
                    total_thrust +=thrust                            # thrust vector의 합                            
                    total_torque +=np.cross(np.array(self.d_CM_e[stage])+np.array(self.r_engines[stage][i]),thrust)

                mdot -= sum([min(f,self.max_thrust[stage]) for f in thrust]) / (self.g * self.Isp[self.state[5]])   # thrust에 따른 연료소모속도
        
        angular_velocity0 = list(action[0:self.num_engines[0]]) + list(action[self.num_engines[0]*3:self.num_engines[0]*3+self.num_engines[1]])
        angular_velocity1 = list(action[self.num_engines[0]:self.num_engines[0]*2]) + list(action[self.num_engines[0]*3+self.num_engines[1]:self.num_engines[0]*3+self.num_engines[1]*2])
        angular_velocity_pair = np.array(list(zip(angular_velocity0, angular_velocity1)))

        return total_torque, total_thrust, mdot, angular_velocity_pair
    
    def get_New_state(self, state, acc_Earth, angular_acc, mdot, d_angVofEngines):
        #input : [position, velocity, rotational angle, angular velocity, feul 질량, current stage, engine angle], acc, angular_acc, engine angular V array
        #output : derivatives of position, velocity, rotational angle, angular velocity, feul 질량, current stage, engine angle
        
        new_position = np.add(state[0], state[1]*self.dt)

        new_velocity = np.add(state[1], acc_Earth*self.dt)
        if np.isnan(new_velocity).any():
            print(new_velocity)

        new_rot_angle = np.add(state[2], state[3]*self.dt)

        angular_acc_Earth = transform_coordinates(angular_acc,state[2][0],state[2][1],state[2][2])
        new_rot_angV = np.add(state[3], angular_acc_Earth*self.dt)

        new_fuel = max(state[4] + mdot*self.dt,0)               # 새 연료량 계산
        detach_time = state[7]
        if new_fuel <= self.engine_cutoff_fuelmass[state[5]]:   # 새 연료량이 분리 연료량보다 작거나 같은 경우(분리가 일어나는 경우)
            if state[5] == 2:                                       # 마지막 stage임에도 분리신호가 들어오는 경우
                print('current stage is last stage. The rocket can not be detached.')
            else:                                                   # 분리 시작                   
                new_stage = state[5] + 1
                new_fuel = self.fuel_mass[new_stage]
                self.current_mass = self.mass[new_stage]
                detach_time = self.dt
                self.I = [0.25 * self.current_mass * (self.D ** 2) + self.current_mass * (self.H[new_stage] ** 2) / 12, 
                  0.25 * self.current_mass * (self.D ** 2) + self.current_mass * (self.H[new_stage] ** 2) / 12,
                  0.5 * self.current_mass * (self.D ** 2)]
        else:                                                   # 분리가 일어나지 않는 경우
            if detach_time > 0:
                if detach_time > 2:                             # 분리에 2초가 걸린다고 가정
                    detach_time = 0
                else:
                    detach_time += self.dt
            new_stage = state[5]
            self.current_mass += new_fuel - state[4]

        new_engine_angle = np.add(state[6], d_angVofEngines*self.dt)
        # theta0 값을 -30에서 30 사이로 제한
        new_engine_angle[:, 0] = np.clip(new_engine_angle[:, 0], -30, 30)

        # theta1 값을 0에서 360 사이로 맞추기 (0보다 작은 값 조정 포함)
        new_engine_angle[:, 1] = new_engine_angle[:, 1] % 360

        return [new_position, new_velocity, new_rot_angle, new_rot_angV, new_fuel, new_stage, new_engine_angle, detach_time] 
    
    def check_crash(self):
        crash = False
        x, y, z = self.state[0]
        vx,vy,vz = self.state[1]
        r = np.sqrt(x**2 + y**2 + z**2)
        altitude = r-self.R_planet
        if (altitude < 0):
            crash = True
        if vx==0 and vy==0 and vz==0:
            crash = True
        if (altitude > self.target_p+1000):
            crash = True
        return crash

    def calculate_reward(self, state):
        # 위치 및 속도
        position = state[0]
        velocity = state[1]
        orientation = state[2]  # 각도 (roll, pitch, yaw)
        angular_velocity = state[3]  # 각속도 (wx, wy, wz)
        fuel_mass = state[4] # 연료 질량

        # 현재 고도
        altitude = np.sqrt(position[0]**2 + position[1]**2 + position[2]**2) - self.R_planet

        # 목표 고도 (원궤도)
        target_altitude = self.target_p
        target_radius = self.R_planet + target_altitude

        # 고도 기반 보상
        dist_to_target_altitude = abs(target_altitude - altitude)
        altitude_reward = np.exp(-dist_to_target_altitude / 1000)

        # 속도 보상: 목표 궤도에서 필요한 궤도 속도와 현재 속도의 차이를 고려
        orbital_speed = np.sqrt(self.G * self.M_planet / target_radius)
        speed_difference = abs(np.linalg.norm(velocity) - orbital_speed)
        speed_reward = np.exp(-speed_difference / 100)

        # y축 방향 회전 각도(pitch) 페널티
        pitch_angle = orientation[1]  # pitch 각도
        pitch_penalty = np.exp(-abs(pitch_angle) / 10)  # pitch 각도 기반 페널티

        # 각속도 x축 성분 보상/페널티
        angular_velocity_x = angular_velocity[0]
        if angular_velocity_x >= 0:
            angular_velocity_reward = 1  # 양수일 때 보상
        else:
            angular_velocity_reward = np.exp(-abs(angular_velocity_x) / 10)  # 음수일 때 페널티
        
        # 발사 초기에 각도가 치우치면 (3도이상 치우치면 penalty)
        if (altitude <15) and (np.any(state[6][:,0] > 3)):
            engine_penalty = -10

        # 총 보상 계산
        reward = altitude_reward + speed_reward + pitch_penalty + angular_velocity_reward + engine_penalty

        # 목표 고도에 가까워졌을 때 추가 보상
        if dist_to_target_altitude < 100:
            reward += 10

        return reward

    def step(self, action):

        torque , thrust, mdot, d_ang_VofEngines = self.get_propulsion(action)
        g_acc,r = self.get_gravity()
    
        aeroF = self.get_aerofriction(r)
        
        vdot = g_acc + (aeroF + transform_coordinates(thrust, self.state[2][0], self.state[2][1], self.state[2][2]))/self.current_mass
        if np.isnan(vdot).any():
            print(vdot)
        wdot = torque /self.I

        new_state = self.get_New_state(self.state, vdot, wdot,mdot, d_ang_VofEngines)
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
        z = self.R_planet * np.outer(np.ones(np.size(u)), np.cos(v))

        # 지구 표면 플롯
        ax.plot_wireframe(x, y, z, color='g', alpha=0.3)

        # 데이터 초기화 및 skip_steps 적용
        data = np.array([state[0] for state in self.state_buffer[::skip_steps]]).T
        line, = ax.plot([], [], [], 'r-', label="Trajectory")  # 빈 궤적 초기화
        marker, = ax.plot([], [], [], 'bo', markersize=5, label="Rocket")  # 로켓 마커 초기화

        # 축 라벨 설정
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
            
            # 현재 로켓 위치 기준으로 축 범위 설정
            current_pos = data[:, num]
            margin = 100000  # 여유 범위 설정
            
            # 축 범위 설정 - 0, 0, 0이 계속 나타나도록 하고 1:1:1 비율 유지
            ax.set_xlim([min(0, current_pos[0] - margin), max(0, current_pos[0] + margin)])
            ax.set_ylim([min(0, current_pos[1] - margin), max(0, current_pos[1] + margin)])
            ax.set_zlim([min(0, current_pos[2] - margin), max(0, current_pos[2] + margin)])
            
            # 모든 축의 비율을 동일하게 설정
            max_range = max(current_pos[0] + margin, current_pos[1] + margin, current_pos[2] + margin)
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])