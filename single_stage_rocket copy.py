#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 21:56:24 2021

@author: carlos
"""

####Import all the modules we need
import numpy as np ###numeric python
import matplotlib.pyplot as plt ###matlab style plotting
import scipy.integrate as sci ##integration toolbox

plt.close("all")

##DEFINE SOME CONSTANT PARAMETERS
G = 6.6742*10**-11; #%%Gravitational constant (SI Unit)

###PLANET
###EARTH
#Rplanet = 6357000.0 #meters
#mplanet = 5.972e24 #kg
###KERBIN
Rplanet = 600000 #meters
mplanet = 5.2915158*10**22 #

##PARAMETERS OF ROCKET
###Initial Conditions for single stage rocket
x0 = Rplanet 
z0 = 0.0
velz0 = 0.0
velx0 = 0.0
period = 500.0
weighttons = 5.3
mass0 = weighttons*2000/2.2 #kg
max_thrust = 167970.0
Isp = 250.0 #seconds
tMECO = 38.0 #main engine cutoff time
tsep1 = 2.0 #length of time to remove 1st stage
mass1tons = 1.0
mass1 = mass1tons*2000/2.2

##Gravitational Acceleration Model
def gravity(x,z):
    global Rplanet,mplanet
    
    r = np.sqrt(x**2 + z**2)
    
    if r < Rplanet:
        accelx = 0.0
        accelz = 0.0
    else:
        accelx = G*mplanet/(r**3)*x
        accelz = G*mplanet/(r**3)*z
        
    return np.asarray([accelx,accelz])

def propulsion(t):   #=> 시간에 따른 함수가 아닌 state에 따른 함수로 변경해서 policy로 나타내보는 것
    global max_thrust,Isp,tMECO,ve
    ##Timing for thrusters
    if t < tMECO:
        #We are firing the main thruster
        thrustF = max_thrust
        mdot = -thrustF/ve
    if t > tMECO and t < (tMECO + tsep1):
        thrustF = 0.0
        ## masslost = mass1
        mdot = -mass1/tsep1
    if t > (tMECO + tsep1):
        thrustF = 0.0
        mdot = 0.0
    
    ##Angle of my thrust
    theta = 10*np.pi/180.0
    thrustx = thrustF*np.cos(theta)
    thrustz = thrustF*np.sin(theta)
      
    
    return np.asarray([thrustx,thrustz]),mdot
    
###Equations of Motion
###F = m*a = m*zddot
## z is the altitude from the center of the planet along the north pole
### x  is the altitude from center along equator through Africa 
## this is in meter
## zdot is the velocity along z
## zddot is the acceleration along z
###Second Order Differential Equation
def Derivatives(state,t):
    #state vector
    x = state[0]
    z = state[1]
    velx = state[2]
    velz = state[3]
    mass = state[4]
    
    #Compute zdot - Kinematic Relationship
    zdot = velz
    xdot = velx
    
    ###Compute the Total Forces
    ###GRavity
    gravityF = -gravity(x,z)*mass
    
    ###Aerodynamics
    aeroF = np.asarray([0.0,0.0])
    
    ###Thrust
    thrustF,mdot = propulsion(t)
    
    Forces = gravityF + aeroF + thrustF
    
    #Compute Acceleration
    if mass > 0:
        ddot = Forces/mass
    else:
        ddot = 0.0
        mdot = 0.0
    
    #Compute the statedot
    statedot = np.asarray([xdot,zdot,ddot[0],ddot[1],mdot])
    
    return statedot


#이 예시는 ODE를 직접 풀어서 방정식을 구하는 형식, 우리는 RL에서 요구하는 
#하나 하나의 step을 정의할 필요가 있음.
#time_step은 delta t로 정의
def take_step(state, t, time_step):
    statedot = Derivatives(state,t)
    new_state = state + statedot*time_step
    return new_state, t+time_step


###########EVERYTHING BELOW HERE IS THE MAIN SCRIPT###

###Test Surface Gravity
print('Surface Gravity (m/s^2) = ',gravity(0,Rplanet))

###Initial Conditions -- For orbit
"""
x0 = Rplanet+600000 ##m
z0 = 0.0
r0 = np.sqrt(x0**2+z0**2)
velz0 = np.sqrt(G*mplanet/r0)*1.1
velx0 = 0.0
period = 2*np.pi/np.sqrt(G*mplanet)*r0**(3.0/2.0)*1.5
"""

##Compute Exit Velocity
ve = Isp*9.81 #m/s
##Populate Initial Condition Vector
stateinitial = np.asarray([x0,z0,velx0,velz0,mass0])

##Time window
tout = np.linspace(0,period,1000)


###Numerical Integration Call
#stateout = sci.odeint(Derivatives,stateinitial,tout)
#stateout 계산
time_step = tout[1]-tout[0] #second
stateout = np.empty((0,5))
for t in tout:
    if t == 0:
        current_state = stateinitial
    next_state = take_step(current_state, t, time_step)[0]
    stateout = np.vstack((stateout, current_state))
    current_state = next_state


###REname variables
xout = stateout[:,0]
zout = stateout[:,1]
altitude = np.sqrt(xout**2+zout**2) - Rplanet
velxout = stateout[:,2]
velzout = stateout[:,3]
velout = np.sqrt(velxout**2 + velzout**2)
massout = stateout[:,4]

###Plot

###ALTITUDE
plt.plot(tout,altitude)
plt.xlabel('Time (sec)')
plt.ylabel('Altitude (m)')
plt.grid()

###VELOCITY
plt.figure()
plt.plot(tout,velout)
plt.xlabel('Time (sec)')
plt.ylabel('Total Speed (m/s)')
plt.grid()

###Mass
plt.figure()
plt.plot(tout,massout)
plt.xlabel('Time (sec)')
plt.ylabel('Mass (kg)')
plt.grid()

##2D Orbit
plt.figure()
plt.plot(xout,zout,'r-',label='Orbit')
plt.plot(xout[0],zout[0],'g*')
theta = np.linspace(0,2*np.pi,1000)
xplanet = Rplanet*np.sin(theta)
yplanet = Rplanet*np.cos(theta)
plt.plot(xplanet,yplanet,'b-',label='Planet')
plt.grid()
plt.legend()


plt.show()