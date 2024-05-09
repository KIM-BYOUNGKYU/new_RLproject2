import numpy as np


class Aerodynamics():
    def __init__(self,name):
        self.name = name
        if name == 'Kerbin':
            ###import the aero model for interpolation
            data = np.loadtxt('kerbin_aerodynamics.txt')
            #print(data)
            self.altitude = data[:,0]
            #print(self.altitude)
            self.density = data[:,3]
            self.rhos = self.density[0]
            self.beta = 0.0
        elif name == 'Earth':
            ##going to use the Earth aero model
            self.beta = 0.1354/1000.0 ##density constant
            self.rhos = 1.225 #kg/m^3
    
    def getDensity(self,altitude):
        if self.name == 'Kerbin':
            ###interpolate
            rho = np.interp(altitude,self.altitude,self.density)
        elif self.name == 'Earth':
            ###Use special equation
            rho = self.rhos*np.exp(-self.beta*altitude)
        return rho