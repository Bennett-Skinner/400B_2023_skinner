
# # Homework 7
# 
# Bennett Skinner, Rixin Li & G. Besla
# 




# import necessary modules
# numpy provides powerful multi-dimensional arrays to hold and manipulate data
import numpy as np
# matplotlib provides powerful functions for plotting figures
import matplotlib.pyplot as plt
# astropy provides unit system and constants for astronomical calculations
import astropy.units as u
import astropy.constants as const
# import Latex module so we can display the results with symbols
from IPython.display import Latex

# **** import CenterOfMass to determine the COM pos/vel of M33
from CenterOfMass import CenterOfMass

# # M33AnalyticOrbit

class M33AnalyticOrbit:
    """ Calculate the analytical orbit of M33 around M31 """
    
    def __init__(self, filename): # **** add inputs
        """ **** ADD COMMENTS """

        ### get the gravitational constant (the value is 4.498502151575286e-06)
        self.G = const.G.to(u.kpc**3/u.Msun/u.Gyr**2).value
        
        ### **** store the output file name
        self.filename = filename
        
        ### get the current pos/vel of M33 
        # **** create an instance of the  CenterOfMass class for M33 
        M33_COM = CenterOfMass('M33_000.txt',2.0)
        # **** store the position VECTOR of the M33 COM (.value to get rid of units)
        M33_v = M33_COM.COM_P(0.1,4)
        # **** store the velocity VECTOR of the M33 COM (.value to get rid of units)
        M33_p = M33_COM.COM_V(M33_v[0],M33_v[1],M33_v[2])
        
        # **** create an instance of the  CenterOfMass class for M33 
        M31_COM = CenterOfMass('M31_000.txt',2.0)
        # **** store the position VECTOR of the M31 COM (.value to get rid of units)
        M31_v = M31_COM.COM_P(0.1,2)
        # **** store the velocity VECTOR of the M31 COM (.value to get rid of units)
        M31_p = M31_COM.COM_V(M31_v[0],M31_v[1],M31_v[2])
        
        M33_v, M33_p, M31_v, M31_p = M33_v.value, M33_p.value, M31_v.value, M31_p.value
        
        ### store the DIFFERENCE between the vectors posM33 - posM31
        # **** create two VECTORs self.r0 and self.v0 and have them be the
        # relative position and velocity VECTORS of M33
        self.r0 = M33_p - M31_p
        self.v0 = M33_v - M31_v
        
        ### get the mass of each component in M31 
        ### disk
        # **** self.rdisk = scale length (no units)
        self.rdisk = 5

        # **** self.Mdisk set with ComponentMass function. Remember to *1e12 to get the right units. Use the right ptype
        self.Mdisk = 0.12e12
        
        ### bulge
        # **** self.rbulge = set scale length (no units)
        self.rbuldge = 1
        
        # **** self.Mbulge  set with ComponentMass function. Remember to *1e12 to get the right units Use the right ptype
        self.Mbuldge= 0.019e12
        
        # Halo
        # **** self.rhalo = set scale length from HW5 (no units)
        self.rhalo = 60
        
        # **** self.Mhalo set with ComponentMass function. Remember to *1e12 to get the right units. Use the right ptype
        self.Mhalo = 1.921e12
    
    
    def HernquistAccel(self,M,r_a,r): # it is easiest if you take as an input the position VECTOR 
        """ **** ADD COMMENTS """
        
        ### **** Store the magnitude of the position vector
        rmag = np.sqrt(r[0]**2+r[1]**2+r[2]**2)
        
        ### *** Store the Acceleration
        Hern =  -self.G*M/(rmag*(r_a+rmag)**2)*r
        # NOTE: we want an acceleration VECTOR so you need to make sure that in the Hernquist equation you 
        # use  -G*M/(rmag *(ra + rmag)**2) * r --> where the last r is a VECTOR 
        
        return Hern
    
    
    
    def MiyamotoNagaiAccel(self, r):# it is easiest if you take as an input a position VECTOR  r 
        """ **** ADD COMMENTS """
        
        ### Acceleration **** follow the formula in the HW instructions
        R = np.sqrt(r[0]**2+r[1]**2)
        b = np.sqrt(r[2]**2+(self.rdisk/5.0)**2)
        B = b + self.rdisk
        accel = -self.G*self.Mdisk/(R**2+B**2)**1.5*r
        accel *= np.array([1,1,B/b])
        
        # AGAIN note that we want a VECTOR to be returned  (see Hernquist instructions)
        # this can be tricky given that the z component is different than in the x or y directions. 
        # we can deal with this by multiplying the whole thing by an extra array that accounts for the 
        # differences in the z direction:
        #  multiply the whle thing by :   np.array([1,1,ZSTUFF]) 
        # where ZSTUFF are the terms associated with the z direction
        
        
        return accel
        # the np.array allows for a different value for the z component of the acceleration
    
    def M31Accel(self, r): # input should include the position vector, r
        """ **** ADD COMMENTS """
        ### Call the previous functions for the halo, bulge and disk
        # **** these functions will take as inputs variable we defined in the initialization of the class like 
        # self.rdisk etc.
        haloAccel = self.HernquistAccel(self.Mhalo,self.rhalo,r)
        buldgeAccel = self.HernquistAccel(self.Mbuldge,self.rbuldge,r)
        diskAccel = self.MiyamotoNagaiAccel(r)
        
        # return the SUM of the output of the acceleration functions - this will return a VECTOR 
        return haloAccel+buldgeAccel+diskAccel
    
    def LeapFrog(self, r,v,dt): # take as input r and v, which are VECTORS. Assume it is ONE vector at a time
        """ **** ADD COMMENTS """
        
        # predict the position at the next half timestep
        rhalf = r+v*dt/2
        
        # predict the final velocity at the next timestep using the acceleration field at the rhalf position 
        vnew = v+self.M31Accel(rhalf)*dt
        
        # predict the final position using the average of the current velocity and the final velocity
        # this accounts for the fact that we don't know how the speed changes from the current timestep to the 
        # next, so we approximate it using the average expected speed over the time interval dt. 
        rnew = rhalf+vnew*dt/2
        
        return rnew, vnew
    
    def OrbitIntegration(self, t0, dt, tmax):
        """ **** ADD COMMENTS """
        # initialize the time to the input starting time
        t = t0
        
        # initialize an empty array of size :  rows int(tmax/dt)+2  , columns 7
        orbit = np.zeros((int(tmax/dt)+2,7))
        
        # initialize the first row of the orbit
        orbit[0] = t0, *tuple(self.r0), *tuple(self.v0)
        # this above is equivalent to 
        # orbit[0] = t0, self.r0[0], self.r0[1], self.r0[2], self.v0[0], self.v0[1], self.v0[2]
        
        
        # initialize a counter for the orbit.  
        i = 1 # since we already set the 0th values, we start the counter at 1
        
        # start the integration (advancing in time steps and computing LeapFrog at each step)
        while (t < tmax):  # as long as t has not exceeded the maximal time 
            t += dt
            orbit[i,0] = t
            rnew,vnew = self.LeapFrog(orbit[i-1,1:4],orbit[i-1,4:],dt)
            orbit[i,1:4] = rnew
            orbit[i,4:] = vnew
            i += 1
        
        np.savetxt(self.filename, orbit, fmt = "%11.3f"*7, comments='#',header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"\
                   .format('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))


# function to compute the magnitude of the difference between two vectors 
# You can use this function to return both the relative position and relative velocity for two 
# galaxies over the entire orbit  
def get_relative_mag(vec1,vec2):
    if (len(vec1) != len(vec2)):
        raise Exception("Cannot add two vectors of different dimensionality")
    vec3 = np.subtract(vec1,vec2)
    return np.sqrt(np.sum(vec3**2))

if __name__ == '__main__' :
    M33Orb = M33AnalyticOrbit("M33_analytic_orbit.txt")
    M33Orb.OrbitIntegration(0,0.1,12)
    M33_M31_3dpos = np.loadtxt(M33Orb.filename)
    #       print([np.sqrt(np.sum(M33_M31_3dpos[timestep,1:4]**2)) for timestep in range(len(M33_M31_3dpos))])
    M33_M31_pos = [np.sqrt(np.sum(M33_M31_3dpos[timestep,1:4]**2)) for timestep in range(len(M33_M31_3dpos))]
    M33_M31_vel = [np.sqrt(np.sum(M33_M31_3dpos[timestep,5:]**2)) for timestep in range(len(M33_M31_3dpos))]
    M31 = np.genfromtxt("../Homework6/Orbit_M31.txt",dtype=None,names=True)
    M33 = np.genfromtxt("../Homework6/Orbit_M33.txt",dtype=None,names=True)
    M33_M31_numpos = [get_relative_mag(np.array([M31["x"][i],M31["y"][i],M31["z"][i]]), np.array([M33["x"][i],M33["y"][i],M33["z"][i]])) for i in range(len(M33))]
    M33_M31_numvel = [get_relative_mag(np.array([M31["vx"][i],M31["vy"][i],M31["vz"][i]]), np.array([M33["vx"][i],M33["vy"][i],M33["vz"][i]])) for i in range(len(M33))]
    plt.title("Triangulum-Andromeda Distance")
    plt.plot(M33_M31_3dpos[:,0],M33_M31_pos,label="Analytical")
    plt.plot(M33["t"],M33_M31_numpos,label="Numerical") 
    plt.xlabel("Time (Gyr)")
    plt.ylabel("Distance (kpc)")
    plt.legend(loc="upper right")
    plt.savefig("M33-M31-analytical_distance.png",dpi=400)
    plt.clf()
    plt.title("Triangulum-Andromeda Mutual Velocity")
    plt.plot(M33_M31_3dpos[:,0],M33_M31_vel,label="Analytical")
    plt.plot(M33["t"],M33_M31_numvel,label="Numerical")
    plt.xlabel("Time (Gyr)")
    plt.ylabel("Velocity (km/s)")
    plt.legend(loc="upper right")
    plt.savefig("M33-M31-analytical_velocity.png",dpi=400)
