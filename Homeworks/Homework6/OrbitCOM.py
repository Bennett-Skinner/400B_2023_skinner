

# Homework 6 Template
# G. Besla & R. Li




# import modules
import numpy as np
import astropy.units as u
from astropy.constants import G

# import plotting modules
import matplotlib.pyplot as plt
import matplotlib

# my modules
from ReadFile import Read
from CenterOfMass import CenterOfMass




def OrbitCOM(galaxy,start,end,n):
    """function that loops over all the desired snapshots to compute the COM pos and vel as a function of time.
    inputs:
        galaxy (string)
            The shortform name of the galaxy (MW, M31, or M33)
        start (int)
            The starting timestamp
        end (int)
            The ending timestamp
        n (int)
            The number of timestamps between returned COMs
    outputs: 
    """
    
    # compose the filename for output
    fileout = "Orbit_" + galaxy + '.txt'
    
    #  set tolerance and VolDec for calculating COM_P in CenterOfMass
    # for M33 that is stripped more, use different values for VolDec
    delta = 0.1
    if galaxy == 'M33':
        volDec = 4
    else:
        volDec = 2
    
    # generate the snapshot id sequence and check generated array is vaild
    snap_ids = np.arange(start,end,n)
    if len(snap_ids) == 0:
        raise Exception("Provided parameters define no valid timestamps")
    
    # initialize the array for orbital info: t, x, y, z, vx, vy, vz of COM
    orbit = np.zeros([len(snap_ids),7])
    
    # a for loop 
    for i,snap_id in enumerate(snap_ids): # loop over files
        
        # compose the data filename (be careful about the folder)
        filename = '000'+str(snap_id)
        filename = "VLowRes/" + "%s_"%(galaxy) + filename[-3:] + '.txt'
        
        # Initialize an instance of CenterOfMass class, using disk particles
        galaxy_COM = CenterOfMass(filename,2)

        # Store the COM pos and vel.
        COM_p = galaxy_COM.COM_P(delta,volDec)
        COM_v = galaxy_COM.COM_V(COM_p[0], COM_p[1], COM_p[2])
       
    
        # store the time, pos, vel in ith element of the orbit array,  without units (.value) 
        # note that you can store 
        # a[i] = var1, *tuple(array1)
        orbit[i,0] = galaxy_COM.time.value/1000
        orbit[i,1:4] = COM_p.value
        orbit[i,4:] = COM_v.value
        
        
        
        # print snap_id to see the progress
        print(snap_id)
        
    # write the data to a file
    # we do this because we don't want to have to repeat this process 
    # this code should only have to be called once per galaxy.
    np.savetxt(fileout, orbit, fmt = "%11.3f"*7, comments='#',
               header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"\
                      .format('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))




# Recover the orbits and generate the COM files for each galaxy
# read in 800 snapshots in intervals of n=5
OrbitCOM("MW", 0, 801, 5)
OrbitCOM("M31", 0, 801, 5)
OrbitCOM("M33", 0, 801, 5)

# Read in the data files for the orbits of each galaxy that you just created
# headers:  t, x, y, z, vx, vy, vz
# using np.genfromtxt
MW = np.genfromtxt("Orbit_MW.txt",dtype=None,names=True)
M31 = np.genfromtxt("Orbit_M31.txt",dtype=None,names=True)
M33 = np.genfromtxt("Orbit_M33.txt",dtype=None,names=True)



# function to compute the magnitude of the difference between two vectors 
# You can use this function to return both the relative position and relative velocity for two 
# galaxies over the entire orbit  
def get_relative_mag(vec1,vec2):
    if (len(vec1) != len(vec2)):
        raise Exception("Cannot add two vectors of different dimensionality")
    vec3 = np.subtract(vec1,vec2)
    return np.sqrt(np.sum(vec3**2))


# Determine the magnitude of the relative position and velocities 

# of MW and M31
if len(MW["t"]) != len(M31["t"]):
    raise Exception("Unequal amount of galaxy timestamps")
for it in range(len(M31["t"])):
    if MW["t"][it] != M31["t"][it]:
        raise Exception("Galaxy data is not at same timestamps")
MW_M31_pos = [get_relative_mag(np.array([MW["x"][i],MW["y"][i],MW["z"][i]]), np.array([M31["x"][i],M31["y"][i],M31["z"][i]])) for i in range(len(MW['x']))]
MW_M31_vel = [get_relative_mag(np.array([MW["vx"][i],MW["vy"][i],MW["vz"][i]]), np.array([M31["vx"][i],M31["vy"][i],M31["vz"][i]])) for i in range(len(MW['x']))]
# of M33 and M31
if len(M33["t"]) != len(M31["t"]):
    raise Exception("Unequal amount of galaxy timestamps")
for it in range(len(M31["t"])):
    if M33["t"][it] != M31["t"][it]:
        raise Exception("Galaxy data is not at same timestamps")
M33_M31_pos = [get_relative_mag(np.array([M31["x"][i],M31["y"][i],M31["z"][i]]), np.array([M33["x"][i],M33["y"][i],M33["z"][i]])) for i in range(len(M31['x']))]
M33_M31_vel = [get_relative_mag(np.array([M31["vx"][i],M31["vy"][i],M31["vz"][i]]), np.array([M33["vx"][i],M33["vy"][i],M33["vz"][i]])) for i in range(len(M33['x']))]



# Plot the Orbit of the galaxies 
#################################
plt.title("Milky Way-Andromeda Distance")
plt.plot(MW['t'],MW_M31_pos)
plt.xlabel("Time (Gyr)")
plt.ylabel("Distance (kpc)")
plt.savefig("MW-M31_distance.png",dpi=400)
plt.clf()
plt.title("Triangulum-Andromeda Distance")
plt.plot(M31['t'],M33_M31_pos)
plt.xlabel("Time (Gyr)")
plt.ylabel("Distance (kpc)")
plt.savefig("M33-M31_distance.png",dpi=400)
plt.clf()



# Plot the orbital velocities of the galaxies 
#################################
plt.title("Milky Way-Andromeda Relative Velocity")
plt.plot(MW['t'],MW_M31_vel)
plt.xlabel("Time (Gyr)")
plt.ylabel("Velocity (km/s)")
plt.savefig("MW-M31_relative_velocity.png",dpi=400)
plt.clf()
plt.title("Triangulum-Andromeda Relative Velocity")
plt.plot(M31['t'],M33_M31_vel)
plt.xlabel("Time (Gyr)")
plt.ylabel("Velocity (km/s)")
plt.savefig("M33-M31_relative_velocity.png",dpi=400)
plt.clf()