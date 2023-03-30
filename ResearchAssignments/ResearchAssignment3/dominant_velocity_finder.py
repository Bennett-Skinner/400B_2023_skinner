'''
We want to see if M33 is becoming more like an elliptical galaxy with time.
So we see how the ratio of the x and y kinetic energies in the galaxy to the z kinetic energy evolves with time.
If the ratios trend to 1, M33 is becoming more symmetric in kinetic energy and thus dispersion-supported.
So M33 is becoming more elliptical.
'''

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from CenterOfMass import CenterOfMass

def RotateFrame(posI,velI):
    """a function that will rotate the position and velocity vectors
    so that the disk angular momentum is aligned with z axis. 
    
    PARAMETERS
    ----------
        posI : `array of floats`
             3D array of positions (x,y,z)
        velI : `array of floats`
             3D array of velocities (vx,vy,vz)
             
    RETURNS
    -------
        pos: `array of floats`
            rotated 3D array of positions (x,y,z) such that disk is in the XY plane
        vel: `array of floats`
            rotated 3D array of velocities (vx,vy,vz) such that disk angular momentum vector
            is in the +z direction 
    """
    
    # compute the angular momentum
    L = np.sum(np.cross(posI,velI), axis=0)
    # normalize the vector
    L_norm = L/np.sqrt(np.sum(L**2))


    # Set up rotation matrix to map L_norm to z unit vector (disk in xy-plane)
    
    # z unit vector
    z_norm = np.array([0, 0, 1])
    
    # cross product between L and z
    vv = np.cross(L_norm, z_norm)
    s = np.sqrt(np.sum(vv**2))
    
    # dot product between L and z 
    c = np.dot(L_norm, z_norm)
    
    # rotation matrix
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    v_x = np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]])
    R = I + v_x + np.dot(v_x, v_x)*(1 - c)/s**2

    # Rotate coordinate system
    pos = np.dot(R, posI.T).T
    vel = np.dot(R, velI.T).T
    
    return pos, vel

# Each file is 0.01429 Myr apart, so want to go to one out of every 7 times for 0.1 Gyr steps
starttime, endtime, timestep = 0,801,7
timestamps = np.arange(starttime,endtime,timestep)
num_timesteps = (endtime-starttime)//timestep
KEs, nKEs = np.zeros((num_timesteps+1,3)), np.zeros((num_timesteps+1,2)) # Initialize matricies to track KE and normalized x/y KE with time
for timestamp in timestamps:
        filename = '000'+str(timestamp)
        filename = 'M33/M33_' + filename[-3:] + '.txt'
        # Create a COM of object for M33 Disk Using Code from Assignment 4
        COMD = CenterOfMass(filename,2)
        # Compute COM of M33 using disk particles
        COMP = COMD.COM_P(0.1,4)
        COMV = COMD.COM_V(COMP[0],COMP[1],COMP[2])
        # Determine positions of disk particles relative to COM
        xD = COMD.x - COMP[0].value
        yD = COMD.y - COMP[1].value
        zD = COMD.z - COMP[2].value
        # total magnitude
        rtot = np.sqrt(xD**2 + yD**2 + zD**2)
        # Determine velocities of disk particles relatiev to COM motion
        vxD = COMD.vx - COMV[0].value
        vyD = COMD.vy - COMV[1].value
        vzD = COMD.vz - COMV[2].value
        # total velocity
        vtot = np.sqrt(vxD**2 + vyD**2 + vzD**2)
        # Vectors for r and v
        r = np.array([xD,yD,zD]).T # transposed
        v = np.array([vxD,vyD,vzD]).T
        rn, vn = RotateFrame(r,v)
        KEx, KEy, KEz = np.sum(vn[0]**2),np.sum(vn[1]**2),np.sum(vn[2]**2)
        KEs[timestamp//timestep,:] = KEx,KEy,KEz
        nKEs[timestamp//timestep,:] = KEx/KEz,KEy/KEz
times = timestamps*0.0142857
plt.legend("Symmetry of Kinetic Energy in M33 with time")
plt.xlabel('Time (Gyr)')
plt.ylabel('Kinetic Energy of Component Over Z Kinetic Energy (Unitless)')
plt.plot(times,nKEs[:,0],linestyle='-',label='Normalized Kinetic Energy in x')
plt.plot(times,nKEs[:,1],linestyle='--',label='Normalized Kinetic Energy in y')
plt.legend(loc='upper left')
plt.savefig("Normalized_M33_KEs.png",dpi=400) # Problem -- This plot does not show the expected trend!
