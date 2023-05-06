'''
We want to see if M33 is becoming more like an elliptical galaxy with time.

So we see how the ratio of the x and y kinetic energies in the galaxy to the z kinetic energy evolves with time.
If the ratios trend to 1, M33 is becoming more symmetric in kinetic energy and thus dispersion-supported.
So M33 is becoming more elliptical.

We also want to see how spread the radial velocities of disk stars in the galaxy vary from the prediction given
by circular orbits, indicating dispersion. We quantify this by the ratio of the average gravitationally-induced circularvelocity for stars in the galaxy to the average dispersion (not related with a gravitationally-induced circular velocity) velocity for stars in the galaxy.
If this ratio is decreasing, stars' gravitationally-induced circular velocities are becoming less important relative to their dispersion velocities, so M33 is becoming more dispersion-supported.
So M33 is becoming more elliptical.
'''

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from CenterOfMass import CenterOfMass
from MassProfile import MassProfile

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
starttime, endtime, timestep = 0,801,1
timestamps = np.arange(starttime,endtime,timestep)
num_timesteps = len(timestamps)
KEs, nKEs = np.zeros((num_timesteps,3)), np.zeros((num_timesteps,2)) # Initialize matricies to track KE and normalized x/y KE with time
avg_vcirc_over_vdisps = np.zeros(num_timesteps)
for timestamp in timestamps:
        print("Timestamp", timestamp)
        filestamp = '000'+str(timestamp)
        filestamp = filestamp[-3:]
        filename = 'M33/M33_' + filestamp + '.txt'
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
        allowedi = np.where(rtot<10) # Restrict ourselves to disk stars within 10 kpc of center
        COMD.data = COMD.data[:][allowedi] # Re-select objects within 10 kpc of center
        # Now re-center only with the objects within 10 kpc of center
        # Compute COM of M33 using disk particles
        COMP = COMD.COM_P(0.1,4)
        COMV = COMD.COM_V(COMP[0],COMP[1],COMP[2])
        # Determine positions of disk particles relative to COM
        xD = COMD.x - COMP[0].value
        yD = COMD.y - COMP[1].value
        zD = COMD.z - COMP[2].value
        # total magnitude
        rtot = np.sqrt(xD**2 + yD**2 + zD**2)
        allowedi = np.where(rtot<10) # Now re-select only particles still within 10 kpc of center
        xD,yD,zD,rtot=xD[allowedi],yD[allowedi],zD[allowedi],rtot[allowedi]
        COMD.vx,COMD.vy,COMD.vz=COMD.vx[allowedi],COMD.vy[allowedi],COMD.vz[allowedi]
        # Determine velocities of disk particles relative to COM motion
        vxD = COMD.vx - COMV[0].value
        vyD = COMD.vy - COMV[1].value
        vzD = COMD.vz - COMV[2].value
        # total velocity
        vtot = np.sqrt(vxD**2 + vyD**2 + vzD**2)
        # Vectors for r and v
        r = np.array([xD,yD,zD]).T # transposed
        v = np.array([vxD,vyD,vzD]).T
        rn, vn = RotateFrame(r,v)
        # Get relative KEs
        KEx, KEy, KEz = np.sum(vn[:,0]**2),np.sum(vn[:,1]**2),np.sum(vn[:,2]**2) # Get 2*the KE in each direction
        KEs[(timestamp-starttime)//timestep,:] = KEx,KEy,KEz # Save 2*the KE
        nKEs[(timestamp-starttime)//timestep,:] = KEx/KEz,KEy/KEz # Save the ratio of rotational KEs to vertical KE
        # Save reference figures of the spatial geometry of the galaxy at the timestamp
        # Face-on
        plt.hist2d(rn[:,0],rn[:,1],bins=500,norm=LogNorm(),cmap='cool')
        plt.title("Face-on configuration of M33 at t="+str(timestamp*0.0142857)+" Gyr")
        plt.xlabel("x (kpc)")
        plt.ylabel("y (kpc)")
        plt.savefig("Images/FaceOn/M33_faceon_image_"+filestamp+".png",dpi=400)
        plt.clf()
        # Edge-on
        plt.hist2d(rn[:,0],rn[:,2],bins=500,norm=LogNorm(),cmap='cool')
        plt.title("Edge-on configuration of M33 at t="+str(timestamp*0.0142857)+" Gyr")
        plt.xlabel("x (kpc)")
        plt.ylabel("z (kpc)")
        plt.savefig("Images/EdgeOn/M33_edgeon_image_"+filestamp+".png",dpi=400)
        plt.clf()
        # Plot phase diagram at this timestamp
        R = np.arange(0,10.01,0.01)
        plt.hist2d(rn[:,0],vn[:,1],bins=500,norm=LogNorm()) # Observed data
        plt.colorbar()
        M33 = MassProfile('M33',filename)
        Vcirc = M33.circularVelocityTotal(R) # Get circular velocities for positive Rs
        plt.plot(np.concatenate((-R,R)),np.concatenate((-Vcirc.value,Vcirc.value)),color='red') # Theoretical data
        plt.xlabel('xy-projected radius (kpc)')
        plt.ylabel('Perpendicular Radial Velocity (km/s)')
        if timestamp == 742: # There is a data anomaly at this time, throw out its dispersion v rotational velocity data
            avg_vcirc_over_vdisps[(timestamp-starttime)//timestep] = avg_vcirc_over_vdisps[(timestamp-starttime-timestep)//timestep]
            continue
        centerdists = np.abs(rn[:,0]) # Get the radial of each particle from the center
        vcircs = M33.circularVelocityTotal(centerdists).value # Get theoretical rotational velocity for each particle
        # Get the signed circular velocity of each particle in galaxy
        for vi in range(len(vcircs)):
            if rn[vi,0] < 0:
                vcircs[vi] = -vcircs[vi]
        vdisps = np.abs(vn[:,1]-vcircs) # Dispersion velocity is simply non-rotational velocity
        avg_vcirc_over_vdisps[(timestamp-starttime)//timestep] = np.mean(np.abs(vcircs))/np.mean(vdisps) # Save v/sigma
        plt.savefig("position_v_radial_velocity/M33_radial_velocity_v_position_"+filestamp+".png",dpi=400)
        plt.clf()
timestep_conversion_factor = 0.0142857 # Number of Gyr between two adjacent timestamps
times = timestamps*timestep_conversion_factor # Convert times to Gyr
# Save figure that uses Kinetic Energy Method
plt.legend("Symmetry of Kinetic Energy in M33 with time")
plt.xlabel('Time (Gyr)')
plt.ylabel('Kinetic Energy of Component Over Z Kinetic Energy (Unitless)')
plt.plot(times,nKEs[:,0],linestyle='-',label='Normalized Kinetic Energy in x')
plt.plot(times,nKEs[:,1],linestyle='--',label='Normalized Kinetic Energy in y')
plt.legend(loc='upper left')
plt.savefig("Normalized_M33_KEs.png",dpi=400)
plt.clf()
# Save figure that uses Support Method
plt.legend("Rotational vs. Dispersion Support in M33 with time")
plt.xlabel('Time (Gyr)')
plt.ylabel('Average rotational velocity over average dispersion velocity (Unitless)')
plt.plot(times,avg_vcirc_over_vdisps,linestyle='-')
plt.savefig("rot_v_disp_over_time.png",dpi=400)
