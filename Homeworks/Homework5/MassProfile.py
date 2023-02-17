# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:37:37 2023

@author: Bennett
"""

import numpy as np # Import numpy
import astropy.units as u # Import units
import astropy.table as tbl # Import table
import matplotlib.pyplot as plt # Implort plotting
from astropy.constants import G # Import Gravitational Constant
from ReadFile import Read # Import function that reads files
from CenterOfMass import CenterOfMass # Import CenterOfMass object
from GalaxyMass import ComponentMass # Import ability to get component mass

class MassProfile:
# Class to define COM position and velocity properties of a given galaxy 
# and simulation snapshot

    def __init__(self, galaxy, snap):
        '''
        Class to calculate the mass and velocity profiles of galaxies.
        
        Parameters
        ----------
        galaxy : str
            The name of the galaxy
        snap : int
            The (0-indexed) number of the time snapshot
        '''
     
        # add a string of the filenumber to the value “000”
        ilbl = '000' + str(snap)
        # remove all but the last 3 digits
        ilbl = ilbl[-3:]
        self.filename = "%s_"%(galaxy) + ilbl + '.txt'
        # read data in the given file using Read
        self.time, self.total, self.data = Read(self.filename)                                                                                             

        # store the mass, positions, and name
        self.gname = galaxy
        self.m = self.data['m']
        self.x = self.data['x']*u.kpc
        self.y = self.data['y']*u.kpc
        self.z = self.data['z']*u.kpc
    
    def MassEnclosed(self, ptype, rs):
        '''
        Get the mass of some galaxy component contained within some radii

        Parameters
        ----------
        rs : array of Quantities
            The radii within which to get the total component masses

        Returns
        -------
        totMass : array of Quantities
            The enclosed total compotnent masses at the radii

        '''
        COM = CenterOfMass(self.filename, ptype) # Create CenterOfMass object
        COM_P = COM.COM_P(0.1) # Get center of mass coordinates
        massesEnclosed = np.zeros(len(rs)) # Generate dummy array to store results
        ptypeis = np.where(self.data['type'] == ptype)[0] # Get indices of particles of correct type
        masses = self.m[ptypeis] # Get masses of particles of correct type
        for ri in range(len(rs)): # For index of radii list
            mass = 0 # Initialize variable to track total mass
            dists = np.sqrt((self.x[ptypeis]-COM_P[0])**2+(self.y[ptypeis]-COM_P[1])**2+(self.z[ptypeis]-COM_P[2])**2) # Get distance of each particle from COM
            withinr = np.where(dists < rs[ri])[0] # Get indices of particles within radius
            for i in withinr: # Iterate through all particles within radius
                mass += masses[i] # Add mass of particle
            massesEnclosed[ri] = mass # Store mass at radius
        return massesEnclosed*1e10*u.Msun # Return masses in the correct units
    
    def MassEnclosedTotal(self, rs):
        '''
        Get the mass of the galaxy contained within some radii

        Parameters
        ----------
        rs : array of Quantities
            The radii within which to get the total masses

        Returns
        -------
        totMass : array of Quantities
            The total masses at the radii

        '''
        totMass = np.zeros(len(rs))*u.Msun # Generate dummy array to store results
        for ptype in range(1,4): # For each particle type
            if not ((ptype == 3) and (self.gname == "M33")): # Don't try to add nonexistent M31 buldge mass
                totMass += self.MassEnclosed(float(ptype), rs) # Add mass contribution from particle type
        return totMass
    
    def CircularVelocity(self, ptype, rs):
        '''
        Get the circular velocity at radii caused by a particle type.

        Parameters
        ----------
        ptype : int
            The particle type to consider (1=DM,2=Disk,3=Buldge)
        rs : array of Quantities
            The radii at which to calculate the circular velocities

        Returns
        -------
        circularVelocities: array of Quantities
            An array where the value of each index is the circular velocity at the corresponding index in rs due to ptype

        '''
        Ms = self.MassEnclosed(ptype, rs) # Get masses enclosed
        return np.round(np.sqrt(G*Ms/rs).to(u.km/u.s),2) # Get and return circular velocity due to component

    def CircularVelocityTotal(self, rs):
        '''
        Get the circular velocity at radii caused by the enclosed mass.

        Parameters
        ----------
        rs : array of Quantities
            The radii at which to calculate the circular velocities

        Returns
        -------
        circularVelocities: array of Quantities
            An array where the value of each index is the circular velocity at the corresponding index in rs

        '''
        Ms = self.MassEnclosedTotal(rs) # Get masses enclosed
        return np.round(np.sqrt(G*Ms/rs).to(u.km/u.s),2) # Get and return total circular velocites

def HernquistMass(r,a,mHalo):
    '''
    Get the enclosed Hernquist Mass at a radius.

    Parameters
    ----------
    r : Quantity
        The radius within which to get mass
    a : Quantity
        The scale factor assumed
    mHalo : Quantity
        The total mass of the Halo

    Returns
    -------
    hernquistMass : Quantity
        The Hernquist mass within the radius

    '''
    return np.round((mHalo*r**2/(a+r)**2).to(u.Msun),2) # Get and return Hernquist mass

def HernquistVCirc(r,a,mHalo):
    '''
    Get the circular velocity at a radius around a Hernquist mass

    Parameters
    ----------
    r : Quantity
        The radius within which to get mass
    a : Quantity
        The scale factor assumed
    mHalo : Quantity
        The total mass of the Halo

    Returns
    -------
    hernquistV : Quantity
        The circular velocity at the radius

    '''
    M = HernquistMass(r, a, mHalo) # Get Hernquist mass
    return np.round(np.sqrt(G*M/r).to(u.km/u.s),2) # Apply circular velocity equation

def mass_curves(galShorthand,timestamp,test_a):
    '''
    Plot mass curves

    Parameters
    ----------
    galShorthand : str
        The shorthand name of the galaxy
    timestamp : int
        The timestamp
    test_a : int
        The scale factor assumed in the Hernquist profile (in kpc)

    Returns
    -------
    None.

    '''
    if galShorthand == 'MW':
        galName = "Milky Way"
    elif galShorthand == 'M31':
        galName = "Andromeda"
    elif galShorthand == 'M33':
        galName = "Triangulum"
    rs = np.arange(0.1, 30.0, 0.5)*u.kpc # Create array of radii to calculate values at
    galaxy = MassProfile(galShorthand, timestamp) # Create object to track galactic properties
    DM_masses = galaxy.MassEnclosed(1, rs) # Get enclosed DM mass
    disk_masses = galaxy.MassEnclosed(2, rs) # Get enclosed disk mass
    if galShorthand != 'M33': # Don't plot buldge values for M33
        buldge_masses = galaxy.MassEnclosed(3, rs) # Get enclosed buldge mass
        plt.semilogy(rs,buldge_masses,color='maroon',linestyle='dotted',label="Buldge Stars") # Plot enclosed buldge masses
    else:
        buldge_masses = np.zeros(len(DM_masses))*u.Msun
    masses = DM_masses+disk_masses+buldge_masses # Calculate total masses at radii
    DM_mass = ComponentMass(galaxy.filename, 1.0)*1e12*u.Msun # Get total galactic DM mass
    hernquist_masses = np.zeros(len(rs))*u.Msun # Create dummy array for Hernquist Masses
    for ri in range(len(rs)): # Index through radii
        hernquist_masses[ri] = HernquistMass(rs[ri],test_a*u.kpc,DM_mass) # Get Hernquist Mass within radii
    plt.title(galName+' Mass Profile') # Add plot title
    plt.semilogy(rs,DM_masses,color='lime',linestyle='dashdot',label="Dark Matter") # Plot enclosed DM mass
    plt.semilogy(rs,disk_masses,color='teal',linestyle='dashed',label="Disk Stars") # Plot enclosed disk mass
    plt.semilogy(rs,masses,color='black',linestyle='solid',label="Total") # Plot velocity curve
    plt.semilogy(rs,hernquist_masses,color='khaki',linestyle='solid',label="Hernquist Best Fit, a="+str(test_a)) # Plot theoretical mass curve
    plt.legend(loc='lower right') # Add labels for curves
    plt.xlabel('Radius (kpc)') # Label x axis
    plt.ylabel('Mass (Solar Masses)') # Label y axis
    plt.savefig(galShorthand+'_Mass_Profile.png', dpi=400) # Save figure as PNG
    plt.clf() # Clear plot to overplot later

def velocity_curves(galShorthand,timestamp,test_a):
    '''
    Plot velocity curves

    Parameters
    ----------
    galShorthand : str
        The shorthand name of the galaxy
    timestamp : int
        The timestamp
    test_a : int
        The scale factor assumed in the Hernquist profile (in kpc)

    Returns
    -------
    None.

    '''
    if galShorthand == 'MW':
        galName = "Milky Way"
    elif galShorthand == 'M31':
        galName = "Andromeda"
    elif galShorthand == 'M33':
        galName = "Triangulum"
    rs = np.arange(0.1, 30.0, 0.5)*u.kpc # Create array of radii to calculate values at
    galaxy = MassProfile(galShorthand, timestamp) # Create object to track galactic properties
    DM_velocities = galaxy.CircularVelocity(1, rs) # Get DM-caused circular velocity
    disk_velocities = galaxy.CircularVelocity(2, rs) # Get disk-caused circular velocity
    if galShorthand != 'M33': # Don't plot buldge values for M33
        buldge_velocities = galaxy.CircularVelocity(3, rs) # Get buldge-caused velocity
        plt.semilogy(rs,buldge_velocities,color='maroon',linestyle='dotted',label="Buldge Stars") # Plot buldge-caused velocity curve
    total_velocities = galaxy.CircularVelocityTotal(rs) # Get circular velocity from galactic mass
    DM_mass = ComponentMass(galaxy.filename, 1.0)*1e12*u.Msun # Get total galactic DM mass
    hernquist_velocities = np.zeros(len(rs))*u.km/u.s # Create dummy array for Hernquist Circular Velocities
    for ri in range(len(rs)): # Index through radii
        hernquist_velocities[ri] = HernquistVCirc(rs[ri],test_a*u.kpc,DM_mass) # Get Hernquist Circular Velocity within radii
    plt.title(galName+' Circular Velocity Profile') # Add plot title
    plt.semilogy(rs,DM_velocities,color='lime',linestyle='dashdot',label="Dark Matter") # Plot DM-caused velocity curve
    plt.semilogy(rs,disk_velocities,color='teal',linestyle='dashed',label="Disk Stars") # Plot disk-caused velocity curve
    plt.semilogy(rs,total_velocities,color='black',linestyle='solid',label="Total") # Plot velocity curve
    plt.semilogy(rs,hernquist_velocities,color='khaki',linestyle='solid',label="Hernquist Best Fit, a="+str(test_a)) # Plot theoretical velocity curve
    plt.legend(loc='lower right') # Add labels for curves
    plt.xlabel('Radius (kpc)') # Label x axis
    plt.ylabel('Velocity (km/s)') # Label y axis
    plt.savefig(galShorthand+'_Circular_Velocity_Profile.png', dpi=400) # Save figure as PNG
    plt.clf() # Clear plot to overplot later

def main():
    mass_curves("MW",0,62) # Plot MW mass curves
    mass_curves("M31",0,60) # Plot M31 mass curves
    mass_curves("M33",0,26) # Plot M33 mass curves
    velocity_curves("MW",0,62) # Plot MW circular velocity curves
    velocity_curves("M31",0,60) # Plot M31 circular velocity curves
    velocity_curves("M33",0,26) # Plot M33 circular velocity curves

main()