# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 11:08:15 2023

@author: Bennett
"""

import numpy as np # Import package that contains efficient data storage structures
from ReadFile import Read # Import function that reads files

def ComponentMass(filename,particle_type):
    '''
    Gets the mass of a component of a galaxy.
    @param filename (str), the name of the file storing the data for the galaxy at some time
    @param particle_type (float), the type of particle the mass is desired for (1.0=Dark Matter, 2.0=Disk Stars, 3.0=Buldge Stars)
    @returns total_mass (float), the mass of the component of the galaxy, in 1e12 solar masses
    '''
    time, num_particles, data = Read(filename) # Read in properties of the galaxy
    total_mass = 0.0 # Initialize a variable to track the mass of a component of the galaxy
    indicies = np.where(data['type'] == particle_type)[0] # Get a list of the indicies of particles w/ a given type
    for index in indicies: # Iterate through every particle in the galaxy of a given type
        total_mass += data['m'][index]/1e2 # Add the mass of the particle (given in 1e10 solar masses) to the mass of the galaxy component
    return np.around(total_mass, 3) # Round and return the total mass

def main():
    '''
    Answers Question 3.
    '''
    table = np.zeros((3),dtype=[('Galaxy Name','U3'),('Halo Mass (1e12 solar masses)',float),
                                ('Disk Mass (1e12 solar masses)',float),
                                ('Buldge Mass (1e12 solar masses)',float),
                                ('Total Mass (1e12 solar masses)',float),
                                ('Baryon Fraction (1e12 solar masses)',float)]) # Initializes a table with the desired column names and types
    table['Galaxy Name'] = ["MW", "M31", "M33"] # Set galaxy names
    for row in table: # Iterate through every row of the table, each representing a galaxy
        galaxy = row['Galaxy Name'] # Get the name of the galaxy
        row['Halo Mass (1e12 solar masses)'] = ComponentMass('~'+galaxy+'_000.txt',1.0) # Get the dark matter halo mass at t=0
        row['Disk Mass (1e12 solar masses)'] = ComponentMass('~'+galaxy+'_000.txt',2.0) # Get the disk mass at t=0
        row['Buldge Mass (1e12 solar masses)'] = ComponentMass('~'+galaxy+'_000.txt',3.0) # Get the buldge mass at t=0
        row['Total Mass (1e12 solar masses)'] = row['Halo Mass (1e12 solar masses)']+row['Disk Mass (1e12 solar masses)']+row['Buldge Mass (1e12 solar masses)'] # Calculate total mass from component masses
        row['Baryon Fraction (1e12 solar masses)'] = np.around((row['Disk Mass (1e12 solar masses)']+row['Buldge Mass (1e12 solar masses)'])/row['Total Mass (1e12 solar masses)'],3) # Get baryon fraction from component masses
    print(table) # Print out table of the properties of the galaxy components