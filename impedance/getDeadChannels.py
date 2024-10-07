"""
Created on 22/3/2024
@author: Athina Apostolelli
"""

import os
import pandas as pd


def find_highImp_channels(session, threshold):
    df_left = pd.read_csv(session, usecols=lambda x: x.lower() in ['channel name', 'impedance magnitude at 1000 hz (ohms)'], skiprows = list(range(65,129)))
    df_right = pd.read_csv(session, usecols=lambda x: x.lower() in ['channel name', 'impedance magnitude at 1000 hz (ohms)'], skiprows = list(range(1,65)))
                
    df_left.rename(columns={'Impedance Magnitude at 1000 Hz (ohms)': 'Impedance'}, inplace=True)
    df_right.rename(columns={'Impedance Magnitude at 1000 Hz (ohms)': 'Impedance'}, inplace=True)
    
    # Find channels with high impedance 
    dead_left = df_left[(df_left['Impedance'] <= threshold[0]) | (df_left['Impedance'] >= threshold[1])]
    dead_right = df_right[(df_right['Impedance'] <= threshold[0]) | (df_left['Impedance'] >= threshold[1])]

    return dead_left, dead_right

    
if __name__ == "__main__":

    # Define parameters
    animals = ['rEO_06']
    basepath = 'D:\Rat_Recording'
    threshold = 2000000
    output_dir = os.path.join(basepath, 'eminhan_impedances')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Import impedance 
    for animal in animals:
        if animal == 'rEO_06':
            impedance_files = [os.path.join(basepath, animal,'9_impedance.csv')] 

            for session in impedance_files:
                dead_left, dead_right = find_highImp_channels(session, threshold)

                print('Session %s dead channels left hemi: ' %session, (dead_left))
                print('Session %s dead channels right hemi: ' %session, (dead_right))
