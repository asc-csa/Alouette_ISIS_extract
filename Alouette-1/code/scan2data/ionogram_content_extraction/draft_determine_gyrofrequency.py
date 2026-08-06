# -*- coding: utf-8 -*-
"""
Starting code to determine gyrofrequency
"""

# Library imports
import pandas as pd

# Equations to solve

#fN**2 = fZ**2 + fZ + fH
#fT**2 = fH**2 + fZ**2 + fZ*fH
#fX = fH + fZ

def equations(p): 
    y,z,t = p 
    f1 = -10*z*t + 4*y*z*t - 5*y*t + 4*t*z^2 - 7 
    f2 = 2*y*z*t + 5*y*t - 3 
    f3 = - 10*t + 2*y*t + 4*z*t - 1 
    return (f1,f2,f3) 

#y,z,t = fsolve(equations) 
    
if __name__ == '__main__':
    path_master_csv = 'E:/AlouetteData/Alouette Data/merge_clean_meta_data.csv'

    df = pd.read_csv(path_master_csv,nrows=1000,usecols=['days','year','Lat','Lon'])
    index = df.isnull().any(axis = 1)
    rows_with_null = df[index ]
    rows_without_null = df[~index ]
    
    year = rows_without_null['year'].tolist()
    days = rows_without_null['days'].tolist()
    
    df['time_reformatted'] = rows_without_null.apply(lambda row: pd.to_datetime(str(int(row['year'])) + '+' + str(int(row['days'])),format="%Y+%j",errors='coerce' ) ,axis = 1)
    df['time_reformatted'] = df['time_reformatted'].astype(str).str[:]