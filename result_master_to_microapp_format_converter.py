# This script reformats the output CSV file from Alouette Extract to the format of the micro application
#
# @author Emiline Filion - Canadian Space Agency
#

import pandas as pd

print("\nLoading the CSV file...")
df = pd.read_csv(r'data/result_master.csv')
print("Loading the CSV file... done\n")
print("Reformatting the CSV file...")

# Rename colunms as required by the micro application
df.rename(columns={'Lng': 'lon', 
                   'Lat': 'lat', 
                   'Station_Name': 'station_name', 
                   'Station_Code': '3_letter_code',
                   'Station_Number': 'station_number',
                   'Timestamp': 'timestamp',
                   'filename': 'file_name',
                   'Subdirectory': 'subdir_name'}, inplace=True)
print("Colunms renamed")

# Delete unnecessary colunms
df.drop('processed_image_class', axis=1, inplace=True)
df.drop('time_quality', axis=1, inplace=True)
print("Unnecessary colunms deleted")

# Add satellite_number
df['satellite_number'] = '1'
print("satellite_number added")

# Iterate through each row and update the file name
print("Formating the file name... (takes several minutes)")
for index, row in df.iterrows():

    df.at[index, "file_name"] = row['Directory'] + '/' + row['subdir_name'] + '/' + row['file_name'].replace(".png", '')
    #df.at[index, "station_name"] = row['station_name'].replace(".No. W. Territories", 'NT, Canada')

# Delete the directory colunm
df.drop('Directory', axis=1, inplace=True)

# Reorder columns
columns_titles = ['file_name','fmin','max_depth','subdir_name','satellite_number','station_number','timestamp','station_name','3_letter_code','lat','lon']
df=df.reindex(columns=columns_titles)

# Save to CSV
df.to_csv('data/final_alouette_data.csv', index=False)

print("\nThe script has ended successfully - Have a nice day!")