#
# reception_facilities.py
# Script that matches the names of the reception facilities between Alouette & ISIS.
#
# @source https://github.com/asc-csa/AlouetteApp
# @author Emiline Filion - Canadian Space Agency
#
# Modification History:
# June 2024: Initial version
#

import pandas as pd

# Constants
INPUT_RECEPTION_FACILITIES = "reception_facilities.csv"
INPUT_DATA = "final_alouette_data.csv"


# Loads matching matrix of reception facilities between Alouette & ISIS
# @return Dataframe that contains the match matrix.
def load_matching_reception_facilities():
    
    # Open the matching file
    matching_matrix = pd.read_csv(INPUT_RECEPTION_FACILITIES)
    return matching_matrix[['Alouette 1 Reception Facility', 'ISIS Reception Facility']]


# Applies the matching matrix, which adjust the names of the reception facilities in the input data file.
# @param input_csv_file Input CSV file (e.g. final_alouette_data.csv)
# @return Dataframe that contains data with the appropriate station names.
def apply_matching_matrix(input_csv_file):
    
    try:
        # Open data file
        df_data = pd.read_csv(input_csv_file)
        df_matching_matrix = load_matching_reception_facilities()
        
        # Loop through all reception facilities from the matching matrix
        for matching_case in df_matching_matrix.index:

            # Make sure the reception facility has the appropriate station name for each ionogram
            tmp_reception_facility_alouette = df_matching_matrix['Alouette 1 Reception Facility'][matching_case]
            tmp_reception_facility_isis = df_matching_matrix['ISIS Reception Facility'][matching_case]
            df_data.replace(tmp_reception_facility_isis, tmp_reception_facility_alouette, inplace=True)
    except Exception as e:
        print('Cannot open or process the input CSV file :' + str(e))

    return df_data

#======================================================================================
# Main part

# Debut
print("\nThis script adjust the name of the stations if the CSV file, which is used by the Alouette micro application.")
print("Input Data file: " + INPUT_DATA)
print("Matching matrix: " + INPUT_RECEPTION_FACILITIES)

# Rename the names of the reception facilities in the CSV file
print("Matching reception facilities...")
df_new_data = apply_matching_matrix(INPUT_DATA)

# Save the new data file
new_data_filename = INPUT_DATA.replace(".csv", "_new.csv")
df_new_data.to_csv(new_data_filename, index=False)

# The End
print("\nThe program ended successfully")
print("New data file to use in the Alouette micro application: " + new_data_filename)
print("Have a good day!\n")