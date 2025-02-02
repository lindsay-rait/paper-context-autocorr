#import libraries 
import pandas as pd
from os.path import basename
from pathlib import Path
import os
import glob
import pandas as pd
import numpy as np
#from scipy.stats import pearsonr
from scipy.interpolate import CubicSpline
import re


# Read the CSV file
switch11_fmri_beh = pd.read_csv("switchfreq11_fmri_rawdata_encoding2.csv")

# Display the DataFrame
print(switch11_fmri_beh)


# Select columns and clean image path
switch11_fmri_beh_select = (
    switch11_fmri_beh[['subject', 'blocknum', 'image', 'trialnum', 'condition', 'image_binary']])
    
switch11_fmri_beh_select['image'] = switch11_fmri_beh_select['image'].apply(lambda x: os.path.basename(x).replace('.jpg', ''))

    
# Display the selected DataFrame
print(switch11_fmri_beh_select)


#load in subject directory 
sub_dir = glob.glob("csv_files/fMRI_raw_new/*/")

sub_dir = sorted(sub_dir)
print(sub_dir)

#fMRI data visual regions 

rois_names = ['posterior_hippocampus_v2']

sf11_fmri_enc_clean_df = pd.DataFrame()

for sub_folder in sub_dir:
    print(sub_folder)
    for roi in rois_names:
        print(roi)
        files = glob.glob(os.path.join(sub_folder, f'*{roi}*enc*_v2.csv'))
        files = sorted(files)
        
        # Process each CSV file
        data_frames = []
        for file in files:
            # Read CSV file
            df = pd.read_csv(file)
            
            # Remove columns with all zeros- makes them NA to be removed later 
            df = df.loc[:, (df != 0).any(axis=0)]
            
            # Append the processed DataFrame to the list
            data_frames.append(df)
        
        curr_df = pd.concat(data_frames, ignore_index=True)
        
        # Check if 'run' column exists
        if 'run' in curr_df.columns:
            blocknum = pd.to_numeric(curr_df['run'].str.extract(r'([0-9]+)', expand=False))
        else:
            # Use a placeholder value (you may adjust this based on your data)
            blocknum = 0
        
        clean_df = curr_df.assign(
            blocknum=blocknum,
            subject=pd.to_numeric(curr_df['sub'].str.extract(r'([0-9]+)', expand=False)),
            trialnum=curr_df['Unnamed: 0'] + 1
        ).drop(['run', 'sub', 'Unnamed: 0'], axis=1)
        
        sf11_fmri_enc_clean = pd.merge(clean_df, switch11_fmri_beh_select,
                                      left_on=['subject', 'blocknum', 'trialnum'],
                                      right_on=['subject', 'blocknum', 'trialnum'],
                                      how='left')
        sf11_fmri_enc_clean['condition'].ffill(inplace=True)
        
        sf11_fmri_enc_clean_df = pd.concat([sf11_fmri_enc_clean_df, sf11_fmri_enc_clean], ignore_index=True)

print(sf11_fmri_enc_clean_df)



#only includes rows for encoding, not for math and recall 
sf11_fmri_enc_clean_df_2 = sf11_fmri_enc_clean_df.groupby(['subject', 'roi', 'blocknum']).apply(
    lambda group: group[group['trialnum'] < 102]
).reset_index(drop=True)

sf11_fmri_enc_clean_df_2['trialnum']


#remove columns 'image' and 'imagebinary'
columns_to_remove = ['image', 'image_binary']

# Use the drop method to remove the specified columns
sf11_fmri_enc_clean_df_2 = sf11_fmri_enc_clean_df_2.drop(columns=columns_to_remove)




#rois
roi_posterior_hippocampus = 'posterior_hippocampus_v2'

# Filter the DataFrame based on the "roi" column
sf11_fmri_enc_df_posterior_hippocampus = sf11_fmri_enc_clean_df_2[sf11_fmri_enc_clean_df_2["roi"] == roi_posterior_hippocampus]

# Drop columns that are completely filled with NaN values- entire run is high motion
sf11_fmri_enc_df_posterior_hippocampus = sf11_fmri_enc_df_posterior_hippocampus.dropna(axis=1, how="all")

print(sf11_fmri_enc_df_posterior_hippocampus)


#I removed outlier fMRI volumes in my data. This leaves rows that are all 0. 
#I replaced them by interpolating voxel values from adjacent volumes, using cubic spline interpolation in python.

def interpolate_subject(subject_df, epsilon=2e-5):
    """
    Interpolate zero and extremely small values in the input subject's data using cubic spline interpolation.

    Parameters:
    - subject_df: Pandas DataFrame representing fMRI data for a specific subject.
    - epsilon: A small threshold for identifying extremely small values.

    Returns:
    - Interpolated fMRI data for the subject.
    """
    # Identify columns with numeric titles
    numeric_columns = [col for col in subject_df.columns if col.isnumeric()]

    # Iterate over each column in the subject-specific DataFrame
    for col in numeric_columns:
        # Extract the numeric column
        numeric_column = pd.to_numeric(subject_df[col], errors='coerce')

        # Find indices of zero and small values in the numeric column
        zero_indices = np.where((numeric_column.values == 0) | (np.abs(numeric_column.values) <= epsilon))
        #print("zero_indices:", zero_indices)

        # Check if there are zero or small values in the column
        if zero_indices[0].size > 0:
            # Extract non-zero and non-small values in the numeric column
            non_zero_values = numeric_column[(numeric_column != 0) & (np.abs(numeric_column) > epsilon)]

            # Create an array of indices for the non-zero and non-small values
            indices = np.arange(len(non_zero_values))

            # Perform cubic spline interpolation with NaN handling
            cubic_spline = CubicSpline(indices, non_zero_values, extrapolate=False, bc_type='natural')

            # Iterate over each row of zero and small value indices in the numeric column
            for row in zero_indices[0]:
                # Print information about the interpolation
                print(f"Subject: {subject_df['subject'].iloc[0]}, Column: {col}, Row: {row}, Interpolated Value: {cubic_spline(row)}") 

                # Replace zero or small values with the interpolated values in the subject-specific DataFrame
                subject_df.iloc[row, subject_df.columns.get_loc(col)] = cubic_spline(row)

    return subject_df

def interpolate_all_subjects(sf11_fmri_enc_df_posterior_hippocampus):
    """
    Interpolate zero and extremely small values for each subject in the input DataFrame.

    Parameters:
    - df: Pandas DataFrame representing fMRI data for multiple subjects.

    Returns:
    - Interpolated fMRI data for all subjects.
    """
    # Group the DataFrame by subject and apply interpolation to each group
    interpolated_dfs = [interpolate_subject(subject_df) for _, subject_df in sf11_fmri_enc_df_posterior_hippocampus.groupby('subject')]

    # Concatenate the results back into a single DataFrame
    interpolated_result = pd.concat(interpolated_dfs, ignore_index=True)

    return interpolated_result
                      
# Call the function to interpolate zero and small values for each subject
df_interpolated_posterior_hippocampus = interpolate_all_subjects(sf11_fmri_enc_df_posterior_hippocampus)


                      
                      
                      
#### single voxel autocorrelation -- posterior_hippocampus


subnum = list(range(4, 7)) + list(range(8, 41)) + [41, 43]
blocknum = list(range(1, 9))


#subnum = list(range(4, 7))
#blocknum = list(range(8, 9))

posterior_hippocampus_autocorrelation_df_lag1_v2 = pd.DataFrame()

for i in subnum:
    print(i)
    for j in blocknum:
        print(j)
        autocorr_df = df_interpolated_posterior_hippocampus[
            (df_interpolated_posterior_hippocampus['blocknum'] == j) &
            (df_interpolated_posterior_hippocampus['subject'] == i)
        ]

        autocorr_df = autocorr_df.dropna(how="all", axis=1)

        factor_cols = ["roi", "subject", "blocknum", "condition", "trialnum"]
        autocorr_df[factor_cols] = autocorr_df[factor_cols].astype('category')
        autocorr_df_factor = autocorr_df.select_dtypes(include='category')

        autocorr_df_numeric = autocorr_df.select_dtypes(include=np.number)
        
        #If too much motion in a given run, skip over the run because could not interpolate 
        if autocorr_df_numeric.isna().any().any():
            print(f"Skipping subject {i}, block {j} due to NaN values.")
            continue

        for l in autocorr_df_numeric.columns:
            #no lag, original timecourse 
            autocorr_df_nolag = autocorr_df_numeric[l].dropna()

            # LAG 1
            #shifted by 1 TR 
            autocorr_df_lag1 = autocorr_df_numeric[l].shift(1)
            masked_lag1 = np.ma.masked_invalid(autocorr_df_lag1)
            posterior_hippocampus_cor_1 = np.ma.corrcoef(autocorr_df_nolag, masked_lag1)[0, 1]
            posterior_hippocampus_cor_1_with_block = pd.DataFrame({
                "autocor_value_lag1": [posterior_hippocampus_cor_1],
                "blocknum": [j],
                "subject": [i],
                "lag_num_lag1": [1]
            })
            posterior_hippocampus_cor_1_with_block2 = pd.merge(
                posterior_hippocampus_cor_1_with_block,
                autocorr_df_factor[['condition', 'roi', 'subject', 'blocknum']],
                on=['subject', 'blocknum'],
                how='left'
            )
            posterior_hippocampus_cor_1_with_block2 = posterior_hippocampus_cor_1_with_block2.drop_duplicates(subset=['autocor_value_lag1', 'blocknum', 'subject', 'lag_num_lag1'])
            posterior_hippocampus_autocorrelation_df_lag1_v2 = posterior_hippocampus_autocorrelation_df_lag1_v2.append(posterior_hippocampus_cor_1_with_block2, ignore_index=True)

        
                      
                                   
                      
# Specify the file path where you want to save the CSV file
csv_file_path_1 = 'posterior_hippocampus_autocorrelation_df_lag1_v2.csv'

# Export the DataFrame to a CSV file
posterior_hippocampus_autocorrelation_df_lag1_v2.to_csv(csv_file_path_1, index=False)  
                      
                      