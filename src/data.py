from pathlib import Path
from datetime import datetime, timedelta
import requests
import os

import numpy as np
import pandas as pd
from tqdm import tqdm 

# Use absolute import
from src.paths import RAW_DATA_DIR

def download_one_file_of_raw_data(year: int, month: int) -> Path:
    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    print(f'Attempting to download from URL: {URL}')
    response = requests.get(URL)
    print(f'Response status code: {response.status_code}')

    if response.status_code == 200:
        path = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        print(f'Saving file to {path}')
        with open(path, "wb") as f:
            f.write(response.content)
        print(f'Downloaded file to {path}')
        return path
    else:
        raise Exception(f'{URL} is not available')

def validate_raw_data(
        rides: pd.DataFrame,
        year: int,
        month: int
        ) -> pd.DataFrame:
    """
    Removes rows with pickup_datetimes outside their valid range
    """
    # keep only rides for this month
    this_month_start = f'{year}-{month:02d}-01'
    next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    rides = rides[rides.pickup_datetime >= this_month_start]
    rides = rides[rides.pickup_datetime < next_month_start]

    return rides

def load_raw_data(
        year: int,
        months=None
) -> pd.DataFrame:
    """
    Loads raw data from local storage or downloads it from the NYC taxi website,
    and then loads it into a Pandas dataframe

    Args:
        year: year of the data to be download
        months: months of the data to be download

    Returns:
        pd.DataFrame: DataFrame with the following columns:
            - pickup_datetime: datetime of the pickup
            - pickup_location_id: ID of the pickup location
    """

    # create empty dataframe to store rides data
    rides = pd.DataFrame()

    # if months isn't specified, download data for the entire year
    if months is None:
        months = list(range(1, 13))
    elif isinstance(months, int):
        # if months is specified, download it as a list
        months = [months]

    # for each month in months list, create a variable called local_file
    for month in months:
        local_file = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        
        # Debug print
        print(f'Checking for file: {local_file}')

        # if local_file does not exist yet, download the file
        if not local_file.exists():
            try:
                # download file from the NYC taxi website
                print(f'Downloading file {year}-{month:02d}')
                download_one_file_of_raw_data(year, month)
            except Exception as e:
                print(f'{year}-{month:02d} file is not available: {e}')
                continue

        # if local_file exists, tell us that it is already in local storage
        else:
            print(f'File {year}-{month:02d} was already in local storage: {local_file}')

        # Debug print: List files in RAW_DATA_DIR
        print(f'Files in RAW_DATA_DIR: {os.listdir(RAW_DATA_DIR)}')

        # load the file into Pandas
        try:
            rides_one_month = pd.read_parquet(local_file)
            print(f'Successfully loaded file: {local_file}')
        except Exception as e:
            print(f'Error loading file {local_file}: {e}')
            continue

        # rename columns
        rides_one_month = rides_one_month[['tpep_pickup_datetime', 'PULocationID']]
        rides_one_month.rename(columns={
            'tpep_pickup_datetime': 'pickup_datetime',
            'PULocationID': 'pickup_location_id'
        }, inplace=True)

        # validate the file
        rides_one_month = validate_raw_data(rides_one_month, year, month)

        # append the existing data
        rides = pd.concat([rides, rides_one_month])

    if rides.empty:
        # return empty dataframe if rides is empty
        return pd.DataFrame()
    else:
        # keep only time and origin of the ride
        rides = rides[['pickup_datetime', 'pickup_location_id']]
        return rides

def add_missing_slots(ts_data: pd.DataFrame) -> pd.DataFrame:

    location_ids = ts_data['pickup_location_id'].unique()

    full_range = pd.date_range(ts_data['pickup_hour'].min(),
                               ts_data['pickup_hour'].max(),
                               freq='h')
    
    output = pd.DataFrame()

    # loop through all the unique IDs in the `pickup_location_id` columns
    # this loop adds each unique locaiton_id's pickup_hour and rides to the output dataframe
    for location_id in tqdm(location_ids):

        # keep only pickup_hour and rides for this location_id
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, ['pickup_hour', 'rides']]

        if ts_data_i.empty:
            ts_data_i = pd.DataFrame.from_dict([
                {'pickup_hour':ts_data['pickup_hour'].max(), 'rides':0}
            ])

        # quick way to add missing dates with 0 in a pandas series
        # set pickup_hour as the index
        ts_data_i.set_index('pickup_hour', inplace=True)
        # change the index to DateTime dtype
        ts_data_i.index = pd.DatetimeIndex(ts_data_i.index)
        # reindex agg_rides_i to align with the list of hours in full_range; empty hours will be assigned a value of 0
        ts_data_i = ts_data_i.reindex(full_range, fill_value=0)

        # add back location_id columns
        ts_data_i['pickup_location_id'] = location_id

        # concats location_id to the output dataframe defined above
        output = pd.concat([output, ts_data_i])

    # move the purchase_day from the index to a dataframe column
    output = output.reset_index().rename(columns={'index':'pickup_hour'})

    return output

def transform_raw_data_into_ts_data(rides: pd.DataFrame) -> pd.DataFrame:
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('h')
    agg_rides = rides.groupby(['pickup_hour','pickup_location_id']).size().reset_index()
    agg_rides.rename(columns={0:'rides'}, inplace=True)
    
    # add rows for (locations, pickup_hours)s with 0 rides
    agg_rides_all_slots = add_missing_slots(agg_rides)

    return agg_rides_all_slots

def transform_ts_data_into_features_and_target(
        ts_data: pd.DataFrame,
        input_seq_len: int,
        step_size: int
):
    """
    Slices and transposes data from time-series format into a (features, target)
    format that we can use to train supervised ML models
    """

    assert set(ts_data.columns) == {'pickup_hour', 'rides', 'pickup_location_id'}

    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()

    full_range = pd.date_range(
        ts_data['pickup_hour'].min(), ts_data['pickup_hour'].max(), freq='H')
    
    output = pd.DataFrame()

    # loop through all the unique IDs in the `pickup_location_id` columns
    # this loop adds each unique locaiton_id's pickup_hour and rides to the output dataframe
    for location_id in tqdm(location_ids):

        # keep only pickup_hour and rides for this location_id
        ts_data_one_location = ts_data.loc[
            ts_data.pickup_location_id == location_id,
            ['pickup_hour', 'rides']
            ].sort_values(by=['pickup_hour'])
        
        # pre-compute cutoff indices to split dataframe rows
        indices = get_cutoff_indices_features_and_target(
            ts_data_one_location,
            input_seq_len,
            step_size
        )

        # slice and transpose data into numpy arrays for features and targets
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        pickup_hours = []

        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

         # numpy -> pandas
        features_one_location = pd.DataFrame(
            x,
            columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['pickup_hour'] = pickup_hours
        features_one_location['pickup_location_id'] = location_id

        # numpy -> pandas
        targets_one_location = pd.DataFrame(y, columns=[f'target_rides_next_hour'])

        # concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets['target_rides_next_hour']


def get_cutoff_indices_features_and_target(
    data: pd.DataFrame,
    input_seq_len: int,
    step_size: int
    ) -> list:

        stop_position = len(data) - 1
        
        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        subseq_mid_idx = input_seq_len
        subseq_last_idx = input_seq_len + 1
        indices = []
        
        while subseq_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
            subseq_first_idx += step_size
            subseq_mid_idx += step_size
            subseq_last_idx += step_size

        return indices