#acquisition
import pandas as pd
import numpy as np
artists_path = '../data/raw/artists.csv'

def acquisition(artists_path):
    artists = pd.read_csv(artists_path)
    print('Charging your network with paintings from some of the most influential painters in history...')
    data_artists = artists[artists['paintings'] >= 239].reset_index(drop=True)
    data_artists.to_csv('../data/processed/data_artists.csv')
    print(f"It was hard, but finally I made my decision! These are the 11 artists I decided to analyze {data_artists['name'].tolist()}")
    return data_artists