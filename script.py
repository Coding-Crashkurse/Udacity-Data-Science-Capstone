import numpy as np
import pandas as pd

portfolio = pd.read_json('C:/Users/User/Desktop/Udacity/data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('C:/Users/User/Desktop/Udacity/data/profile.json', orient='records', lines=True)
transcript = pd.read_json('C:/Users/User/Desktop/Udacity/data/transcript.json', orient='records', lines=True)

### Analysis of profile

profile.shape
profile.isnull().sum()

profile[profile["age"] == 118]["age"].count() / len(profile)
