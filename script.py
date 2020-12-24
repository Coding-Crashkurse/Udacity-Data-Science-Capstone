import numpy as np
import pandas as pd
from datetime import datetime
import itertools

portfolio = pd.read_json('C:/Users/User/Desktop/Udacity/data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('C:/Users/User/Desktop/Udacity/data/profile.json', orient='records', lines=True)
transcript = pd.read_json('C:/Users/User/Desktop/Udacity/data/transcript.json', orient='records', lines=True)

### Analysis of profile
### Portfolio
portfolio["channels"].str.join('|').str.get_dummies()
portfolio = pd.concat([portfolio, portfolio["channels"].str.join('|').str.get_dummies()], axis = 1)
portfolio.drop(columns = ["channels"], inplace=True)
portfolio = pd.get_dummies(portfolio, columns=["offer_type"], prefix="d")
portfolio.rename(columns={"id": "offer_id"}, inplace=True)

### Profile

profile["id"].duplicated().sum()
profile["age"] = profile["age"].apply(lambda x: np.nan if x == 118 else x)

profile["membersince"] = datetime.today().date() - pd.to_datetime(profile["became_member_on"], format='%Y%m%d').dt.date
profile["membersince"]


transcript['value'].dtype


x = {'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9'}
x = list(itertools.chain.from_iterable(x.items()))




transcript["value"].apply(lambda x: True if ("offer") in x else x)

res = pd.DataFrame(transcript["value"].apply(lambda x: list(itertools.chain.from_iterable(x.items()))).to_list())

#transscript = pd.concat([transcript, type_and_offer])
#transscript


"offer_id" in {'offer_id': '2906b810c7d4411798c6938adc9daaa5'}

"x" in ["abs", "xyz"]
