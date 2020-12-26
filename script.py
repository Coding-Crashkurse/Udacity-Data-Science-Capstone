import numpy as np
import pandas as pd
from datetime import datetime
import itertools
from operator import itemgetter


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
portfolio

offers = transcript.copy()
offers["offer_id"] = offers["value"].apply(lambda x: list(x.values())[0] if list(x.keys())[0] in ("offer id", "offer_id") else np.nan)
offers["reward"] = offers["value"].apply(lambda x: list(x.values())[1] if list(x.keys())[0] == "offer_id" else np.nan)
offers["amount"] = offers["value"].apply(lambda x: list(x.values())[0] if list(x.keys())[0] == "amount" else np.nan)


### Responsiveness
def create_single_customer_history(customer_id):
    one_person = offers[offers["person"] == customer_id]
    is_offer_viewed = False
    now_offer_id = np.nan
    
    for index, row in one_person.iterrows():
        if row['event'] == "offer viewed":
            is_offer_viewed = True
            now_offer_id = row['offer_id']
            
        elif row['event'] == "transaction" and is_offer_viewed:
            one_person.at[index, 'offer_id'] = now_offer_id
    
        elif row['event'] == "offer completed":
            is_offer_viewed = False
            now_offer_id = np.nan
    
    one_person["offer_id"] = one_person["offer_id"].replace(np.nan, "without offer")
    return one_person
    
def reaction_to_offer(df, offer_id):
    
    one_id = df[df["offer_id"] == offer_id]
    events = one_id["event"].to_list()
    
    if not events:
        return "not received"
    elif "offer completed" in events:
        return "completed"
    elif "offer viewed" in events:
        return "viewed"
    elif "offer received" in events:
        return "unresponsive"


response_df = pd.DataFrame(index=profile.id, columns=portfolio["offer_id"])

for index, person in enumerate(response_df.index):
    print(f'{index}/{len(response_df)} verarbeitet...')
    one_person = create_single_customer_history(person)
    for offer in response_df.columns:
        response = reaction_to_offer(one_person, offer)
        response_df.loc[person, offer] = response
        
one_person = create_single_customer_history("e4052622e5ba45a8b96b59aba68cf068")
one_person


def get_spendings(df):
    amount = df.groupby("offer_id", as_index=False)["amount"].sum()
    amount = amount.transpose()
    amount.columns = amount.iloc[0]
    amount.drop(amount.index[0], inplace=True)
    amount["person"] = df["person"].iloc[0]
    amount.set_index("person", inplace=True, drop=True)
    return amount




cols = portfolio["offer_id"].to_list()
cols.extend(["without offer"])

amount_df = pd.DataFrame(index=None, columns=cols)
amount_df


### Profile
profile["id"].duplicated().sum()
profile["age"] = profile["age"].apply(lambda x: np.nan if x == 118 else x)
profile["membersince"] = datetime.today().date() - pd.to_datetime(profile["became_member_on"], format='%Y%m%d').dt.date

