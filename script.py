import numpy as np
import pandas as pd
from datetime import datetime
import itertools
from operator import itemgetter
import seaborn as sns

portfolio = pd.read_json('C:/Users/User/Desktop/Udacity/data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('C:/Users/User/Desktop/Udacity/data/profile.json', orient='records', lines=True)
transcript = pd.read_json('C:/Users/User/Desktop/Udacity/data/transcript.json', orient='records', lines=True)

### Analysis of profile
profile.dropna(inplace=True)
profile['age'] = profile['age'].apply(lambda x: np.nan if x == 118 else x)
profile['memberdays'] = datetime.today().date() - pd.to_datetime(profile['became_member_on'], format='%Y%m%d').dt.date
profile['memberdays'] = profile['memberdays'].dt.days
profile = pd.get_dummies(data=profile, columns=["gender"])
profile.drop("gender_O", axis=1, inplace=True)


### Portfolio
portfolio["channels"].str.join('|').str.get_dummies()
portfolio = pd.concat([portfolio, portfolio["channels"].str.join('|').str.get_dummies()], axis = 1)
portfolio.drop(columns = ["channels"], inplace=True)
portfolio = pd.get_dummies(portfolio, columns=["offer_type"], prefix="d")
portfolio.rename(columns={"id": "offer_id"}, inplace=True)


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
        
def summarise_spendings(person):
    spendings = person.groupby("offer_id", as_index=False)["amount"].sum()
    spendings = spendings.transpose()
    spendings.columns = spendings.iloc[0]
    spendings.drop(spendings.index[0], inplace=True)
    spendings["person"] = person["person"].iloc[0]
    spendings.set_index("person", inplace=True, drop=True)
    return spendings

def create_spendings(person_df):
    cols = portfolio["offer_id"].to_list()
    cols.extend(["without offer"])
    persons = person_df["id"].to_list()
    amount_df = pd.DataFrame(index=None, columns=cols)
    
    for index, person in enumerate(persons):
        print(f'{index}/{len(persons)} verarbeitet...')
        one_person_df = create_single_customer_history(person)
        personal_spendings = summarise_spendings(one_person_df)
        amount_df = pd.concat([amount_df, personal_spendings])
    return amount_df

spendings_df = create_spendings(person_df=profile)
spendings_df["with offer"] = spendings_df.iloc[:,0:10].sum(axis=1)
spendings_df["overall_spendings"] = spendings_df[["without offer", "with offer"]].sum(axis=1)

### Build customer groups with clustering
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(profile[["age", "income", "memberdays"]])
profile[["age_scaled", "income_scaled", "memberdays_scaled"]] = scaler.transform(profile[["age", "income", "memberdays"]])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

X = profile[["age_scaled", "income_scaled", "memberdays_scaled", "gender_F", "gender_M"]]

sil_score_max = -1

for n_clusters in range(2,6):
  model = KMeans(n_clusters = n_clusters, init='k-means++', max_iter=100, n_init=1)
  labels = model.fit_predict(X)
  sil_score = silhouette_score(X, labels)
  print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
  if sil_score > sil_score_max:
    sil_score_max = sil_score
    best_n_clusters = n_clusters
    
final_model = KMeans(n_clusters = best_n_clusters, init='k-means++', max_iter=100, n_init=1)
final_model.fit_predict(X)
final_model.labels_

profile["cluster"] = final_model.labels_


