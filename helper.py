from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


from imblearn.over_sampling import SMOTE


def clean_profile(profile):
    profile.dropna(inplace=True)
    profile['age'] = profile['age'].apply(lambda x: np.nan if x == 118 else x)
    profile['memberdays'] = datetime.today().date() - pd.to_datetime(profile['became_member_on'], format='%Y%m%d').dt.date
    profile['memberdays'] = profile['memberdays'].dt.days
    profile = pd.get_dummies(data=profile, columns=["gender"])
    profile.drop("gender_O", axis=1, inplace=True)
    profile.reset_index(drop=True, inplace=True)
    return(profile)

def clean_portfolio(portfolio):
    portfolio["channels"].str.join('|').str.get_dummies()
    portfolio = pd.concat([portfolio, portfolio["channels"].str.join('|').str.get_dummies()], axis = 1)
    portfolio.drop(columns = ["channels"], inplace=True)
    portfolio = pd.get_dummies(portfolio, columns=["offer_type"], prefix="d")
    portfolio.rename(columns={"id": "offer_id"}, inplace=True)
    return portfolio

def clean_transcript(transcript):
    offers = transcript.copy()
    offers["offer_id"] = offers["value"].apply(lambda x: list(x.values())[0] if list(x.keys())[0] in ("offer id", "offer_id") else np.nan)
    offers["reward"] = offers["value"].apply(lambda x: list(x.values())[1] if list(x.keys())[0] == "offer_id" else np.nan)
    offers["amount"] = offers["value"].apply(lambda x: list(x.values())[0] if list(x.keys())[0] == "amount" else np.nan)
    return offers

def create_single_customer_history(offers, customer_id):
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

def create_response_df(profile, portfolio, offers):
    response_df = pd.DataFrame(index=profile.id, columns=portfolio["offer_id"])

    for index, person in enumerate(response_df.index):
        print(f'{index}/{len(response_df)} done...')
        one_person = create_single_customer_history(offers, person)
        for offer in response_df.columns:
            response = reaction_to_offer(one_person, offer)
            response_df.loc[person, offer] = response
    return response_df

def summarise_spendings(person):
    spendings = person.groupby("offer_id", as_index=False)["amount"].sum()
    spendings = spendings.transpose()
    spendings.columns = spendings.iloc[0]
    spendings.drop(spendings.index[0], inplace=True)
    spendings["person"] = person["person"].iloc[0]
    spendings.set_index("person", inplace=True, drop=True)
    return spendings

def create_spendings(portfolio, offers, person_df):
    cols = portfolio["offer_id"].to_list()
    cols.extend(["without offer"])
    persons = person_df["id"].to_list()
    amount_df = pd.DataFrame(index=None, columns=cols)
    
    for index, person in enumerate(persons):
        print(f'{index}/{len(persons)} done...')
        one_person_df = create_single_customer_history(offers, person)
        personal_spendings = summarise_spendings(one_person_df)
        amount_df = pd.concat([amount_df, personal_spendings])
    return amount_df

def plot_dflist(dflist, hue_var, grid=True):
    if(grid):
        fig, axes = plt.subplots(ncols=2, nrows=5)
        for i, ax in zip(range(len(dflist)), axes.flat):
            plot = sns.barplot(data=dflist[i], x="offer", y="value", hue=hue_var, ax=ax)
            plot.legend_.remove()
        plt.figure(figsize=(10,8))
        plt.show()
    else:
        for i, df in enumerate(dflist):
            plt.figure(i)
            sns.barplot(data=df, x="offer", y="value", hue=hue_var)

def build_prediction_score(classifier, profile, Y_var, oversampling):
    scaler = MinMaxScaler()
    LE = LabelEncoder()
    ros = SMOTE(sampling_strategy="auto", random_state=0, k_neighbors=5, n_jobs=-1)
    prediction_df = profile[["age", "income", "memberdays", "gender_F", "gender_M", "overall_spendings" , Y_var]]
    prediction_df = prediction_df[prediction_df[Y_var] != "not received"]
    prediction_df[["age", "income", "memberdays", "overall_spendings"]] = scaler.fit_transform(prediction_df[["age", "income", "memberdays", "overall_spendings"]])
    prediction_df[Y_var] = LE.fit_transform(prediction_df[Y_var])
    X = prediction_df.drop(Y_var, axis=1)
    y = prediction_df[Y_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    if oversampling:
        X_train, y_train = ros.fit_resample(X_train, y_train)
    clf = classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1score = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    return f1score, y_pred, cm


def use_best_model(profile, col, oversampling):
        classifier_list = ["logreg", "knn", "tree", "rf", "boostedtree", "gbc"]
        classifier_dict = {
            "logreg": OneVsRestClassifier(LogisticRegression(), n_jobs=-1),
            "knn": OneVsRestClassifier(KNeighborsClassifier(), n_jobs=-1),
            "tree": OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=-1),
            "rf": OneVsRestClassifier(RandomForestClassifier(), n_jobs=-1),
            "boostedtree": OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier(), learning_rate=0.1), n_jobs=-1),
            "gbc": OneVsRestClassifier(GradientBoostingClassifier(), n_jobs=-1)
            }
        
        score_list = []
        for idx, clf in enumerate(classifier_list):
            score, y_pred, cm = build_prediction_score(classifier_dict.get(classifier_list[idx]), profile, Y_var=col, oversampling=True)
            score_list.append(score)
        best_idx = score_list.index(max(score_list))
        print(f'Best model {classifier_list[best_idx]} with f-score: {score_list[best_idx]}')
        best_prediction = build_prediction_score(classifier_dict.get(classifier_list[best_idx]), profile, Y_var=col, oversampling=True)
        return best_prediction
    
   
    
    
    
    
    