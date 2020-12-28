import numpy as np
import pandas as pd
from datetime import datetime
import itertools
from operator import itemgetter
import seaborn as sns
import matplotlib.pyplot as plt

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
profile.reset_index(drop=True, inplace=True)

### Portfolio
portfolio["channels"].str.join('|').str.get_dummies()
portfolio = pd.concat([portfolio, portfolio["channels"].str.join('|').str.get_dummies()], axis = 1)
portfolio.drop(columns = ["channels"], inplace=True)
portfolio = pd.get_dummies(portfolio, columns=["offer_type"], prefix="d")
portfolio.rename(columns={"id": "offer_id"}, inplace=True)
sns.displot(data=profile, x="age")
sns.displot(data=profile, x="income")


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


def create_response_df():
    response_df = pd.DataFrame(index=profile.id, columns=portfolio["offer_id"])

    for index, person in enumerate(response_df.index):
        print(f'{index}/{len(response_df)} verarbeitet...')
        one_person = create_single_customer_history(person)
        for offer in response_df.columns:
            response = reaction_to_offer(one_person, offer)
            response_df.loc[person, offer] = response
    return response_df

response_df = create_response_df()

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
        
### EDA to check responsiveness between the soziodemographic variables
profile = profile.merge(response_df, left_on="id", right_index=True)
profile = profile.merge(spendings_df[["overall_spendings", "without offer", "with offer"]], left_on="id", right_index=True)
profile["income_cat"] = pd.cut(profile["income"], 3)
profile["age_cat"] = pd.cut(profile["age"], 5)
profile["spendings_cat"] = pd.cut(profile["overall_spendings"], 4)


relevant_columns = portfolio["offer_id"].to_list()
relevant_columns

profile[relevant_columns]
overall = profile[relevant_columns].mask(lambda x: x.eq('not received')).apply(pd.value_counts)
overall_perc = profile[relevant_columns].mask(lambda x: x.eq('not received')).apply(pd.value_counts, normalize=True).mul(100).round(4)

income_list = []
for col in relevant_columns:
    subset = profile[profile[col] != "not received"]
    cross_tab = pd.crosstab(subset["income_cat"], subset[col], normalize="index").round(4)*100
    result = cross_tab.stack().reset_index().rename(columns={0: "value"})
    result.rename(columns={result.columns[1] : "offer"}, inplace=True)
    income_list.append(result)  
  
age_list = []
for col in relevant_columns:
    subset = profile[profile[col] != "not received"]
    cross_tab = pd.crosstab(subset["age_cat"], subset[col], normalize="index").round(4)*100
    result = cross_tab.stack().reset_index().rename(columns={0: "value"})
    result.rename(columns={result.columns[1] : "offer"}, inplace=True)
    age_list.append(result)

gender_list = []
for col in relevant_columns:
    subset = profile[profile[col] != "not received"]
    cross_tab = pd.crosstab(subset["gender_M"], subset[col], normalize="index").round(4)*100
    result = cross_tab.stack().reset_index().rename(columns={0: "value"})
    result.rename(columns={result.columns[1] : "offer"}, inplace=True)
    gender_list.append(result)
    
spendings_list = []
for col in relevant_columns:
    subset = profile[profile[col] != "not received"]
    cross_tab = pd.crosstab(subset["spendings_cat"], subset[col], normalize="index").round(4)*100
    result = cross_tab.stack().reset_index().rename(columns={0: "value"})
    result.rename(columns={result.columns[1] : "offer"}, inplace=True)
    spendings_list.append(result)


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
        
        
plot_dflist(income_list, hue_var="income_cat")
plot_dflist(age_list, hue_var="age_cat")
plot_dflist(gender_list, hue_var="gender_M")
plot_dflist(spendings_list, hue_var="spendings_cat")

### Can we predict the responses to the offers?
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

random_state = 0
classifiers = []
classifiers.append(OneVsOneClassifier(LogisticRegression(), n_jobs=-1))
classifiers.append(OneVsOneClassifier(KNeighborsClassifier(), n_jobs=-1))
classifiers.append(OneVsOneClassifier(DecisionTreeClassifier(), n_jobs=-1))
classifiers.append(OneVsOneClassifier(RandomForestClassifier(), n_jobs=-1))
classifiers.append(OneVsOneClassifier(AdaBoostClassifier(DecisionTreeClassifier(), learning_rate=0.1), n_jobs=-1))
classifiers.append(OneVsOneClassifier(GradientBoostingClassifier(), n_jobs=-1))

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
scaler = MinMaxScaler()
LE = LabelEncoder()
ros = SMOTE(sampling_strategy="auto", random_state=0, k_neighbors=5, n_jobs=-1)

def build_prediction_df(profile, Y_var, oversampling=True):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import LabelEncoder
    scaler = MinMaxScaler()
    LE = LabelEncoder()
    prediction_df = profile[["age", "income", "memberdays", "gender_F", "gender_M", "overall_spendings" , Y_var]]
    prediction_df = prediction_df[prediction_df[Y_var] != "not received"]
    prediction_df[["age", "income", "memberdays", "overall_spendings"]] = scaler.fit_transform(prediction_df[["age", "income", "memberdays", "overall_spendings"]])
    prediction_df[Y_var] = LE.fit_transform(prediction_df[Y_var])
    X = prediction_df.drop(Y_var, axis=1)
    y = prediction_df[Y_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    if oversampling:
        X_train, y_train = ros.fit_resample(X_train, y_train)
    #print(f'N in training data: {len(y_train)}')
    clf = OneVsOneClassifier(GradientBoostingClassifier(), n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm

models = []
for column in relevant_columns:
    models.append(build_prediction_df(profile, column, oversampling=True))

for model in models:
    if len(model) == 3: ### Kick out informational
        print(model[1,1] / model[1].sum() * 100)


### Predict the spendings
sns.displot(profile, x="overall_spendings")

#profile_reg = profile_reg[profile_reg["overall_spendings"] < 250]
overall_spendings = profile[["income", "overall_spendings"]]
without_offer = profile[["income", "without offer"]].dropna().astype("int64")
with_offer = profile[["income", "with offer"]].dropna().astype("int64")

sns.displot(profile, x="overall_spendings")
sns.lineplot(data=overall_spendings, x="income", y="overall_spendings")
sns.lineplot(data=without_offer, x="income", y="without offer")
sns.lineplot(data=with_offer, x="income", y="with offer")


sns.lineplot(data=profile, x="income", y="overall_spendings")
sns.lineplot(data=profile, x="memberdays", y="overall_spendings")
sns.lineplot(data=profile, x="age", y="overall_spendings")
sns.heatmap(profile[["age", "income", "gender_F", "gender_M"]].corr(), annot=True)

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

prediction_df = profile[["age", "income", "memberdays", "gender_F", "gender_M", "overall_spendings"]]
#prediction_df = prediction_df[prediction_df["overall_spendings"] < 300]

prediction_df[["age", "income", "memberdays"]] = scaler.fit_transform(prediction_df[["age", "income", "memberdays"]])
X = prediction_df.drop("overall_spendings", axis=1)
y = prediction_df["overall_spendings"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

regr_trans = TransformedTargetRegressor(regressor=RidgeCV(), func=np.log1p, inverse_func=np.expm1)
regr_trans.fit(X_train, y_train)
y_pred = regr_trans.predict(X_test)
y_test.reset_index(drop=True, inplace=True)

result = pd.concat([y_test, pd.Series(y_pred)], axis=1)
result.rename(columns={0: "prediction", "overall_spendings": "actual_value"}, inplace=True)

mean_squared_error(y_test, y_pred)

sns.scatterplot(data=result, y="actual_value", x="prediction")



