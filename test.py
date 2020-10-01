#%%
import pandas as pd

#%%
data = pd.read_csv("data/projeto4_telecom_treino.csv")

#%%
data.columns

#%%
feature = "account_length"

churn_by_feature = data.groupby([feature, "churn"]).agg(
    count=pd.NamedAgg(column="churn", aggfunc="count")
)

rows_present = [row[0] for row in churn_by_feature.iterrows()]

rows_absent = [
    (value, churn_)
    for value in data[feature].unique()
    for churn_ in ["no", "yes"]
    if (value, churn_) not in rows_present
]

rows_absent = pd.DataFrame(
    rows_absent,
    columns=[feature, "churn"],
)

rows_absent["count"] = 0

churn_by_feature = churn_by_feature.reset_index()
churn_by_feature = churn_by_feature.append(rows_absent)

churn_by_feature = churn_by_feature.sort_values(by=[feature, "churn"]).reset_index(
    drop=True
)

churn_by_feature

#%%
