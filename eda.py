#%%
import pandas as pd
import seaborn as sns

import eda_tools

#%%
data = pd.read_csv("data/projeto4_telecom_treino.csv")

#%% [markdown]
"""
As we can see, this dataset is comprised of a total of 3333 observations and 21 features, which are divided among the types "float64," "int64," and "object." The last feature, "churn," is the one we are going to try to predict.
"""

#%%
data.info()

#%% [markdown]
"""
I think it's useful to begin by taking a look at the customer churn. We can see this feature is encoded as an object type. About 14.5% (483) of the customers are in the "yes" class. Since the classes are imbalanced, we need to take care when choosing the proper metric to validate any kind of prediction.
"""

#%%
print(
    f'"Churn"  is encoded as {data["churn"].dtype}.\n',
    data["churn"].value_counts(),
    "",
    data["churn"].value_counts(normalize=True),
    sep="\n",
)

#%% [markdown]
"""
Having taken the churn out of the way, we might go on to take a look at the actual first feature, which is in fact unnamed. By computing this feature's difference from a straight sequence, beginning at 1, we can see it's in reality just the data frame index. For this reason, we can confidently drop it.
"""

#%%
idx_diff = data["Unnamed: 0"] - pd.Series(range(1, 3334))
idx_diff.sum()

#%%
data = data.drop(columns="Unnamed: 0")

#%% [markdown]
"""
It shouldn't come as a surprise that the feature named "state" is a USA state name abbreviation (+ DC).
"""

#%%
states = data["state"].unique()
print(
    f"States listed: {sorted(states)}",
    f'Number of states: {data["state"].nunique()}',
    sep="\n",
)

#%% [markdown]
"""
Let's see if we can find any kind of correlation among the state the customer is from and the churn. To do this, we group the customers by state and compute the percentage of them falling into each churn category.
"""

#%%
churn_by_state = eda_tools.group_by_feature(
    data, "state", churn="yes", normalize=True
).sort_values(by="percent", ascending=False)

#%% [markdown]
"""
We can see NJ and CA are the states with the highest churn (about 26%), while HI and AK present the lowest (about 6%).
"""

#%%
print(churn_by_state.iloc[[0, 1, -2, -1], :])

sns.catplot(
    kind="bar",
    data=churn_by_state,
    x="percent",
    y="state",
    height=10,
    aspect=0.5,
)

#%% [markdown]
"""
I wonder if the USA regions would provide more meaningful information. Let's try to do this split in the dataset. First we have to classify the states. According to [Wikipedia](https://upload.wikimedia.org/wikipedia/commons/f/f1/Census_Regions_and_Division_of_the_United_States.svg), we can do it as follows:
"""

#%%
regions = {
    "AK": "west",
    "HI": "west",
    "WA": "west",
    "OR": "west",
    "CA": "west",
    "ID": "west",
    "MT": "west",
    "WY": "west",
    "NV": "west",
    "UT": "west",
    "CO": "west",
    "AZ": "west",
    "NM": "west",
    "ND": "midwest",
    "SD": "midwest",
    "NE": "midwest",
    "KS": "midwest",
    "MN": "midwest",
    "IA": "midwest",
    "MO": "midwest",
    "WI": "midwest",
    "IL": "midwest",
    "MI": "midwest",
    "IN": "midwest",
    "OH": "midwest",
    "TX": "south",
    "OK": "south",
    "AR": "south",
    "LA": "south",
    "MS": "south",
    "TN": "south",
    "AL": "south",
    "KY": "south",
    "GA": "south",
    "FL": "south",
    "SC": "south",
    "NC": "south",
    "VA": "south",
    "DC": "south",
    "WV": "south",
    "MD": "south",
    "DE": "south",
    "PA": "northeast",
    "NJ": "northeast",
    "NY": "northeast",
    "VT": "northeast",
    "NH": "northeast",
    "ME": "northeast",
    "MA": "northeast",
    "CT": "northeast",
    "RI": "northeast",
}

data["region"] = data["state"].map(regions)

#%% [markdown]
"""
As we can see, the Northeast region does indeed present a little more churn than the other regions, but it's hard to tell if that's meaningful as of now.
"""

#%%
churn_by_region = eda_tools.group_by_feature(
    data, "region", churn="yes", normalize=True
)

sns.catplot(
    kind="bar",
    data=churn_by_region,
    x="region",
    y="percent",
)

#%% [markdown]
"""
The "account_length" feature is a bit more mysterious. The name is not self explanatory and I honestly can't tell what it means by looking at its values. Since we don't have great documentation, we'll have to go blind on this one.
"""

#%%
print(
    f"Number of unique account lengths: {data['account_length'].nunique()}\n",
    f"Account lengths sorted by number of occurrences:\n{data['account_length'].value_counts()}",
    sep="\n",
)

churn_by_account_length = eda_tools.group_by_feature(data, "account_length")

#%% [markdown]
"""
Looking at the histogram we see both churn and no-churn show very similar distributions.
"""

#%%
sns.displot(
    kind="hist",
    data=data,
    x="account_length",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%% [markdown]
"""
The next feature is "area_code," which definitely relates to the phones area codes. There are only three distinct values.
"""

#%%
print(
    f"Area codes:\n{data['area_code'].value_counts()}",
)

#%% [markdown]
"""
We can clean it up a little bit by removing the redundant part in the values and leaving only the numbers.
"""

#%%
data["area_code"] = data["area_code"].map(lambda code: code[-3:])
print(
    f"Area codes:\n{data['area_code'].value_counts()}",
)

#%% [markdown]
"""
The area code also doesn't show any direct relationship to the costumer churn.
"""

#%%
churn_by_area_code = eda_tools.group_by_feature(data, "area_code", normalize=True)

#%%
sns.catplot(
    kind="bar",
    data=churn_by_area_code,
    x="area_code",
    y="percent",
    hue="churn",
)

#%% [markdown]
"""
The international plan feature is a much more interesting one. We can clearly see from the graph the costumer that _do have_ an international plan tend to churn a lot more. Maybe that's because the plan is not worth it, so they tend to give it up.
"""

#%%
data["international_plan"].value_counts()

#%%
churn_by_int_plan = eda_tools.group_by_feature(
    data, "international_plan", normalize=True
)

#%%
sns.catplot(
    kind="bar",
    data=churn_by_int_plan,
    x="international_plan",
    y="percent",
    hue="churn",
)

#%% [markdown]
"""
But on the other hand, the users that have the voice mail plan tend to churn a bit less.
"""

#%%
data["voice_mail_plan"].value_counts()

#%%
churn_by_voice_plan = eda_tools.group_by_feature(
    data, "voice_mail_plan", normalize=True
)

#%%
sns.catplot(
    kind="bar",
    data=churn_by_voice_plan,
    x="voice_mail_plan",
    y="percent",
    hue="churn",
)

#%% [markdown]
"""
With respect to the number of vmail (voice mail?) messages, we can see that users who have a low number (about 0) churn over 10% more than those who have upwards of 10 messages.
"""

#%%
data["number_vmail_messages"].value_counts()

sns.displot(
    kind="hist",
    data=data,
    x="number_vmail_messages",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%% [markdown]
"""
Interestingly, the amount of total daily minutes spent presents a _bimodal_ distribution for the churn class. Around its second peak there's definitely more churn than no-churn.
"""

#%%
sns.displot(
    kind="hist",
    data=data,
    x="total_day_minutes",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%% [markdown]
"""
Nothing particularly interesting about the total daily calls.
"""

#%%
sns.displot(
    kind="hist",
    data=data,
    x="total_day_calls",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%% [markdown]
"""
Total daily charge is not surprisingly very similar to daily minutes.
"""

#%%
sns.displot(
    kind="hist",
    data=data,
    x="total_day_charge",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%% [markdown]
"""
The total eve and night numbers are also not very enlightening.
"""

#%%
sns.displot(
    kind="hist",
    data=data,
    x="total_eve_minutes",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%%
sns.displot(
    kind="hist",
    data=data,
    x="total_eve_calls",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%%
sns.displot(
    kind="hist",
    data=data,
    x="total_eve_charge",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%%
sns.displot(
    kind="hist",
    data=data,
    x="total_night_minutes",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%%
sns.displot(
    kind="hist",
    data=data,
    x="total_night_calls",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%%
sns.displot(
    kind="hist",
    data=data,
    x="total_night_charge",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%% [markdown]
"""
The churn class seems to be somewhat more concentrated on a lower number of international calls, but spending a bit more total minutes on them.
"""

#%%
sns.displot(
    kind="hist",
    data=data,
    x="total_intl_minutes",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%%
sns.displot(
    kind="hist",
    data=data,
    x="total_intl_calls",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%%
sns.displot(
    kind="hist",
    data=data,
    x="total_intl_charge",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%% [markdown]
"""
The dissatisfied customers clearly tend to make more customer service calls.
"""

#%%
sns.displot(
    kind="hist",
    data=data,
    x="number_customer_service_calls",
    hue="churn",
    cumulative=True,
    stat="probability",
    common_norm=False,
    element="step",
)

#%%
sns.displot(
    kind="hist",
    data=data,
    x="total_day_minutes",
    hue="churn",
    stat="probability",
    common_norm=False,
    element="step",
)

#%%
sns.jointplot(
    kind="hist",
    data=data,
    x="total_day_minutes",
    y="total_day_charge",
    hue="churn",
)

#%%
sns.displot(
    kind="hist",
    data=data,
    y="state",
    hue="churn",
    stat="probability",
    multiple="fill",
    shrink=0.6,
    height=10,
    aspect=0.5,
)
