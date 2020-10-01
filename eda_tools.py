from pandas import DataFrame


def group_by_feature(data, feature, churn=None, normalize=False):

    churn_by_feature = DataFrame(
        data.groupby(feature)["churn"]
        .value_counts(normalize=normalize)
        .rename("percent" if normalize else "count")
    )

    churn_by_feature = _input_zero_churn(churn_by_feature, feature, normalize=normalize)

    if churn is not None:
        churn_by_feature = churn_by_feature[churn_by_feature["churn"] == churn]

    churn_by_feature = churn_by_feature.sort_values(by=[feature, "churn"]).reset_index(
        drop=True
    )

    return churn_by_feature


def _input_zero_churn(churn_by_feature, feature, normalize):
    """Some of the values for a given feature won't have both 'yes' and 'no' for the churn. In these cases, the particular value for the churn ('yes' or 'no') won't show up in the final grouped_df_by_feature data frame. This code forces it to appear with a count of 0 instead."""

    rows_present = [row[0] for row in churn_by_feature.iterrows()]

    churn_by_feature = churn_by_feature.reset_index()

    rows_absent = [
        (value, churn_)
        for value in churn_by_feature[feature].unique()
        for churn_ in ["no", "yes"]
        if (value, churn_) not in rows_present
    ]

    rows_absent = DataFrame(
        rows_absent,
        columns=[feature, "churn"],
    )

    rows_absent["percent" if normalize else "count"] = 0

    churn_by_feature = churn_by_feature.append(rows_absent)

    return churn_by_feature
