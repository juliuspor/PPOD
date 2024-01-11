import pandas as pd

from Data import dataProvider

from scipy.stats import ks_2samp


def get_spans(df, partition, scale=None):
    """
    Calculates and returns the spans (range of values) for each column in a specified partition of a dataframe, with
    an option to scale these spans by provided values.

    :param        df: the dataframe for which to calculate the spans
    :param partition: the partition for which to calculate the spans
    :param     scale: if given, the spans of each column will be divided
                      by the value in scale for that column
    :returns        : The spans of all columns in the partition
    """
    spans = {}
    for feature_column in quasi_identifiers:
        if feature_column in categorical:
            span = len(df[feature_column][partition].unique())
        else:
            span = df[feature_column][partition].max() - df[feature_column][partition].min()
        if scale is not None:
            span = span / scale[feature_column]
        spans[feature_column] = span
    return spans


def split(df, partition, column):
    """
    Divides a specified partition of a dataframe into two parts based on the median or unique values of a given column,
    returning a tuple with the indices of these two parts.

    :param        df: The dataframe to split
    :param partition: The partition to split
    :param    column: The column along which to split
    :returns        : A tuple containing a split of the original partition
    """
    dfp = df[column][partition]
    if column in categorical:
        values = dfp.unique()
        lv = set(values[:len(values) // 2])
        rv = set(values[len(values) // 2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)


def is_k_anonymous(df, partition, sensitive_column, k=3):
    """
    Checks if a partition is k-anonymous by comparing its amount of entries with the required (k).

    :param               df: The dataframe on which to check the partition.
    :param        partition: The partition of the dataframe to check.
    :param sensitive_column: The name of the sensitive column
    :param                k: The desired k
    :returns               : True if the partition is valid according to our k-anonymity criteria, False otherwise.
    """
    if len(partition) < k:
        return False
    return True


def partition_dataset(df, feature_columns, sensitive_column, scale, is_valid):
    """
    Partitions a dataframe into valid subsets based on specified feature columns, a sensitive column, and span scales,
    using a validity function to ensure each partition meets certain criteria.

    :param               df: The dataframe to be partitioned.
    :param  feature_columns: A list of column names along which to partition the dataset.
    :param sensitive_column: The name of the sensitive column (to be passed on to the is_valid function)
    :param            scale: The column spans as generated before.
    :param         is_valid: A function that takes a dataframe and a partition and returns True if the partition is valid.
    :returns               : A list of valid partitions that cover the entire dataframe.
    """
    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns], partition, scale)
        for column, span in sorted(spans.items(), key=lambda x: -x[1]):
            lp, rp = split(df, partition, column)
            if not is_valid(df, lp, sensitive_column) or not is_valid(df, rp, sensitive_column):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions


def agg_categorical_column(series):
    """
    Aggregates the values of a series with categorical values by concatenating them.

    :param           series: A series of categorical values that need to be aggregated.
    :returns               : A string with all the values in the series joined with a ',' (comma).
    """
    series = set(series.astype(str))
    return [','.join(series)]


def agg_numerical_column(series):
    """
    Aggregates the values of a series with numerical values by taking their mean.

    :param           series: A series of numerical values that need to be aggregated.
    :returns               : Mean value of the values in the series.
    """
    print(series)
    return [series.mean()]


def build_anonymized_dataset_sensitives_seperated(df, partitions, feature_columns, sensitive_columns,
                                                  max_partitions=None):
    """
    Constructs an anonymized dataset by aggregating feature columns and sensitive columns separately for each partition.

    :param                df: The dataframe to be anonymized.
    :param        partitions: A list of indices for each partition of the dataframe.
    :param   feature_columns: A list of feature column names to aggregate.
    :param sensitive_columns: A list of sensitive column names to aggregate.
    :param    max_partitions: Optional, maximum number of partitions to process.
    :returns                : An anonymized dataframe with aggregated feature and sensitive columns.
    """
    aggregations = {}
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column
    rows = []
    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print("Finished {} partitions...".format(i))
        if max_partitions is not None and i > max_partitions:
            break
        grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
        values = grouped_columns.to_dict()
        # Iterate through each sensitive column and aggregate counts
        for sensitive_column in sensitive_columns:
            sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column: 'count'})
            for sensitive_value, count in sensitive_counts[sensitive_column].items():
                if count == 0:
                    continue
                sensitive_values = values.copy()
                sensitive_values.update({
                    sensitive_column: sensitive_value,
                    'count': count,
                })
                rows.append(sensitive_values)
    return pd.DataFrame(rows)


def build_anonymized_dataset_sensitives_combined(df, partitions, feature_columns, sensitive_columns,
                                                 max_partitions=None):
    """
    Creates an anonymized dataset by aggregating both feature and sensitive columns within each partition, combining their results.

    :param                df: The dataframe to be anonymized.
    :param        partitions: A list of indices for each partition of the dataframe.
    :param   feature_columns: A list of feature column names to aggregate.
    :param sensitive_columns: A list of sensitive column names to aggregate together.
    :param    max_partitions: Optional, maximum number of partitions to process.
    :returns                : An anonymized dataframe with combined aggregated feature and sensitive columns.
    """
    aggregations = {col: (agg_categorical_column if col in categorical else agg_numerical_column)
                    for col in feature_columns}

    rows = []
    for i, partition in enumerate(partitions):
        if max_partitions is not None and i >= max_partitions:
            break
        if i % 100 == 1:
            print("Finished {} partitions...".format(i))

        partition_data = df.loc[partition]
        aggregated_data = partition_data[feature_columns].agg(aggregations)

        # Aggregating sensitive columns
        sensitive_aggregated = {s_col: partition_data[s_col].value_counts().to_dict()
                                for s_col in sensitive_columns}

        # Combining feature, sensitive data, and count
        row_data = {**aggregated_data.to_dict(), **sensitive_aggregated, 'count': len(partition_data)}
        rows.append(row_data)

    return pd.DataFrame(rows)


# t-closeness

def t_closeness_numerical(df, partition, column):
    # Extract the values of the column for the entire dataset and the partition
    full_data = df[column]
    partition_data = df.loc[partition, column]

    # Compute the KS statistic between the partition and the full dataset
    ks_stat, _ = ks_2samp(full_data, partition_data)
    return ks_stat


def t_closeness_categorical(df, partition, column, global_freqs):
    total_count = float(len(partition))
    d_max = None
    group_counts = df.loc[partition].groupby(column, observed=False)[column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count / total_count
        d = abs(p - global_freqs[value])
        if d_max is None or d > d_max:
            d_max = d
    return d_max


def is_t_close(df, partition, sensitive_columns, global_freqs, t=0.3):
    for sensitive_column in sensitive_columns:
        if sensitive_column not in categorical:
            distance = t_closeness_numerical(df, partition, sensitive_column)
        else:
            distance = t_closeness_categorical(df, partition, sensitive_column, global_freqs[sensitive_column])
        if distance > t:
            return False
    return True


# Load initial dataset as dataframe
df = dataProvider.get_joined_athletes_dataset()

# some fields are categorical and will require special treatment
categorical = {'region', 'team', 'affiliate', 'gender', 'eat', 'train', 'background', 'experience', 'schedule',
               'howlong'}

for name in categorical:
    df[name] = df[name].astype('category')

# quasi-identifiers that should be taken into acount
quasi_identifiers = ['age', 'region', 'team', 'affiliate', 'gender', 'height', 'weight', 'eat', 'train', 'background',
                     'experience', 'schedule', 'howlong']

# sensitive-values that should be taken into account
sensitive_columns = ['fran', 'helen', 'grace', 'filthy50', 'fgonebad', 'run400', 'run5k', 'candj', 'snatch', 'deadlift',
                     'backsq', 'pullups']

# Obtain the spans for each column in the dataframe
full_spans = get_spans(df, df.index)

# Generate the global frequencies for the sensitive column
global_frequencies = {sensitive_column: {} for sensitive_column in sensitive_columns}
total_count = len(df)
# Determine frequency for each sensitive value
for sensitive_column in sensitive_columns:
    group_counts = df.groupby(sensitive_column, observed=False)[sensitive_column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count / total_count
        global_frequencies[sensitive_column][value] = p


# Partition dataset based on k-anonymity and t-closeness
finished_t_close_partitions = partition_dataset(
    df, quasi_identifiers, sensitive_columns, full_spans,
    lambda *args: is_k_anonymous(*args) and is_t_close(*args, global_frequencies)
)

# Build anonymized dataset based on partitions
dft = build_anonymized_dataset_sensitives_seperated(df, finished_t_close_partitions, quasi_identifiers,
                                                    sensitive_columns)
