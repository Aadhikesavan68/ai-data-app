def weighted_average(df, value_col, weight_col):
    return round((df[value_col] * df[weight_col]).sum() / df[weight_col].sum(), 2)
