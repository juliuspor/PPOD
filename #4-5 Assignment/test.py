import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import prince
# Load the dataset
df = pd.read_csv('police_shooting_anonymized.csv')


# Initialize FAMD object: specifying the number of components (n_components)
famd = prince.FAMD(
    n_components=2,
    n_iter=3,
    copy=True,
    check_input=True,
    random_state=42,
    engine="sklearn",
    handle_unknown="error"  # same parameter as sklearn.preprocessing.OneHotEncoder
)
# Fit FAMD on the dataset
famd = famd.fit(df)
print(famd.eigenvalues_summary)
print(famd.row_coordinates(df).head())
print(famd.column_coordinates_)

famd.plot(
    df,
    x_component=0,
    y_component=1
)