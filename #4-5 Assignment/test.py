import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from prince import FAMD
# Load the dataset
df = pd.read_csv('police_shooting_anonymized.csv')


# Initialize FAMD object: specifying the number of components (n_components)
famd = FAMD(n_components=2, n_iter=3, random_state=42)

# Fit FAMD on the dataset
famd = famd.fit(df)

# Transform the dataset
df_transformed = famd.transform(df)


# Assuming 'df_transformed' is the DataFrame obtained after applying FAMD
z_scores = np.abs(stats.zscore(df_transformed))
outliers = np.where(z_scores > 3)

# 'outliers' now contains the indices of the outliers in the transformed dataset
# Assuming outliers are in rows
outlier_rows = df.iloc[outliers[0]]
outlier_rows.to_csv('outliers.csv', index=False)

# Plotting the transformed dataset
plt.figure(figsize=(10, 7))
plt.scatter(df_transformed[0], df_transformed[1], alpha=0.5)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('FAMD - Reduced Dimensionality Visualization')
plt.show()
