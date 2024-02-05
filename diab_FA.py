
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_diabetes
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data

# Convert the array to a DataFrame
columns = ['Age', 'Sex', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
df_diabetes = pd.DataFrame(X, columns=columns)

# Select relevant columns for factor analysis (numeric variables)
selected_columns = df_diabetes

# Handling missing values (replace NaNs with the mean)
imputer = SimpleImputer(strategy='mean')
selected_columns_imputed = imputer.fit_transform(selected_columns)

# Standardize the data
scaler = StandardScaler()
selected_columns_standardized = scaler.fit_transform(selected_columns_imputed)

# Calculate KMO Measure
kmo_all, kmo_model = calculate_kmo(selected_columns_standardized)
print(f"KMO Measure: {kmo_model}")

# Apply Bartlett's Test
chi_square_value, p_value = calculate_bartlett_sphericity(selected_columns_standardized)
print(f"Chi-Square Value: {chi_square_value}")
print(f"P-value: {p_value}")

# Create a FactorAnalysis pipeline
fa = make_pipeline(FactorAnalysis(n_components=3, random_state=42))

# Fit the FactorAnalysis model to your data
fa.fit(selected_columns_standardized)

# Get the factor loadings
factor_loadings = fa.named_steps['factoranalysis'].components_

# Display factor loadings
print("\nFactor Loadings:")
print(factor_loadings)

# Display factors along with loadings for each variable
print("\nFactors and Loadings:")
for i, variable in enumerate(selected_columns.columns):
    print(f"{variable}:")
    for j, factor in enumerate(factor_loadings):
        print(f"  Factor {j} Loading: {factor[i]:.2f}")
    print()

# Calculate communalities
communalities = np.sum(factor_loadings**2, axis=0)
print("\nCommunalities:")
print(communalities)

# Calculate specific variances
specific_variances = 1 - communalities
print("\nSpecific Variances:")
print(specific_variances)

# Create matrix of specific variances
matrix_specific_variance = np.diag(specific_variances)
print("\nMatrix of Specific Variances:")
print(matrix_specific_variance)

# Compute covariance matrix from factor loading matrix and matrix of specific variances
covariance_matrix = np.dot(factor_loadings.T, factor_loadings) + matrix_specific_variance
print("\nCovariance Matrix:")
print(covariance_matrix)

# Visualize the results (heatmap of factor loadings)
plt.figure(figsize=(10, 6))
sns.heatmap(factor_loadings, annot=True, cmap='coolwarm', xticklabels=selected_columns.columns)
plt.title('Factor Loadings Heatmap')
plt.show()
