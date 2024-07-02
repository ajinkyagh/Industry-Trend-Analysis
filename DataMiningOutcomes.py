import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the CSV file to understand its structure and contents
data = pd.read_csv('organizations-10000.csv')

# Display the first few rows of the dataset
#print(data.head())

# Data Cleaning and Transformation Steps
# Check for missing values
missing_values = data.isnull().sum()

# Check for duplicate rows
duplicate_rows = data.duplicated().sum()

# Standardize and clean the 'Country' and 'Website' columns for consistency
data['Country'] = data['Country'].str.title()  # Capitalize country names
data['Website'] = data['Website'].str.lower()  # Ensure website URLs are in lowercase

# Convert profile_info dictionary to a DataFrame for better readability
profile_info_df = pd.DataFrame({
    'Column': data.columns,
    'Non-Null Count': data.notnull().sum(),
    'Data Type': data.dtypes,
    'Unique Values': data.nunique(),
    'Missing Values': missing_values,
})

# Removing duplicate rows if any
data_cleaned = data.drop_duplicates()

# Summary of changes
changes_summary = {
    'Original Rows': len(data),
    'Cleaned Rows': len(data_cleaned),
    'Duplicate Rows Removed': duplicate_rows,
}

#print(profile_info_df)
#print(changes_summary)

# Exploratory Data Analysis (EDA)
numerical_summary = data.describe()
industry_counts = data['Industry'].value_counts()
country_counts = data['Country'].value_counts()

numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()


#2 Data Mining Techniques Application

# Cluster Analysis with KMeans
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Founded', 'Number of employees']])
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(data_scaled)
data['Cluster Labels'] = cluster_labels

# Dimensionality reduction with PCA for visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Classification with Naive Bayes
top_industries = industry_counts.head(5).index
data_top_industries = data[data['Industry'].isin(top_industries)]
X_class = data_top_industries[['Founded', 'Number of employees']]
y_class = data_top_industries['Industry']
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
classifier = MultinomialNB()
classifier.fit(X_train_class, y_train_class)
y_pred_class = classifier.predict(X_test_class)
class_report = classification_report(y_test_class, y_pred_class)
conf_matrix_class = confusion_matrix(y_test_class, y_pred_class)

# Regression with Linear Regression
X_reg = data[['Founded']]
y_reg = data['Number of employees']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
regressor = LinearRegression()
regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = regressor.predict(X_test_reg)
regression_coefficients = regressor.coef_
regression_intercept = regressor.intercept_

# Text Mining with TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Description'])
feature_names = tfidf_vectorizer.get_feature_names_out()
top_terms = feature_names[tfidf_matrix.sum(axis=0).argsort()[0,-20:]].tolist()

# Results Summary
results = {
    'Cluster Analysis': {
        'Clusters Found': len(set(cluster_labels))
    },
    'Classification': {
        'Classification Report': class_report,
        'Confusion Matrix': conf_matrix_class
    },
    'Regression': {
        'Coefficients': regression_coefficients,
        'Intercept': regression_intercept
    },
    'Top Terms from Text Mining': top_terms
}

# Visualization of the clusters
plt.figure(figsize=(10, 8))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data['Cluster Labels'], cmap='viridis', marker='o')
plt.title('Visualization of Clusters')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Print out top terms from text mining
print("Top terms from text mining: ", top_terms)

# Print out the classification report and confusion matrix
print("Classification Report:\n", class_report)
print("Confusion Matrix:\n", conf_matrix_class)

# Print out the regression coefficients
print("Regression Coefficients: ", regression_coefficients)
print("Regression Intercept: ", regression_intercept)


