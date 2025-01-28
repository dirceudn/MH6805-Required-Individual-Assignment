import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# MH6805: Individual Assignment
# Dirceu de Medeiros Teixeira

#Q4.

# Iris file
input_file = "https://raw.githubusercontent.com/dirceudn/MH6805-Required-Individual-Assignment/refs/heads/main/Iris.csv"
new_output_file = 'newiris.csv'

# Caravan file
caravan = "https://raw.githubusercontent.com/dirceudn/MH6805-Required-Individual-Assignment/refs/heads/main/Caravan.csv"
caravan_scaled = "caravan_scaled.csv"


def data_frame(url):
    """
    Reads a CSV file from the specified path and returns it as a Pandas DataFrame.

    Parameters:
        url (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    df = pd.read_csv(url)
    return df

def pre_processing_data_set():
    """
    a)
    Delete the column containing the species type and use “newiris” for this
    new dataset.

    Performs preprocessing on the dataset by dropping the 'Species' column and saving the cleaned data to a new CSV file.

    Global Variables Used:
        input_file (str): The path to the original dataset.
        new_output_file (str): The path to save the cleaned dataset.

    Returns:
        None: Prints status messages and writes to a new file.
    """
    print("Start pre-processing ...")
    df = data_frame(input_file)
    df.drop(['Species'], axis='columns', inplace=True)
    df.to_csv(new_output_file, index=False)
    print("Pre-processing completed. Cleaned data saved to:", new_output_file)


def get_new_iris_scaler():
    """
    Loads the cleaned dataset, removes an unnamed index column if present, and scales the data using StandardScaler.

    Global Variables Used:
        new_output_file (str): The path to the cleaned dataset.

    Returns:
        numpy.ndarray: The standardized dataset.
    """
    df = data_frame(new_output_file)
    new_iris = df.drop(columns=['Unnamed: 0'], errors='ignore')
    scaler = StandardScaler()
    new_iris_scaled = scaler.fit_transform(new_iris)
    return new_iris_scaled

def get_irpc_value_after_scale():
    """
    Performs Principal Component Analysis (PCA) on the scaled dataset and returns both the transformed PCA DataFrame and raw PCA values.

    Returns:
        pd.DataFrame: A DataFrame containing the transformed PCA values.
        numpy.ndarray: The raw PCA transformed array.
    """
    new_iris_scaled = get_new_iris_scaler()
    pca = PCA()
    irpc = pca.fit_transform(new_iris_scaled)
    irpc_df = pd.DataFrame(irpc, columns=[f'PC{i + 1}' for i in range(irpc.shape[1])])
    return irpc_df, irpc

def get_eigenvalues_value_after_processing():
    """
    Computes the eigenvalues and eigenvectors of the correlation matrix of the scaled dataset.

    Returns:
        numpy.ndarray: The eigenvalues of the correlation matrix.
        pd.DataFrame: The DataFrame containing eigenvectors.
        pd.DataFrame: The DataFrame containing eigenvalues.
    """
    new_iris_scaled = get_new_iris_scaler()
    cor_matrix = np.corrcoef(new_iris_scaled, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cor_matrix)
    eigen_df = pd.DataFrame({'Eigenvalue': eigenvalues})
    eigenvectors_df = pd.DataFrame(eigenvectors, columns=[f'PC{i + 1}' for i in range(len(eigenvalues))])
    return eigenvalues,eigenvectors_df,eigen_df



def print_processing_pca():
    """
    Perform Principal Component Analysis (PCA) on the scaled dataset 'newiris'.

    This function performs the following steps:
    1. Retrieves the PCA results after scaling the dataset 'newiris'.
    2. Stores the PCA results in a DataFrame 'irpc_df'.
    3. Prints the PCA results to see the output.

    Returns:
        None
    """
    irpc_df, _ = get_irpc_value_after_scale()
    print(irpc_df)

def print_eigenvectors_output():
    """
    Print the eigenvalues and eigenvectors after processing.

    This function performs the following steps:
    1. Retrieves the eigenvalues and eigenvectors after processing.
    2. Stores the eigenvalues in a DataFrame 'eigen_df'.
    3. Stores the eigenvectors in a DataFrame 'eigenvectors_df'.
    4. Prints the eigenvalues and eigenvectors to see the output.

    Returns:
        None
    """
    _, eigenvectors_df, eigen_df = get_eigenvalues_value_after_processing()
    print(eigen_df)
    print(eigenvectors_df)



def plot_analysis():

    """
    Visualize the results of Principal Component Analysis (PCA).

    This function performs the following steps:
    1. Retrieves the scaled Iris dataset and the eigenvalues/eigenvectors after processing.
    2. Calculates the explained variance ratio for each principal component.
    3. Plots the explained variance ratio (Scree Plot).
    4. Plots the PCA biplot showing the PCA scores and loadings.

    The Scree Plot helps determine the number of principal components to retain by showing
    the explained variance ratio for each component. The PCA biplot visualizes the
    transformed data points (PCA scores) and the contribution of original features
    to the first two principal components (PC1 and PC2).

    Returns:
        None
    """

    irpc_df,_ = get_irpc_value_after_scale()
    eigenvalues, eigenvectors_df, eigen_df = get_eigenvalues_value_after_processing()

    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='-',
             label="Explained Variance")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot of PCA Eigenvalues")
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.grid(True)
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(irpc_df["PC1"], irpc_df["PC2"], alpha=0.5, label="PCA Scores")

    for i, (pc1, pc2) in enumerate(zip(eigenvectors_df["PC1"], eigenvectors_df["PC2"])):
        ax.arrow(0, 0, pc1, pc2, color='r', alpha=0.7, head_width=0.05, label="Loadings" if i == 0 else "")

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Biplot")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.legend()
    plt.show()

def calculate_variance_proportion():
    """
    Understanding PCA Formulas for PC1 and PC2

    When we do PCA, we transform the original features (X1, X2, X3, X4) into a new set of uncorrelated variables called Principal Components (PCs). These components are just linear combinations of the original features, each weighted by certain coefficients.

    So, let's break down the formulas:

    1. Understanding PC1 (First Principal Component)

    PC1 = 0.5211 * X1 + (-0.2693) * X2 + 0.5804 * X3 + 0.5649 * X4


    In our case we know that PC1 is a mix of the original features (X1, X2, X3, X4), where each feature contributes differently and the coefficients (loadings) show how important each feature is in PC1.
    In general, features with bigger absolute coefficients have a bigger impact on PC1.

    Interpreting the Coefficients below:

    X1 (Sepal Length): 0.5211 → Has a strong positive impact.
    X2 (Sepal Width): -0.2693 → Has a negative impact.
    X3 (Petal Length): 0.5804 → Has the biggest positive impact.
    X4 (Petal Width): 0.5649 → Also has a strong positive impact.

    About PC1:

    With Petal Length (X3) and Petal Width (X4) have the highest positive coefficients, PC1 mostly captures variations in petal size.
    A higher PC1 score means the flower has bigger petals, while a lower score means smaller petals.
    Since Sepal Width (X2) has a negative loading, an increase in Sepal Width decreases the PC1 value.

    About PC2:

    PC2 = (-0.3774) * X1 + (-0.9233) * X2 + (-0.0245) * X3 + (-0.0669) * X4

    What is the conclusion?

    PC2 is another mix of the original features and captures the second most important pattern of variance in the data.

    Interpreting the Coefficients below:

    X1 (Sepal Length): -0.3774 → Has a moderate negative impact.
    X2 (Sepal Width): -0.9233 → Has the strongest negative impact.
    X3 (Petal Length): -0.0245 → Has a very small negative impact.
    X4 (Petal Width): -0.0669 → Also has a small negative impact.

    To sum up the PC representation, we have:

    Sepal Width (X2) dominates PC2 with the largest negative coefficient (-0.9233).
    Since Sepal Length (X1) also has a moderate negative contribution, PC2 mostly captures variations in sepal size.
    A higher PC2 score means the flower has a narrower sepal, while a lower score means a wider sepal.
    Unlike PC1, which was mostly about petal size, PC2 separates flowers based on sepal width.

    Calculate the variance proportion explained by the first two principal components.

    This function performs the following steps:
    1. Scales the new Iris dataset.
    2. Computes the correlation matrix of the scaled data.
    3. Calculates the eigenvalues and eigenvectors of the correlation matrix.
    4. Constructs the formulas for the first two principal components (PC1 and PC2).
    5. Calculates the total variance and the proportion of variance explained by PC1 and PC2.

    Returns:
        tuple: A tuple containing:
            - pc1_formula (str): The formula for the first principal component.
            - pc2_formula (str): The formula for the second principal component.
            - pc1_variance_ratio (float): The proportion of variance explained by PC1.
            - pc2_variance_ratio (float): The proportion of variance explained by PC2.
    """
    new_iris_scaled = get_new_iris_scaler()

    cor_matrix = np.corrcoef(new_iris_scaled, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cor_matrix)
    pc1_formula = "PC1 = " + " + ".join(f"{coef:.4f} * X{i + 1}" for i, coef in enumerate(eigenvectors[:, 0]))
    pc2_formula = "PC2 = " + " + ".join(f"{coef:.4f} * X{i + 1}" for i, coef in enumerate(eigenvectors[:, 1]))
    total_variance = np.sum(eigenvalues)
    pc1_variance_ratio = eigenvalues[0] / total_variance
    pc2_variance_ratio = eigenvalues[1] / total_variance

    return pc1_formula, pc2_formula, pc1_variance_ratio, pc2_variance_ratio

def plot_pca_scatter_pc1_x_pc2():
    """
    This visualization helps us to understand how the data points are distributed along the first two principal components.

    If the dataset contains distinct clusters, they should be visible here.
    The x-axis (PC1) mainly represents variations in petal size.
    The y-axis (PC2) mainly represents variations in sepal width.

    Plot a scatter plot of the first two principal components (PC1 and PC2).

    This function performs the following steps:
    1. Retrieves the PCA results after scaling the dataset 'newiris'.
    2. Creates a scatter plot of PC1 vs. PC2.
    3. Adds labels, a title, grid lines, and axis lines for better visualization.
    4. Displays the plot.

    Returns:
        None
    """
    irpc_df, _ = get_irpc_value_after_scale()
    plt.figure(figsize=(8, 6))
    plt.scatter(irpc_df["PC1"], irpc_df["PC2"], alpha=0.7, edgecolors="k")

    plt.xlabel("Principal Component 1 (PC1)")
    plt.ylabel("Principal Component 2 (PC2)")
    plt.title("PCA Scatter Plot: PC1 vs. PC2")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.7)
    plt.grid(True)
    plt.show()

#Q5.

def pre_processing_caravan_data():

    """
    In our case scaling is necessary because the first variable has a much larger variance than the second.
    Without scaling, the K-Nearest Neighbors (KNN) method would be dominated by the variable with a larger variance,
    which can lead to biased results.

    The recommendation here is use the  Standardization (Z-score normalization)

    Preprocess the Caravan dataset by scaling the features.

    This function performs the following steps:
    1. Loads the Caravan dataset into a DataFrame.
    2. Separates the features (X) and the target variable (y).
    3. Applies Standardization (Z-score normalization) to the features to ensure that each feature has a mean of 0 and a standard deviation of 1.
    4. Creates a new DataFrame with the scaled features and the original target variable.
    5. Saves the scaled DataFrame to a CSV file.
    6. Prints the scaled DataFrame.

    Notes:
    - Scaling is necessary because the first variable has a much larger variance than the second.
    - Without scaling, the K-Nearest Neighbors (KNN) method would be dominated by the variable with a larger variance, leading to biased results.

    Returns:
        None

    """

    df = data_frame(caravan)
    X = df.iloc[:, :-1]

    y = df.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    df_scaled['Purchase'] = y
    scaled_file_path = caravan_scaled

    df_scaled.to_csv(scaled_file_path, index=False)
    print(df_scaled)


def get_labels():
    """
    Split the Caravan dataset into training and testing sets.

    This function performs the following steps:
    1. Loads the Caravan dataset into a DataFrame.
    2. Splits the dataset into training and testing sets:
        - The first 1000 rows are used for testing.
        - The remaining rows are used for training.
    3. Separates the features (X) and the target variable (y) for both training and testing sets.
    4. Returns the training data, testing data, training labels, and testing labels.

    Returns:
        tuple: A tuple containing:
            - train_data (DataFrame): The training features.
            - test_data (DataFrame): The testing features.
            - train_labels (Series): The training labels.
            - test_labels (Series): The testing labels.
    """
    df = data_frame(caravan)

    train_data = df.iloc[1000:, :-1]
    test_data = df.iloc[:1000, :-1]

    train_labels = df.iloc[1000:, -1]
    test_labels = df.iloc[:1000, -1]
    return train_data, test_data, train_labels, test_labels

def scale_training_data():
    """
    Scale the training and testing data using StandardScaler.

    This function performs the following steps:
    1. Retrieves the training and testing data along with their labels using the `get_labels` function.
    2. Applies StandardScaler to scale the training and testing data.
    3. Creates DataFrames for the scaled training and testing data, including the target labels.
    4. Saves the scaled training and testing DataFrames to CSV files.
    5. Returns the scaled training and testing data as well as the DataFrames.

    Returns:
        tuple: A tuple containing:
            - train_data_scaled (ndarray): The scaled training features.
            - test_data_scaled (ndarray): The scaled testing features.
            - train_df_scaled (DataFrame): The scaled training DataFrame with labels.
            - test_df_scaled (DataFrame): The scaled testing DataFrame with labels.
    """

    train_data, test_data, train_labels, test_labels = get_labels()

    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    train_df_scaled = pd.DataFrame(train_data_scaled, columns=train_data.columns)
    test_df_scaled = pd.DataFrame(test_data_scaled, columns=test_data.columns)
    train_df_scaled['Purchase'] = train_labels.values
    test_df_scaled['Purchase'] = test_labels.values


    train_scaled_file_path = "caravan_train_scaled.csv"
    test_scaled_file_path = "caravan_test_scaled.csv"

    train_df_scaled.to_csv(train_scaled_file_path, index=False)
    test_df_scaled.to_csv(test_scaled_file_path, index=False)
    return train_data_scaled, test_data_scaled,train_df_scaled, test_df_scaled

def apply_knn_with_misclassification():
    random_seed = 98765
    np.random.seed(random_seed)

    train_data, test_data, train_labels, test_labels = get_labels()

    train_data_scaled, test_data_scaled,train_df_scaled, test_df_scaled = scale_training_data()
    y_train = train_labels.map({'Yes': 1, 'No': 0})
    y_test = test_labels.map({'Yes': 1, 'No': 0})

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_data_scaled, y_train)

    y_pred = knn.predict(test_data_scaled)

    misclassification_error = 1 - accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    conf_matrix_df = pd.DataFrame(conf_matrix,index=['Actual No', 'Actual Yes'],columns=['Predicted No', 'Predicted Yes'])


    misclassification_error_value = np.round(misclassification_error, 4)

    print(conf_matrix_df)
    print("Misclassification Error Value:", misclassification_error_value)

def get_optimal_k_with_randon_seed(random_seed):
    """
    Apply K-Nearest Neighbors (KNN) classifier and calculate the misclassification error.

    This function performs the following steps:

    1. Retrieves the training and testing data along with their labels using the `get_labels` function.
    2. Scales the training and testing data using the `scale_training_data` function.
    3. Maps the labels 'Yes' and 'No' to binary values (1 and 0).
    4. Initializes and trains a KNN classifier with 3 neighbors.
    5. Predicts the labels for the testing data.
    6. Calculates the misclassification error and the confusion matrix.
    7. Creates a DataFrame for the confusion matrix for better visualization.
    8. Prints the confusion matrix and the misclassification error.

    Returns:
        None
    """

    train_data, test_data, train_labels, test_labels = get_labels()
    train_data_scaled, test_data_scaled,train_df_scaled, test_df_scaled = scale_training_data()

    y_train = train_labels.map({'Yes': 1, 'No': 0})

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(train_data_scaled, y_train,
                                                              test_size=0.2, random_state=random_seed)
    k_values = list(range(3, 21))
    validation_errors = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_sub, y_train_sub)
        y_val_pred = knn.predict(X_val)
        validation_error = 1 - accuracy_score(y_val, y_val_pred)
        validation_errors.append(validation_error)


    optimal_k = k_values[np.argmin(validation_errors)]
    validation_results_df = pd.DataFrame({'K': k_values, 'Validation Error': validation_errors})

    return optimal_k, validation_results_df

def  apply_minus_one_metric_compare_results():
    """
    Compare the optimal K values and validation results using two different random seeds.

    This function performs the following steps:
    1. Sets the original seed and a new seed (original seed - 1).
    2. Retrieves the optimal K value and validation results for both seeds using the `get_optimal_k_with_randon_seed` function.
    3. Prints the optimal K values for both seeds.
    4. Prints the validation results for both seeds.

    Returns:
        None
    """
    original_seed = 12345
    new_seed = original_seed - 1

    optimal_k_original, validation_results_original = get_optimal_k_with_randon_seed(original_seed)
    optimal_k_new, validation_results_new = get_optimal_k_with_randon_seed(new_seed)

    print("Optimal K original seed:", optimal_k_original)
    print("Optimal K new seed (matric number - 1):", optimal_k_new)

    print("\nValidation results with original seed:")
    print(validation_results_original)
    print("\nValidation results with new seed:")
    print(validation_results_new)

def apply_five_fold_cross_validation():
    """
    Apply 5-fold cross-validation to determine the optimal K value for KNN.

    This function performs the following steps:

    1. Retrieves the training and testing data along with their labels using the `get_labels` function.
    2. Scales the training data using StandardScaler.
    3. Maps the training labels 'Yes' and 'No' to binary values (1 and 0).
    4. Defines a range of K values to test (from 3 to 20).
    5. Performs 5-fold cross-validation for each K value and calculates the cross-validation error.
    6. Determines the optimal K value with the lowest cross-validation error.
    7. Creates a DataFrame to store the cross-validation results.
    8. Prints the optimal K value and the cross-validation results.

    Returns:
        None
    """

    train_data, test_data, train_labels, test_labels = get_labels()

    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)

    y_train = train_labels.map({'Yes': 1, 'No': 0})


    k_values = list(range(3, 21))
    cross_val_errors = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, train_data_scaled, y_train, cv=5, scoring='accuracy')
        cross_val_error = 1 - np.mean(scores)
        cross_val_errors.append(cross_val_error)

    optimal_k = k_values[np.argmin(cross_val_errors)]

    cross_val_results_df = pd.DataFrame({'K': k_values, 'Cross-Validation Error': cross_val_errors})

    print("Optimal K using 5-fold cross-validation:", optimal_k)
    print("\nCross-validation results:")
    print(cross_val_results_df)

if __name__ == '__main__':
    pre_processing_data_set()
    print_processing_pca()
    print_eigenvectors_output()
    plot_analysis()
    print(calculate_variance_proportion())
    plot_pca_scatter_pc1_x_pc2()
    pre_processing_caravan_data()
    scale_training_data()
    print(get_optimal_k_with_randon_seed(12345))
    apply_knn_with_misclassification()
    apply_minus_one_metric_compare_results()
    apply_five_fold_cross_validation()