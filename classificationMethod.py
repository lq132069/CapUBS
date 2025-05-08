import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import scipy.io
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import numpy as np


def filter_classify_report(classify_report):
    """
    Filters the classification report to keep only 'label' and 'precision'.

    Args:
        classify_report (dict): The classification report dictionary.

    Returns:
        str: A string containing the filtered classification report.
    """
    # Keep only 'label' and 'precision', remove other metrics
    filtered_report = {}

    # Iterate over the classification report items
    for label, metrics in classify_report.items():
        if label != 'accuracy':  # Skip accuracy as it's a single value
            # Keep only the 'precision' for each label
            filtered_report[label] = {'precision': round(metrics['precision'] * 100, 2)}

    # Convert filtered report to string
    report_str = ""

    # Convert the filtered report into string format
    for label, metrics in filtered_report.items():
        report_str += f"{label} -> precision: {metrics['precision']}%\n"

    return report_str


def calculate_metrics(y_true, y_pred):
    """
    Calculates the confusion matrix, Overall Accuracy (OA), Average Accuracy (AA), and Kappa coefficient.

    Args:
        y_true (np.array): The ground truth labels.
        y_pred (np.array): The predicted labels.

    Returns:
        tuple: A tuple containing OA, AA, and Kappa.
    """
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Overall Accuracy (OA)
    oa = accuracy_score(y_true, y_pred)

    # Average Accuracy (AA)
    aa = np.mean(np.diag(conf_matrix) / np.sum(conf_matrix, axis=1))

    # Kappa coefficient
    kappa = cohen_kappa_score(y_true, y_pred)

    # Round the values to two decimal places
    oa = round(oa * 100, 2)
    aa = round(aa * 100, 2)
    kappa = round(kappa * 100, 2)
    return oa, aa, kappa



def classify_SVM(data, label):
    """
    Classifies the entire image using Support Vector Machine and returns the classification results.

    Args:
        data (np.array): The hyperspectral image data (H, W, C).
        label (np.array): The ground truth label map (H, W).

    Returns:
        tuple: A tuple containing the filtered classification report (str),
               Overall Accuracy (float), Average Accuracy (float), Kappa coefficient (float),
               and the predicted label image (np.array).
    """
    # Flatten the data and labels to 2D
    X = data.reshape(-1, data.shape[2])
    y = label.flatten()

    # Remove background pixels (i.e., pixels with label 0)
    X = X[y > 0]
    y = y[y > 0]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, stratify=y)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Support Vector Machine classifier
    svm_clf = SVC(kernel='linear', random_state=42)
    svm_clf.fit(X_train, y_train)

    # Standardize and predict on the entire dataset (including background)
    X_full = data.reshape(-1, data.shape[2])  # Original full data
    X_full_scaled = scaler.transform(X_full)
    y_pred_full = np.zeros_like(label.flatten())  # Initialize with background label 0
    y_pred_full[label.flatten() > 0] = svm_clf.predict(X_full_scaled[label.flatten() > 0])

    # Reshape the predicted labels back to the original image shape
    y_pred_image = y_pred_full.reshape(label.shape)

    # Calculate evaluation metrics (on the test set only)
    classify_report = classification_report(y_test, svm_clf.predict(X_test), output_dict=True)
    classify_report = filter_classify_report(classify_report)
    OA, AA, Kappa = calculate_metrics(y_test, svm_clf.predict(X_test))

    return classify_report, OA, AA, Kappa, y_pred_image

