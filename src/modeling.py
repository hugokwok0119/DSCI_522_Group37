from sklearn.svm import SVC

def create_svm_model(kernel="rbf"):
    """
    Create and return an SVM classifier.
    Parameters
    ----------
    kernel : str
        Kernel type for SVM.
    Returns
    -------
    sklearn.svm.SVC
        Unfitted SVM model.
    """
    return SVC(kernel=kernel)
