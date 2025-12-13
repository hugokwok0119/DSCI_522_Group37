from sklearn.svm import SVC
from src.modeling import create_svm_model

def test_create_svm_model_returns_svc():
    model = create_svm_model()
    assert isinstance(model, SVC)

def test_create_svm_model_kernel():
    model = create_svm_model(kernel="linear")
    assert model.kernel == "linear"
