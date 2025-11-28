import torch
import logging
import pandas as pd

from lasso_prior.model.decoder import TabPFNFeatureSelector
from high_tab.utils.hardware import log_mem, memory_cleanup

logger = logging.getLogger(__name__)

_TABPFN_MODEL_CACHE = {}

def get_cached_model(model_name, checkpoint_path, embedding_layer, device):
    """
    Get or create a shared TabPFN model (singleton pattern).
    This prevents loading multiple copies of the huge encoder.
    """
   

class TabPFNSelector:
    """
    For preprocessing, use selected_columns_ (similar to random_fs method).
    """
    def __init__(self, num_features, device):
        self.K = int(num_features)
        self.checkpoint_dir = "decoder_models/train_lasso_4000_20251120_231955/"
        self.model_name = "best_model"
        self.selected_cols_ = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self):
        model = TabPFNFeatureSelector(
            model_name="TabPFN-Wide-5k",
            model_checkpoint_dir="external/tabpfnwide/models",
            embedding_layer=4, 
            device=self.device,
        )
        model.load_decoder_checkpoint("external/lasso_checkpoints/best_model.pt")
        model.eval
        return model.to(self.device)


    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        train_size = int(X.shape[0] * 0.8)
        column_names = X.columns.tolist()

        X = torch.Tensor(X.values).unsqueeze(1).to(self.device) 
        y_train = torch.Tensor(y.iloc[:train_size].values).unsqueeze(1).to(self.device)  
        
        with torch.no_grad():
            coefficients = self.model(X, y_train)

        topk_idx = torch.topk(coefficients, self.K).indices.tolist()[0]
        self.selected_cols_ = [column_names[i] for i in topk_idx]

     
        del X, y_train, coefficients, self.model
        self.model = None
        memory_cleanup()

        return self
