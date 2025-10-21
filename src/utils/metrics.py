from typing import Dict
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def softmax(x, dim=1):
    import torch
    return torch.softmax(x, dim=dim)

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = y_prob.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro')
    f1w = f1_score(y_true, y_pred, average='weighted')
    return {'acc': acc, 'f1_macro': f1m, 'f1_weighted': f1w}

class MetricTracker:
    def __init__(self): self.reset()
    def reset(self):
        self._tr_losses, self._va_losses = [], []
        self._tr_logits, self._tr_labels = [], []
        self._va_logits, self._va_labels = [], []
    def update_train(self, loss, logits, labels):
        self._tr_losses.append(loss); self._tr_logits.append(logits); self._tr_labels.append(labels)
    def update_val(self, loss, logits, labels):
        self._va_losses.append(loss); self._va_logits.append(logits); self._va_labels.append(labels)
    def summarize(self, reset=True):
        import torch as th, numpy as np
        tr_loss = float(np.mean(self._tr_losses)) if self._tr_losses else 0.0
        va_loss = float(np.mean(self._va_losses)) if self._va_losses else 0.0
        def _cat(xs): return th.cat(xs, dim=0) if xs else None
        tr_logits = _cat(self._tr_logits); tr_labels = _cat(self._tr_labels)
        va_logits = _cat(self._va_logits); va_labels = _cat(self._va_labels)
        def _m(logits, labels):
            if logits is None: return {'acc':0.0,'f1':0.0}
            y_pred = logits.argmax(dim=1).numpy()
            y_true = labels.numpy()
            return {'acc': accuracy_score(y_true, y_pred),
                    'f1': f1_score(y_true, y_pred, average='macro')}
        trm = _m(tr_logits, tr_labels); vam = _m(va_logits, va_labels)
        out = {'train_loss': tr_loss, 'val_loss': va_loss, 'train_acc': trm['acc'],
               'train_f1': trm['f1'], 'val_acc': vam['acc'], 'val_f1': vam['f1']}
        if reset: self.reset()
        return out
