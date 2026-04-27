import torch


class TANMetric(object):

    def __init__(self, main_indicator='acc', **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.lpred_loss_sum = 0.0
        self.rrepred_loss_sum = 0.0
        self.triplet_loss_sum = 0.0
        self.batch_count = 0
        self.char_correct = 0.0
        self.char_total = 0.0
        self.rel_correct = 0.0
        self.rel_total = 0.0

    @staticmethod
    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach()
        return torch.as_tensor(x)

    def __call__(self, pred_label, batch=None, training=False):
        char_logits = pred_label['char_logits'].detach()
        relation_logits = pred_label['relation_logits'].detach()

        ly = self._to_tensor(batch[2]).to(char_logits.device)
        ly_mask = self._to_tensor(batch[3]).to(char_logits.device)
        rre = self._to_tensor(batch[9]).to(relation_logits.device)
        rre_mask = self._to_tensor(batch[10]).to(relation_logits.device)

        char_pred = char_logits.argmax(dim=-1)
        rel_pred = relation_logits.argmax(dim=-1)

        char_mask = ly_mask > 0
        rel_mask = rre_mask > 0
        batch_char_correct = 0.0
        batch_char_total = 0.0
        batch_rel_correct = 0.0
        batch_rel_total = 0.0
        if char_mask.any():
            batch_char_correct = (char_pred[char_mask] == ly[char_mask]).float().sum().item()
            batch_char_total = char_mask.float().sum().item()
            self.char_correct += batch_char_correct
            self.char_total += batch_char_total
        if rel_mask.any():
            batch_rel_correct = (rel_pred[rel_mask] == rre[rel_mask]).float().sum().item()
            batch_rel_total = rel_mask.float().sum().item()
            self.rel_correct += batch_rel_correct
            self.rel_total += batch_rel_total

        batch_loss = float(pred_label['loss'].detach().item())
        batch_lpred_loss = float(pred_label['lpred_loss'].detach().item())
        batch_rrepred_loss = float(pred_label['rrepred_loss'].detach().item())
        batch_triplet_loss = float(pred_label['triplet_loss'].detach().item())
        self.loss_sum += batch_loss
        self.lpred_loss_sum += batch_lpred_loss
        self.rrepred_loss_sum += batch_rrepred_loss
        self.triplet_loss_sum += batch_triplet_loss
        self.batch_count += 1
        return {
            'acc': batch_char_correct / max(batch_char_total, 1.0),
            'relation_acc': batch_rel_correct / max(batch_rel_total, 1.0),
            'loss': batch_loss,
            'lpred_loss': batch_lpred_loss,
            'rrepred_loss': batch_rrepred_loss,
            'triplet_loss': batch_triplet_loss,
            'score': -batch_loss,
        }

    def get_metric(self):
        batch_count = max(self.batch_count, 1)
        loss = self.loss_sum / batch_count
        metrics = {
            'acc': self.char_correct / max(self.char_total, 1.0),
            'relation_acc': self.rel_correct / max(self.rel_total, 1.0),
            'loss': loss,
            'lpred_loss': self.lpred_loss_sum / batch_count,
            'rrepred_loss': self.rrepred_loss_sum / batch_count,
            'triplet_loss': self.triplet_loss_sum / batch_count,
            'score': -loss,
        }
        self.reset()
        return metrics
