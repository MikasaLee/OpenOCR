from .rec_metric import RecMetric


class RecTextIDSMetric(object):
    """
    Metric for TextIDS model with dual branches: text branch and IDS branch.
    The text branch results are used as main metrics (acc, norm_edit_dis),
    and IDS branch results are used as auxiliary metrics (ids_acc, ids_norm_edit_dis).
    """

    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=False,
                 stream=False,
                 with_ratio=False,
                 max_len=25,
                 max_len_ids=None,
                 max_ratio=4,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        
        # Use separate max_len for text and IDS branches if specified
        self.max_len_text = max_len
        self.max_len_ids = max_len_ids if max_len_ids is not None else max_len
        
        self.text_metric = RecMetric(main_indicator=main_indicator,
                                     is_filter=is_filter,
                                     ignore_space=ignore_space,
                                     stream=stream,
                                     with_ratio=with_ratio,
                                     max_len=self.max_len_text,
                                     max_ratio=max_ratio)
        self.ids_metric = RecMetric(main_indicator=main_indicator,
                                    is_filter=is_filter,
                                    # IDS contains structural spaces, we might not want to ignore them by default
                                    ignore_space=ignore_space, 
                                    stream=stream,
                                    with_ratio=with_ratio,
                                    max_len=self.max_len_ids,
                                    max_ratio=max_ratio)

    def __call__(self,
                 pred_label,
                 batch=None,
                 training=False,
                 *args,
                 **kwargs):

        # pred_label[0] is text branch, pred_label[1] is IDS branch
        text_metric = self.text_metric(pred_label[0], batch, training=training)
        ids_metric = self.ids_metric(pred_label[1], batch, training=training)
        
        # Set text branch as main metric (acc, norm_edit_dis)
        # IDS branch as auxiliary metric (ids_acc, ids_norm_edit_dis)
        text_metric['ids_acc'] = ids_metric['acc']
        text_metric['ids_norm_edit_dis'] = ids_metric['norm_edit_dis']
        return text_metric

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
                 'ids_acc': 0,
                 'ids_norm_edit_dis': 0,
            }
        """
        text_metric = self.text_metric.get_metric()
        ids_metric = self.ids_metric.get_metric()
        
        # Set text branch as main metric (acc, norm_edit_dis)
        # IDS branch as auxiliary metric (ids_acc, ids_norm_edit_dis)
        text_metric['ids_acc'] = ids_metric['acc']
        text_metric['ids_norm_edit_dis'] = ids_metric['norm_edit_dis']
        return text_metric
if __name__ == "__main__":
    print("Initializing RecTextIDSMetric Check...")
    
    try:
        # Check if RecMetric handles (preds, labels) format
        metric = RecTextIDSMetric(main_indicator='acc', ignore_space=True)
        
        # Simulated Output from PostProcess
        # Text: 1 correct, 1 wrong
        text_preds = [("hello", 0.99), ("world", 0.8)]
        # RecMetric expects labels to be (target, some_other_info) tuple unless tweaked?
        # looking at RecMetric: for (pred, pred_conf), (target, _) in zip(preds, labels):
        # so target must be a tuple too?
        # Let's check RecMetric implementation again... it expects tuple unpacking.
        text_labels = [("hello", None), ("word", None)] 
        
        # IDS: 2 correct
        ids_preds = [("H", 0.99), ("W", 0.99)] 
        ids_labels = [("H", None), ("W", None)]
        
        pred_label_input = [
            (text_preds, text_labels),
            (ids_preds, ids_labels)
        ]
        
        metric(pred_label_input)
        
        final_metric = metric.get_metric()
        
        print("Final Metrics:", final_metric)
        
        # Check standard accuracy
        # Text Acc: 1/2 = 0.5
        # IDS Acc: 2/2 = 1.0
        
        epsilon = 1e-6
        assert abs(final_metric['acc'] - 0.5) < epsilon, f"Expected acc 0.5, got {final_metric['acc']}"
        assert abs(final_metric['ids_acc'] - 1.0) < epsilon, f"Expected ids_acc 1.0, got {final_metric['ids_acc']}"
        
        print("RecTextIDSMetric Check Passed!")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Check Failed: {e}")
