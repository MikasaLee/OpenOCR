import copy

__all__ = ['build_metric']

from .rec_metric import RecMetric
from .rec_metric_gtc import RecGTCMetric
from .rec_metric_long import RecMetricLong
from .rec_metric_mgp import RecMPGMetric
from .rec_metric_text_ids import RecTextIDSMetric
from .rec_metric_char_verify import RecCharVerifyMetric
from .rec_metric_rtd import RecRTDMetric
from .tan_metric import TANMetric

support_dict = ['RecMetric', 'RecMetricLong', 'RecGTCMetric', 'RecMPGMetric', 'RecTextIDSMetric', 'RecCharVerifyMetric', 'RecRTDMetric', 'TANMetric']


def build_metric(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'metric only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class