
import math
from dataclasses import dataclass, field

import torch
from torch._C import Value
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterionConfig
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss

from fairseq.criterions.label_smoothed_cross_entropy_weighted import WeightedLabelSmoothedCrossEntropyCriterion
import logging


# LabelSmoothedCrossEntropyCriterionConfig

logger = logging.getLogger(__name__)


def symmetric_KL_loss(p, q, pad_mask):
    """ symmetric KL-divergence 1/2*(KL(p||q)+KL(q||p)) """
    p, q, pad_mask = p.float(), q.float(), pad_mask.view(-1)
    dict_size = q.size(-1)
    non_pad_mask = ~pad_mask
    p = p.view(-1, dict_size)[non_pad_mask]
    q = q.view(-1, dict_size)[non_pad_mask]
    loss = (p - q) * (torch.log(p) - torch.log(q))
    return 0.5 * loss.sum()


@dataclass
class ThorWeightedLabelSmoothedCrossEntropyCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    num_experts: int = field(
        default=2,
        metadata={"help": "number of experts"},
    )
    consistency_alpha: float = field(
        default=1.0,
        metadata={"help": "weight of the consistency loss"},
    )
    inference_level: int = field(
        default=0,
        metadata={"help": "0 for token level, 1 for sentence level"},
    )
    

@register_criterion(
    "thor_wlabel_smoothed_cross_entropy", dataclass=ThorWeightedLabelSmoothedCrossEntropyCriterionConfig
)
class ThorWeightedLabelSmoothedCrossEntropyCriterion(WeightedLabelSmoothedCrossEntropyCriterion):
    def __init__(
            self, task, 
            sentence_avg, label_smoothing, 
            num_experts, consistency_alpha=1.0, inference_level=0,
            ignore_prefix_size=0, report_accuracy=False,
        ):
        self.consistency_alpha = consistency_alpha
        self.num_experts = num_experts
        self.inference_level = inference_level
        super().__init__(task, sentence_avg, label_smoothing, 
            ignore_prefix_size=ignore_prefix_size, report_accuracy=report_accuracy)
    
    def _get_loss(self, model, sample, reduce=True, weights=None, get_out=False):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce, weights=weights)
        logits = F.softmax(net_output[0].float(), dim=-1)
        if get_out:
            return loss, logits, nll_loss, net_output
        else:
            return loss, logits, nll_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        weights = sample.get('weights', None)
        if weights is not None:
            assert weights.size(0) == sample["target"].size(0), f'debug strict: {weights.size()=}/{sample["target"].size(0)=} {weights}'
        
        # sample 1
        loss1, logits1, nll_loss1 = self._get_loss(model, sample, reduce, weights)
        
        # sample 2
        loss2, logits2, nll_loss2 = self._get_loss(model, sample, reduce, weights)

        pad_mask = sample["target"].eq(self.padding_idx)
        consistency_loss = symmetric_KL_loss(logits1, logits2, pad_mask)
        loss = loss1 + loss2 + consistency_loss * self.consistency_alpha
        nll_loss = nll_loss1 + nll_loss2

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "consistency": consistency_loss.data,
        }
        if self.report_accuracy:
            # n_correct, total = self.compute_accuracy(model, net_output, sample)
            # logging_output["n_correct"] = utils.item(n_correct.data)
            # logging_output["total"] = utils.item(total.data)
            raise ValueError(f'report_accuracy not supported')
        return loss, sample_size, logging_output
    
    
    @classmethod
    def reduce_metrics(cls, logging_outputs):
        super().reduce_metrics(logging_outputs)

        # follows the reduce_metrics() function in label_smoothed_cross_entropy.py
        loss_sum = sum(log.get("consistency", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "consistency", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

