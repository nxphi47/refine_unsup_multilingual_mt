
import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterionConfig
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss


@register_criterion(
    "weighted_label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class WeightedLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
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
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce, weights=weights)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output
    
    def get_lprobs_and_target(self, model, net_output, sample, weights=None):
        # lprobs: [bsz, slen, d]
        # target: [bsz, slen, d]
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        assert lprobs.dim() == 3, f'{lprobs.size()}'
        if weights is not None:
            lprobs = lprobs * weights[:, None, None].type_as(lprobs).to(lprobs.device)
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True, weights=None):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample, weights=weights)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss
