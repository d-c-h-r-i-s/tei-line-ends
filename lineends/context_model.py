#!/usr/bin/env python3
"""
Context Model
=============

Which of several readings does *this* sentence want? An interpolated bigram
model over the corpus's own line-interior lexicon, shared by the two stages
that ask that question at a line end:

    resolve_break_context.py     one word or two — "Kaiserin Zita" against
                                 "der Kaiser in Wien"
    resolve_line_end_context.py  which word — a final `n` misread as `r`,
                                 where the misreading is itself a real word

The two are the same arithmetic on the same tables and differ only in what
they put in the middle slot, which is why this lives here rather than in
either of them. `validate_xml_hyphens.build_lexicon` produces the tables;
interior tokens are the right reference for both, being the one position a
line break cannot damage.

The interpolation is the whole design, and it is not a detail of smoothing —
see `BIGRAM_WEIGHT`. Getting it wrong made the first version of the hyphen
model answer "one word" to almost everything, and the free gold standard is
what caught it.

Author: Christian Lendl
Created: 2026-08-30
"""

import math
from collections import Counter
from typing import List, Optional


# Weight on the bigram estimate against the unigram backoff.
#
# The first version of this model smoothed add-one over the whole vocabulary,
# which charged every reading ~12 nats per token before any evidence was
# considered - and the separated reading has one token more. That is a fixed
# thumb on the scale worth far more than anything the sentence could say, and
# it showed: join recall 1.00, split recall 0.00, balanced accuracy 0.557
# against a majority-class baseline of 0.718. The model was not reading the
# sentence, it was paying a length penalty.
#
# Interpolating instead charges the extra token its honest unigram cost.
# Balanced accuracy over the same gold standard: 0.835. The weight is high
# because the bigram evidence is what carries the decision; the unigram mass is
# a floor, not a partner.
BIGRAM_WEIGHT = 0.99


class ContextModel:
    """
    An interpolated bigram model over the line-interior lexicon.

    The same tables `validate_xml_hyphens.build_lexicon` produces, used for a
    different question: not "how is this pair usually written" but "which
    reading does *this* sentence make probable". Interior tokens are the right
    reference for both, being the one position a line break cannot damage.

    The interpolation is the whole design. See `BIGRAM_WEIGHT`: smoothing this
    the obvious way makes the model answer "one word" to almost everything, and
    the free gold standard is what caught it.
    """

    def __init__(self, unigrams: Counter, bigrams: Counter,
                 weight: float = BIGRAM_WEIGHT):
        self.unigrams = unigrams
        self.bigrams = bigrams
        self.weight = weight
        self.vocabulary = len(unigrams)
        self.tokens = sum(unigrams.values())

    def _logp(self, previous: Optional[str], word: str) -> float:
        probability = ((self.unigrams[word] + 1)
                       / (self.tokens + self.vocabulary))
        if previous is not None and self.unigrams[previous]:
            probability = (
                self.weight * self.bigrams[(previous, word)]
                / self.unigrams[previous]
                + (1 - self.weight) * probability)
        # The floor only ever fires on a degenerate model: with `weight` below
        # 1 the unigram term already guarantees a positive probability.
        return math.log(max(probability, 1e-12))

    def score(self, sequence: List[str]) -> float:
        return sum(self._logp(sequence[i - 1] if i else None, word)
                   for i, word in enumerate(sequence))

    def margin(self, before: List[str], left: str, right: str,
               after: List[str], window: int = 1) -> float:
        """
        How much the joined reading beats the separated one, in log units.

        Positive favours `left+right` as one word, negative favours two. The
        separated reading carries one token more and pays for it, which is a
        language model working as intended - probability is assigned to whole
        strings and the longer string has more to explain. What matters is that
        it pays the *right* price: the extra token's own unigram cost, not a
        flat smoothing penalty that would decide every break before the
        sentence was consulted (see `BIGRAM_WEIGHT`).

        `window` is a parameter and not a constant because it once was one, in
        the module this class was lifted out of, and the coupling was invisible
        until a second caller existed. One is the only value that does anything
        in a bigram model: every term beyond the token adjacent to the break is
        identical under both readings and cancels out of the difference.
        """
        head = before[-window:] if window else []
        tail = after[:window] if window else []
        return (self.score(head + [left + right] + tail)
                - self.score(head + [left, right] + tail))
