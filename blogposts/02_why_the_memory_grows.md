## Why the memory grows

Consider a recording of someone saying “hello there.” The transcript tells us
the words and their order, but it does not tell us exactly when each word was
spoken. A different speaker might stretch the first word or pause before the
second. Training needs to account for these possible timings rather than assume
one fixed match between the audio and the text.

RNN-T handles this by considering all valid alignments between the recording and
its transcript. Picture a grid: one axis follows the audio through time, while
the other follows progress through the transcript. A path through the grid
describes one possible alignment. The training calculation combines the
probabilities of all valid paths to measure how well the model explains the
transcript. That measure is called the loss.

[Animation: the same transcript, different possible timings.](animations/alignment.html)

At each position, a part of the model called the joint network combines what
the model has heard with the text it has already accounted for. It produces
scores for every possible next token—a word or word fragment in the model's
vocabulary. But speech unfolds over many audio steps, and each step does not
necessarily produce a new token. The model therefore also scores a special
**blank** token: “advance to the next audio step without adding text.” Blank can
occur during speech, not just silence.

This is where the storage grows. Twice as many audio steps means roughly twice
as many grid positions. If the transcript is also twice as long, the grid is
roughly four times larger. Each position carries an entire vocabulary's worth
of scores, alongside the joint network's internal values. Training on a batch
of recordings multiplies that storage again.

Calculating the loss is only the first part of training. The model then uses
backpropagation to work backward through the calculation and determine how its
parameters should change. This requires intermediate values from the earlier
calculation, so they are normally kept in memory until then. These saved values
can consume much of the available GPU memory, even when the recordings are only
around 20 seconds long.
