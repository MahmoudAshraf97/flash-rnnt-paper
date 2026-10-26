## Compute it, use it, make room

Keeping those intermediate values avoids repeating work, but the loss
calculation needs much less information than the joint network produces. At
each grid position, an alignment matching the transcript can either advance
through the audio using blank or produce the next transcript token. Calculating
these two probabilities requires scores across the vocabulary; once calculated,
however, the two probabilities are enough to continue the alignment calculation.

This gives us a way to work through the grid without keeping all its vocabulary
scores in memory. Flash groups grid positions into tiles. For each tile, it
computes the scores, saves the two probabilities per position, and discards the
scores and internal joint network values. The next tile reuses that working
space. Once all tiles are processed, their saved probabilities provide what we
need to calculate the loss.

To use that loss to train the model, we still need the values we discarded.
Flash therefore revisits each tile during backpropagation, recreates the
values needed for that calculation, and frees them before moving to the next
tile. This is called activation checkpointing: we repeat part of the computation so
that the large intermediate tensors only need to exist for one tile at a time.

[Animation: save the small outputs, recreate the large intermediates when needed.](animations/recomputation.html)

That exchange gives us control over the memory requirement. Smaller tiles use
less working space, while larger tiles give the GPU more work at once and reduce
the overhead of processing separate pieces. The saved probabilities still occupy
space across the grid, but the much larger joint network intermediates now
depend on the tile size rather than the length of the recording.
