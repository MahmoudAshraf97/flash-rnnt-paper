## Making each operation more efficient

Recomputing those values saves memory, but it also means doing more work. To make
these calculations more efficient, Flash combines consecutive steps so the GPU
can pass results directly between them, instead of writing them to memory and
reading them back. This also removes the need to store some temporary results.

Flash implements these combined steps in specialized GPU functions, called
kernels. Its joint kernel combines the audio and text representations and
processes the result without saving a separate intermediate result between
steps.

The same approach helps when converting vocabulary scores into the two
probabilities needed by the loss. Rather than storing a probability for every
vocabulary entry and then selecting two, the kernel calculates and writes those
two directly. During backpropagation, it also overwrites the scores with their
gradients, avoiding another equally large allocation.

These improvements are useful with or without recomputation. Flash therefore
also offers an untiled mode that keeps the joint intermediates rather than
recreating them. Both modes use the specialized kernels; tiled execution spends
additional computation to save more memory, while untiled execution avoids that
cost when the available memory allows it.
