# Flash RNN-T: Making Room to Train Speech Recognition

![Tarteel’s logo formed from emerald glass bars, with four adjacent bars illuminated in mint to represent one active computation tile.](images/hero-concepts/08-tarteel-four-bar-tile.png)

Training speech recognition models can quickly exhaust GPU memory.
For RNN-T, an architecture used in streaming speech recognition, a major source
of that memory usage is the calculation that connects the audio to its transcript.
As both grow longer, the temporary data produced by this calculation grows rapidly.

We built Flash RNN-T (referred to as ‘Flash’ in this post) to reduce that memory
requirement. It combines more
efficient GPU operations with a way to process the calculation in small pieces,
keeping less temporary data and recreating it when needed.
