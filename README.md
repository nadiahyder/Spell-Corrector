# Spell-Corrector
Noisy channel spell corrector in Python. Has an edit model for generating word edits, then trains a number of language models on a data corpus to determine the most probable edit. Includes the following language models: smoothed unigram, smoothed bigram, unsmoothed bigram with backoff, and a 4-gram. Performance on the test set was:
• Unigram: 0.012739
• Uniform: 0.055202
• Smooth unigram: 0.110403
• Smooth bigram: 0.140127
• Backoff model: 0.182590
• 4-gram: 0.195329
