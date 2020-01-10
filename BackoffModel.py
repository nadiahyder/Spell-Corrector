import math, collections

class BackoffModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
    # Tip: To get words from the corpus, try
    #    for sentence in corpus.corpus:
    #       for datum in sentence.data:
    #         word = datum.word

    for sentence in corpus.corpus:
      for i in xrange(0, len(sentence.data)-1):
        token = sentence.data[i].word
        current = sentence.data[i+1].word

        bigram = (token, current)

        self.bigramCounts[bigram] += 1
        self.unigramCounts[token] += 1
        self.total += 1

    pass


  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """

    score = 0.0

    for i in xrange(0, len(sentence) - 1):
      w1 = sentence[i]
      w2 = sentence[i+1] # current word

      bigram = (w1, w2)
      bigramCount = self.bigramCounts[bigram]

      if bigramCount > 0: #bigram
        score += math.log(bigramCount)
        score -= math.log(self.unigramCounts[w1])
      else:
        N = self.total
        V = len(self.unigramCounts)
        count = self.unigramCounts[w2]
        score += math.log(count + 1)
        score -= math.log(N + V)


    return score

