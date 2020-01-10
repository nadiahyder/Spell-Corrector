import math, collections

class SmoothBigramModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.smoothBigramCounts = collections.defaultdict(lambda: 0)
    self.twoWordCounts = collections.defaultdict(lambda: 0)
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
        twoWords = token + " " + sentence.data[i+1].word
        self.smoothBigramCounts[token] += 1
        self.twoWordCounts[twoWords] +=1
        self.total += 1

    pass

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """

    score = 0.0

    for i in xrange(0, len(sentence)-1):
      token = sentence[i]
      countW1 = self.smoothBigramCounts[token] # sentence[i] is token

      twoWords = token + " " + sentence[i + 1]
      countW1W2 = self.twoWordCounts[twoWords]

      V = len(self.smoothBigramCounts)

      score += math.log(countW1W2 + 1)
      score -= math.log(countW1 + V)

    return score
