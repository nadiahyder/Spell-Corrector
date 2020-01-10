import math, collections

class CustomModel: #4-gram
  # note: takes a while to load but it gets there eventually

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.trigramCounts = collections.defaultdict(lambda: 0)
    self.quadgramCounts = collections.defaultdict(lambda: 0)

    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
    for sentence in corpus.corpus:
      thirdPrev = None
      secondPrev = None
      firstPrev = None

      for datum in sentence.data:
        current = datum.word
        self.unigramCounts[tuple([current])] += 1
        if firstPrev != None:
          self.bigramCounts[tuple([firstPrev, current])] += 1
        if secondPrev != None:
          self.trigramCounts[tuple([secondPrev, firstPrev, current])] += 1
        if thirdPrev != None:
          self.quadgramCounts[tuple([thirdPrev, secondPrev, firstPrev, current])] += 1

        thirdPrev = secondPrev
        secondPrev = firstPrev
        firstPrev = current

  def score(self, sentence):
    """ With list of strings, return the log-probability of the sentence with language model. Use
        information generated from train.
    """

    score = 0.0
    firstPrev = None
    secondPrev = None
    thirdPrev = None

    for current in sentence:

      quadCount = self.quadgramCounts[tuple([thirdPrev, secondPrev, firstPrev, current])]
      triCount = self.trigramCounts[tuple([secondPrev, firstPrev, current])]
      biCount = self.bigramCounts[tuple([firstPrev, current])]
      uniCount = self.unigramCounts[tuple([current])]

      if (quadCount > 0):
        score += math.log(quadCount)
        score -= math.log(triCount)
      else:
        if (triCount > 0):
          score += math.log(triCount)
          score -= math.log(biCount)
        else:
          if (biCount > 0):
            score += math.log(biCount)
            score -= math.log(self.unigramCounts[tuple([firstPrev])])
          else:
            score += math.log(uniCount + 1.0)
            score -= math.log(sum(self.unigramCounts.values()) + len(self.unigramCounts))

      thirdPrev = secondPrev
      secondPrev = firstPrev
      firstPrev = current

    return score