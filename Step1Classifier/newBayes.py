#!/usr/bin/python
"""Naive Bayes Classifier
"""
from __future__ import division  # Always use float division
import re
import unicodedata
import sys
import re
import nltk
import string
from collections import defaultdict
from operator import mul
from string import ascii_uppercase
import time
from nltk.stem.wordnet import WordNetLemmatizer
stop_words = set(["&&","the","of","and","a","how","to","in","is","you","that","it","he","for","was","on","are","as","they","at","be","this","from","i","have","or","by","one","had","not","but","what","all","were","when","we","there","can","an","your","which","their","said","if","do","will","each","about","up","out","them","then","so","these","would","into","has","more","two","him","see","could","no","make","than","first","been","its","who","made","over","did","down","only","way","find","use","may","long","very","after","words","called","just","where","know","get","through","back","much","go","new","our","me","man","too","any","also","around","another","came","above","again","against","am","arent","because","before","being","below","between","both","cant","cannot","couldnt","didnt","does","doesnt","doing","dont","during","few","further","hadnt","hasnt","havent","having","hed","hell","here","hows","i","id","ill","im","ive","isnt","itself","lets","most","mustnt","my","myself","nor","off","once","ought","ours ","ourselves","own","same","shant","she","shed","shell","shes","should","shouldnt","some","such","thats","theirs","themselves","theres","theyd","theyll","theyre","theyve","those","under","until","wasnt","wed","well","weve","werent","whats","whens","wheres","while","whos","whom","why","whys","wont","wouldnt","youd","youll","youre","youve","yours","yourself","yourselves","able","across","almost","among","dear","either","else","ever","every","got","hers","however","least","let","likely","might","must","neither","often","rather","say","says","since","tis","twas","us","wants","yet"])

my_features = ["pain ","patients  ","glucose ","primary  ","life ","of life ","trial  ","the primary ","insulin ","quality of life ","survival ","efficacy ","scale ","patients with  ","free ","the efficancy ","treatment  ","fat ","symptoms ","secondary ","death ","score ","fasting ","events ","free survival ","protein ","the treatment ","overall survival ","postoperative ","end points ","overall ","the treatment of ","quality of ","quality ","randomized  ","serum ","diabetes ","scores ","costs ","hazard ratio ","concentrations","treatment of","improvement ","type diabetes ","concentrations ","dietary ","patients were","levels"]

class Feature(object):
  """A Feature represents a boolean assessment of an input string.
  The judge(s) method determines whether this feature is present in s.

  This feature base class only tests for the presence of a substring in s.
  """
  

	
	
  def __init__(self, base):
    self.base = sanitize(base)
    # Counts is a 2-level dict that keeps a count of our training strings.
    # The first level index is the class_number.
    # The second level index is a bool indicating the presence of this feature.
    # The values are initialized to one for smoothing
    self.counts = defaultdict(lambda: {True: 1, False: 1})

  def judge(self, target):
    """Judges the target against base.
    Override this method for more complicated Features (regex, etc.).

    Args:
      target: String the test string to judge against.
    Returns:
      Returns True iff base is a substring within target.
    """
    return self.base in target

  def train(self, target, class_number):
    """Trains the system against the target string and class_number.
    Args:
      target: String the training string to train with.
      class_number: Int the authoritative class the target is a part of.
    """
    presence = self.judge(target)
    self.counts[class_number][presence] += 1

  def test(self, target, class_number):
    """Determines the probability of this feature's presence for the class.

    Returns:
      Float probability that the feature is present for the class_number.
    """
    presence = self.judge(target)
    return (self.counts[class_number][presence] /
      (self.counts[class_number][False] + self.counts[class_number][True]))


class RegexFeature(Feature):
  """Regex-based Features."""
  def __init__(self, r):
    self.r = re.compile(r, re.IGNORECASE)
    super(RegexFeature, self).__init__(r)

  def judge(self, target):
    return (bool)(self.r.match(target))


class Classifier(object):
  """Naive Bayes Classifer.

  Useful methods:
    addFeature(f): Adds a feature f to the Classifier.
    train(s, c): Trains the classifier according to string s and class c.
    classify(s): Classifies string s in a class based on MAP.
    test(s, c): Returns the probability string s belongs to class c.
  """
  def __init__(self):
    self.features = []  # A list of all Feature objects
    self.total_classes = 2
    self.class_counts = defaultdict(lambda: 0)

  def priorProbability(self, class_number):
    """Returns the prior probability of class_number."""
    return (self.class_counts[class_number] /
      (self.class_counts[0] + self.class_counts[1]))

  def likelihood(self, target, class_number):
    """Returns the likelihood of the target given the class_number.
    This is P(F1=f1 ... Fn=fn | C=class_number).
    """
    likelihoods = [f.test(target, class_number) for f in self.features]
    return (reduce(mul, likelihoods))  # Product of the list items

  def addFeature(self, f):
    """Adds a feature f to our Classifier."""
    self.features.append(f)

  def train(self, s, class_number):
    """Trains the classifier according to string s and class class_number.

    (1) Updates the counts that are used to compute the likelihoods.
    (2) Updates the counts that are used to compute the prior probabilities.
    """
    for f in self.features:
      f.train(s, class_number)            # (1)
    self.class_counts[class_number] += 1  # (2)

  def classify(self, s):
    """Returns the class_number that string s most likely belongs to."""
    probabilities = [self.test(s, c) for c in range(self.total_classes)]
    # Return the class_number/index of the largest probability
    return max(enumerate(probabilities), key=lambda x, y: cmp(x[1], y[1]))[0]

  def test(self, s, class_number):
    """Returns the probability s belongs to class_number."""
    classes = xrange(self.total_classes)
    
    # Prior probabilities
    # P(C=class_number)
    priors = [self.priorProbability(c) for c in classes]
	# Likelihoods
    # P(F1=f1 ... Fn=fn | C=classNumber)
    """for c in classes:
		print s
		print self.likelihood(s,c)
		time.sleep(0.1)"""
    likelihoods = [self.likelihood(s, c) for c in classes]
    # Intermediate Probabilities
    # P(C=classNumber) * P(F1=f1 ... Fn=fn | C=classNumber)
	
    intermediates = [priors[i] * likelihoods[i] for i in classes]
    # Posterior Probability
    # P(C=classNumber | F1=f1 and F2=f2 and ... Fn=fn)
	 
    return intermediates[class_number] / sum(intermediates)

def remove_stop_words(input_string):
    for item in stop_words:
        input_string = input_string.replace(item, '')
    return input_string

def sanitize(s):
  """Sanitize input string s and return it."""
  """while  not s.endswith('\n'):
    newline = remove_stop_words(s)
    print "A" """
  lmtzr = WordNetLemmatizer()
  stopword = "DONE"
  line = re.sub(r'[^\x00-\x7f]',r'',s)
  line = line.lower()
  exclude = set(string.punctuation)
  line = ''.join(ch for ch in line if ch not in exclude)
  newline1 = (" ".join(word for word in line.split() if word not in stop_words))
  newline2 = (" ".join(word for word in newline1.split() if not word.isdigit()))
  newline = (" ".join(lmtzr.lemmatize(word) for word in newline2.split() ))
  
  _ret = re.sub(r'[^\x00-\x7f]',r'',newline) # Remove Special characters.
  _ret = ' '.join(_ret.split())  # Remove whitespace
  _ret = _ret.upper()  # Capitalize
  return _ret

  

def main():
  if len(sys.argv) != 5:
    print """\nUsage:
    $ bayes.py class1-trainfile class2-trainfile class1-testfile class2-testfile
    """
    return -1
  trainFile1, trainFile2 = sys.argv[1:3]
  testFile1, testFile2 = sys.argv[3:5]

  # Populate Classifier
  classifier = Classifier()
  digrams = ["th", "he", "in", "en", "nt", "re", "er", "an", "ti", "on", "at"]
  trigrams = ["tha", "ent", "tio", "nde", "nce", "edt", "tis", "sth", "men"]
  for c in (list(ascii_uppercase) +digrams ):
    # Add features for single letters, digrams, and trigrams
    classifier.addFeature(Feature(c))
  # Add some prefix and suffix regexes
  prefixes = [r"^anti.*$", r"^de.*$", r"^dis.*$", r"^en.*$", r"^em.*$",
      r"^fore.*$", r"^in.*$",  r"^im.*$", r"^inter.*$", r"^mis.*$",
      r"^non.*$", r"^over.*$",  r"^pre.*$", r"^re.*$", r"^semi.*$",
      r"^sub.*$", r"^super.*$",  r"^trans.*$", r"^un.*$", r"^under.*$",
      r"^san.*$"]
  suffixes = [r"^.*ville$", r"^.*sk$", r"^.*able$", r"^.*en$", r"^.*er$",
      r"^.*est$", r"^.*ful$",  r"^.*ing$", r"^.*ion$", r"^.*ty$", r"^.*ive$",
      r"^.*less$", r"^.*ly$",  r"^.*ment$", r"^.*ness$", r"^.*ous$", r"^.*es$",
      r"^.*port$", r"^.*city$", r"^.*ss$"]

  list_features  = []
  with open(trainFile1, 'r') as f:
    for line in f:
      line = line.lower()
      line = sanitize(line)
      for word in line.split():
        if word not in list_features :
          list_features.append(word)
      """for item in nltk.bigrams(line.split()) :
        #print "here" 
        #print ' '.join(item)
        list_features.append(' '.join(item))"""
 
  
  print ("list complete  " )
  print( len(list_features) )
  
  
  
  
  for word in list_features:
    print word
    classifier.addFeature(Feature(word))
    
  for word in my_features:
    classifier.addFeature(Feature(word))
  print ("list complete  " )
  print( len(list_features) )
    
	
 

  
 
  print ("********************training************")
  
  
 # Train!
  with open(trainFile1, 'r') as f:
    for line in f:
      classifier.train(sanitize(line), 0)  # Train class_number=0
  with open(trainFile2, 'r') as f:
    for line in f:
      classifier.train(sanitize(line), 1)  # Train class_number=1
	  
  print ("***********Training Complete***********")
  """"file1 = open('likelihoods453.csv', 'w')
  
  for feat in classifier.features:
    file1.write( feat.base + "," )   
    file1.write(  str(classifier.likelihood(feat.base,0)) + ",")
    file1.write(  str(classifier.likelihood(feat.base,1)))
    file1.write ("\n")"""
    
    


  

  # Test!
  results = []
  results1 = []
  results2  = []
  with open(testFile1, 'r') as f:
    actual_class_number = 0
    for line in f:
      if not line.strip():
        continue
      target = sanitize(line)
      probability = classifier.test(target, 0)
      predicted_class_number = (int)(probability <= 0.48)
      results.append([target, actual_class_number, predicted_class_number,
          probability,
          actual_class_number == predicted_class_number])
      results1.append([target, actual_class_number, predicted_class_number,
          probability,
          actual_class_number == predicted_class_number])
	 	 
	
  with open(testFile2, 'r') as f:
    actual_class_number = 1
    for line in f:
      if not line.strip():
        continue
      target = sanitize(line)
      probability = classifier.test(target, 0)
      predicted_class_number = (int)(probability <= 0.48) 
      results.append([target, actual_class_number, predicted_class_number,
          probability,
          actual_class_number == predicted_class_number])
      results2.append([target, actual_class_number, predicted_class_number,
          probability,
          actual_class_number == predicted_class_number])
  # Print results
  fileOut1 = open('out1.txt', 'w')
  fileOut0 = open('out0.txt', 'w')
  print ""
  print "string                          true class  pred.class   postr C=1  correct?"
  print "------                          ----------  ----------   ---------  --------"
  print ""
  for r1 in results1:
    if len(r1[0]) >= 40:
      r1[0] = r1[0][:37] + "... "
    sys.stdout.write(r1[0])
    sys.stdout.write(' ' * (41 - len(r1[0])))  # Print spaces
    sys.stdout.write("{:d}".format(1 - r1[1]))
    sys.stdout.write(' ' * 11)
    sys.stdout.write("{:d}".format(1 - r1[2]))
    fileOut1.write("{:d}".format(1 - r1[2]))
    fileOut1.write("    ")
    fileOut1.write(r1[0])
    
    fileOut1.write("\n")
    sys.stdout.write(' ' * 4)
    sys.stdout.write("{0:.6f}".format(r1[3]))
    sys.stdout.write(' ' * 9)
    if r1[4]:
      sys.stdout.write('Y')
    else:
      sys.stdout.write('N')
    sys.stdout.write('\n')
  print "R2 starts"
  print ""
  print ""

  for r2 in results2:
    if len(r2[0]) >= 40:
      r2[0] = r2[0][:37] + "... "
    sys.stdout.write(r2[0])
    sys.stdout.write(' ' * (41 - len(r2[0])))  # Print spaces
    sys.stdout.write("{:d}".format(1 - r2[1]))
    sys.stdout.write(' ' * 11)
    sys.stdout.write("{:d}".format(1 - r2[2]))
    fileOut0.write("{:d}".format(1 - r2[2]))
    fileOut0.write("    ")
    fileOut0.write(r2[0])
    fileOut0.write("\n")
    sys.stdout.write(' ' * 4)
    sys.stdout.write("{0:.6f}".format(r2[3]))
    sys.stdout.write(' ' * 9)
    if r2[4]:
      sys.stdout.write('Y')
    else:
      sys.stdout.write('N')
    sys.stdout.write('\n')

  # Compute and print accuracy and mean squared error
  n_results = len(results)
  n_results1 = len(results1)
  n_results2 = len(results2)
  n_correct1 = len([r1[4] for r1 in results1 if r1[4]])
  n_correct2 = len([r2[4] for r2 in results2 if r2[4]])

  accuracy1 = n_correct1 / n_results1
  accuracy2 = n_correct2 / n_results2
  #error = sum([((1 - r[1]) - r[3]) ** 2 for r in results]) / n_results
  print ""
  print "Summary of Class 1 : {:d} test cases, {:d} correct; accuracy = {:.2f}".format(
      n_results1, n_correct1, accuracy1)
  print "Summary of cLASS 2 : {:d} test cases, {:d} correct; accuracy = {:.2f}".format(
      n_results2, n_correct2, accuracy2)
  #print "Mean squared error: {:.6f}".format(error)


if __name__ == "__main__":
  main()
