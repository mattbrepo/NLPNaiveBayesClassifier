# NLPNaiveBayesClassifier
A simple [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) for a [Natural Language Processing (NLP)](https://en.wikipedia.org/wiki/Natural_language_processing) problem.

**Language: Python**

**Start: 2017**

## Why
I read an article in 2017 about how to build a simple [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) for a [Natural Language Processing (NLP)](https://en.wikipedia.org/wiki/Natural_language_processing) problem. I recently stumbled into the notes I took at that time. Along with the notes, there was a piece of code that I forgot about. I started from that and created this example that builds a model to predict which of two writers is the author of a given sentence. 

I took three Charles Dickens books from the [Project Gutenberg](https://gutenberg.org/):
- [Oliver Twist](https://gutenberg.org/ebooks/730)
- [David Copperfield](https://gutenberg.org/ebooks/766)
- [A Christmas Carol](https://gutenberg.org/ebooks/24022)

and three Jane Austen books:
- [Emma](https://www.gutenberg.org/ebooks/158)
- [Sense and Sensibility](https://www.gutenberg.org/ebooks/161)
- [Pride and Prejudice](https://www.gutenberg.org/ebooks/1342)

For each author, 1000 lines are sampled to compose the training set (aka the knowledge base). Then the Bayes' Theorem is used to predict the author of 20 sentences (test set), 10 of Jane Austen (positive class) and 10 of Charles Dickens (negative class). The results are not great, but honestly way better that I would have expected:

```
15 correct out of 20
Sensitivity: 0.7
Specificity: 0.89
Accuracy: 0.79
```

The model is extremely dependent from the sampling and therefore very unstable. Over many runs, it's not difficult to see way worse results:

```
12 correct out of 20
Sensitivity: 1.0
Specificity: 0.33
Accuracy: 0.75
```

## A bit of theory
The idea is to compare the probability of two possible classes (_C1_ and _C2_) for a given sentence (_S_). The probabilities are calculated with the Bayes' Theorem:

$$ P(C_1 | S) > P(C_2 | S) $$

$$ \frac{P(S | C_1) \cdot \cancel{P(C_1)}}{\cancel{P(S)}} > \frac{P(S | C_2) \cdot \cancel{P(C_2)}}{\cancel{P(S)}} $$

P(C1) = P(C2) because, in this case, the same amount of sentences are taken for both authors. 

The P(S | C) is calculated as the product of the probabilities of each word being part of the class _C_. For example, if we assume _S_ is composed of four words (_W1_, _W2_, _W3_, _W4_):

$$ P(S | C_1) = P(W_1, W_2, W_3, W_4 | C_1) = P(W_1 | C_1) \cdot P(W_2 | C_1) \cdot P(W_3 | C_1) \cdot P(W_4 | C_1) $$

For each probability, we add 1 to every count of the word so that the product of the probabilities cannot go to zero (due to a word not present in the knowledge base) and we compensate by dividing by the number of all possible words. This technique is called [Laplace smoothing](https://en.wikipedia.org/wiki/Additive_smoothing).
