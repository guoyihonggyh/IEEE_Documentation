Latent Dirichlet Allocation (LDA)
========================================

LDA introduction
------------------------
LDA, an unsupervised learning method, is used to classify text in a document to a particular topic.
It builds a topic per document (abstract) model and words per topic model, modeled as Dirichlet distributions.

#. Each document is modeled as a multinomial distribution of topics and each topic is modeled as a multinomial distribution of words.
#. LDA assumes that the every chunk of text we feed into it will contain words that are somehow related. Therefore choosing the right corpus of data is crucial.
#. It also assumes documents are produced from a mixture of topics. Those topics then generate words based on their probability distribution.

Implementation
------------------------------
We do the LDA with sparknlp with the following steps:

#. Given all the abstracts, we pre-process the data, remove the punctuation of the data, etc.
#. We tokenize the data and normalize it.
#. We use TF-IDF to obtain the candidate set of words.
#. We use sparknlp to train LDA model with 6, 10, 20, 50 clusters respectively.
#. Store the topic distribution result.

The example code:










