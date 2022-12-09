BERT for abstract embedding
========================================

BERT introduction
--------------------

Bidirectional Encoder Representations from Transformers (BERT)

Bidirectional Encoder Representations from Transformers (BERT)
is a transformer-based machine learning technique for natural language processing (NLP)
pre-training developed by Google. In our work, we propose to use BERT to obtain a embedding for each
abstract. Specifically, we apply universal sentence encoder (USE) for abstract embedding.

Implementation
--------------------
Given an abstract, USE can directly output an embedding. USE for English trained
with a conditional masked language model.
The universal sentence encoder family of models maps
the text into high dimensional vectors that capture sentence-level semantics.
Our English-base (en-base) model is trained using a conditional
masked language model. The model is intended to be used for text classification,
text clustering, semantic textual similarity, etc.
It can also be used used as modularized input for multimodal
tasks with text as a feature. The base model employs a 12 layer BERT transformer architecture.



The example code:

.. code-block:: python






With the embedding, we can move forward to the feature engineering

