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

::

    with open('/home/ubuntu/IEEE-UCLA-New/Data/abs_dict.json') as f:
        abs_dict = json.load(f)
    from reference_dict_new import *
    with open('/home/ubuntu/IEEE-UCLA-New/Data/concepts.json') as f:
        concepts = json.load(f)
    with open('/home/ubuntu/IEEE-UCLA-New/Data/definitions.json') as f:
        definitions = json.load(f)

    from tagging import Tagging
    tagging_test = Tagging(num_papers=10000,num_seeds=len(definitions),paperabstract_dict=abs_dict,seeds_dict=definitions)
    tagging_test.load_ieee_model()
    print("Step 1 completed")
    # should set the compute_flag = True when first time running
    tagging_test.create_paper_embeddings(compute_flag = False)
    print("Step 2 completed")
    # should set the compute_flag = True when first time running
    tagging_test.create_seeds_embeddings(compute_flag = False)
    print("Step 3 completed")
    tagging_test.create_similarity_scores_torch(compute_flag = False)
    print("Step 4 completed")




With the embedding, we can move forward to the feature engineering

