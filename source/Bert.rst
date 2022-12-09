Universal Sentence Encoder for abstract embedding
===============================================================================

Introduction
--------------------

Bidirectional Encoder Representations from Transformers (BERT)
is a transformer-based machine learning technique for natural language processing (NLP)
pre-training developed by Google. In our work, we apply Universal Sentence Encoder (USE), a pre-trained BERT model, for
abstract embedding. Given an abstract, USE can directly output an embedding. USE for English trained
with a conditional masked language model.
The universal sentence encoder family of models maps
the text into high dimensional vectors that capture sentence-level semantics.
Our English-base (en-base) model is trained using a conditional
masked language model. The model is intended to be used for text classification,
text clustering, semantic textual similarity, etc.
It can also be used used as modularized input for multimodal
tasks with text as a feature. The base model employs a 12 layer BERT transformer architecture and output a 512 dimension
vector for each abstract.

Implementation
--------------------
Here is an example code for generating abstract embedding for level 0&1 samples. Please refer to ~/IEEE-JHU/alicia for
more examples.

    ::

        # Read in data
        train = pd.read_csv("~/IEEE-JHU/00_data/01_sample_level_0_level_1/paper_fos_abstract_level_01_en_train.csv")
        valid = pd.read_csv("~/IEEE-JHU/00_data/01_sample_level_0_level_1/paper_fos_abstract_level_01_en_valid.csv")

        train_df = spark.createDataFrame(train[["fos", "paperid", "abstract"]]).toDF("fos", "paperid", "abstract")
        valid_df = spark.createDataFrame(valid[["fos", "paperid", "abstract"]]).toDF("fos", "paperid", "abstract")

        documentAssembler = DocumentAssembler() \
            .setInputCol("abstract") \
            .setOutputCol("document")

        use = UniversalSentenceEncoder.pretrained()\
         .setInputCols(["document"])\
         .setOutputCol("sentence_embeddings")


        embeddingsFinisher = EmbeddingsFinisher() \
            .setInputCols(["sentence_embeddings"]) \
            .setOutputCols("finished_embeddings") \
            .setOutputAsVector(True) \
            .setCleanAnnotations(True)

        pipeline = Pipeline().setStages([
            documentAssembler,
            use,
            embeddingsFinisher
        ])

        train_result = pipeline.fit(train_df).transform(train_df)
        valid_result = pipeline.fit(train_df).transform(valid_df)


With the embedding, we can move forward to the feature engineering.

