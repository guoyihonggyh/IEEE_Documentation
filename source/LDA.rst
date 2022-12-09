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
#. We use IDF to obtain the candidate set of words.
#. We use sparknlp to train LDA model with 6, 10, 20, 50 clusters respectively.
#. Store the topic distribution result.

Load the dataset and convert it to spark dataframe
    ::

        # Read in data
        data_path = "~/IEEE-JHU/00_data/02_sample_level_2/sample_l2_v2_73.csv"
        data = pd.read_csv(data_path)
        train = data[data.train_valid == 0]
        valid = data[data.train_valid == 1]
        train_df = spark.createDataFrame(train[["fos", "paperid", "abstract"]]).toDF("fos", "paperid", "abstract")
        valid_df = spark.createDataFrame(valid[["fos", "paperid", "abstract"]]).toDF("fos", "paperid", "abstract")

Tokenize the sentence, normalize it, remove the stop words, build a pipeline and fit transform
    ::

        # clean input
        document_assembler = DocumentAssembler() \
            .setInputCol("abstract") \
            .setOutputCol("document") \
            .setCleanupMode("shrink")
        # Split sentence to tokens(array)
        tokenizer = Tokenizer() \
          .setInputCols(["document"]) \
          .setOutputCol("token")
        # clean unwanted characters and garbage
        normalizer = Normalizer() \
            .setInputCols(["token"]) \
            .setOutputCol("normalized")
        # remove stopwords
        stopwords_cleaner = StopWordsCleaner()\
              .setInputCols("normalized")\
              .setOutputCol("cleanTokens")\
              .setCaseSensitive(False)
        # stem the words to bring them to the root form.
        stemmer = Stemmer() \
            .setInputCols(["cleanTokens"]) \
            .setOutputCol("stem")
        # Finisher is the most important annotator. Spark NLP adds its own structure when we convert each row in the dataframe to document. Finisher helps us to bring back the expected structure viz. array of tokens.
        finisher = Finisher() \
            .setInputCols(["stem"]) \
            .setOutputCols(["tokens"]) \
            .setOutputAsArray(True) \
            .setCleanAnnotations(True)
        # We build a ml pipeline so that each phase can be executed in sequence. This pipeline can also be used to test the model.
        nlp_pipeline = Pipeline(
            stages=[document_assembler,
                    tokenizer,
                    normalizer,
                    stopwords_cleaner,
                    stemmer,
                    finisher])
        nlp_model = nlp_pipeline.fit(train_df)
        train_df = nlp_model.transform(train_df)
        valid_df = nlp_model.transform(valid_df)

IDF
    ::

        cv = CountVectorizer(inputCol="tokens", outputCol="features", vocabSize=10000, minDF=3.0)
        idf = IDF(inputCol="features",outputCol='vectorizedFeatures')
        pipeline = Pipeline(
            stages = [cv, idf]
        )

        tfidf_model = pipeline.fit(train_df)
        vectorized_tokens_train = tfidf_model.transform(train_df)
        vectorized_tokens_valid = tfidf_model.transform(valid_df)

Train the LDA model with 100 topics and transform
    ::

        num_topics = 100
        lda = LDA(k=num_topics, maxIter=10)
        model = lda.fit(vectorized_tokens_train)
        trans_train = model.transform(vectorized_tokens_train)
        trans_valid = model.transform(vectorized_tokens_valid)












