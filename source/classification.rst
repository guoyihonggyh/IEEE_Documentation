Classification model
========================================

Logistic Regression
-----------------------
With the feature generated in the previous section and the corresponding field of study generated in the sample
creation, we can train a classification model for tagging. Specifically, we employ the multinomial
logistic regression with L1 and L2 regularization, where :math:`\alpha` and :math:`\lambda` are hyper-parameters.

:math:`\alpha(\lambda ||w||_1) + (1-\alpha) (\frac{\lambda}{2}||w||^2_2) ,\alpha \in [0,1] ,\lambda \geq 1`

We tune the :math:`\alpha` and :math:`\lambda` for logistic regression with grid search.



We have two set of data, both of which are pre-processed with feature engineering and the two set to data has totally
different field of study set with hierarchy relation. So we decided to train two model separately.

#. level 0&1 samples

#. level 2 samples

Here is an example code for training on level 0&1 samples. Please refer to ~/IEEE-JHU/alicia for
more examples on different models and features.

Load the train and valid data and and obtain the best parameters with grid search.
    ::

        # load train and validation data
        sc = SparkContext.getOrCreate(SparkConf())
        pickleRdd_train = sc.pickleFile("./features/use_lda10_s1_train.pkl").collect()
        pickleRdd_valid = sc.pickleFile("./features/use_lda10_s1_valid.pkl").collect()

        train_df = spark.createDataFrame(pickleRdd_train)
        valid_df = spark.createDataFrame(pickleRdd_valid)

        # searching for the best params
        for regParam, elasticNetParam in itertools.product(*permutations):

        lr = LogisticRegression(maxIter = 20,
                                regParam = regParam,
                                elasticNetParam = elasticNetParam)
        lrModel = lr.fit(train_df)
        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

        predictions_train = lrModel.transform(train_df)
        predictions_valid = lrModel.transform(valid_df)

        train_acc = evaluator.evaluate(predictions_train)
        valid_acc = evaluator.evaluate(predictions_valid)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_result["regParam"] = regParam
            best_result["elasticNetParam"] = elasticNetParam

Train the best model and evaluate the accuracy
    ::

        # Best Model:
        lr = LogisticRegression(maxIter=20, regParam=best_result["regParam"], elasticNetParam=best_result["elasticNetParam"])
        lrModel = lr.fit(train_df)
        predictions_train = lrModel.transform(train_df)
        predictions_valid = lrModel.transform(valid_df)

        train_acc = evaluator.evaluate(predictions_train)
        valid_acc = evaluator.evaluate(predictions_valid)
        print('train accuracy', train_acc)
        print('validation accuracy', valid_acc)

Store the prediction
    ::

        predictions_valid_label = predictions_valid.select("paperid", "prediction")
        predictions_valid_label_df = predictions_valid_label.toPandas()
        predictions_valid_label_df.to_csv("./output/model3_valid_label.csv")


Other model
---------------

Please refer to ~/IEEE-JHU/alicia for other examples.

#. model1_mlr_s1_avg_bert.ipynb
#. model2_mlr_s1_use.ipynb
#. model3_mlr_s1_use_lda10.ipynb
#. model4_rf_s1_tfidf.ipynb
#. model5_mlp_s1_use_lda10.ipynb
#. model6_rf_s1_use_lda10.ipynb
#. model7_mlr_s3_use.ipynb
#. model8_mlr_s3_lda100.ipynb
#. model9_rf_s3_lda100.ipynb
#. model10_rf_s3_use.ipynb
#. model12_lr_s3_use_256_128.ipynb
#. model13_mlr_s3_use_lda.ipynb