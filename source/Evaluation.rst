Evaluation
========================================

We introduces several evaluations metrics derived from accuracy.

Accuracy
----------
For each abstract in test set, we assign a single label based on our classification model.
And if that assigned label equals to the true label in the test set, we count one. The calculation is as followed:

:math:`Accuracy = \frac{\sum_i I(predicted_i = truth_i)}{number of all paper}`,

where :math:`I` is an indicator function,
:math:`predicted_i` is the predicted field of study of abstract i and :math:`truth_i` is the true field of study of abstract i
shown in the test set.

    ::

        >> predicted_label = [[1],[3]]
        >> true_label = [[0],[1]]
        >> accuracy_0 = 0
        >> accuracy_1 = 0
        >> average_accuracy = (0 + 0)/2 = 0

Example code of accuracy calculation.
    ::

        predictions_train = lrModel.transform(train_df)
        predictions_valid = lrModel.transform(valid_df)

        train_acc = evaluator.evaluate(predictions_train)
        valid_acc = evaluator.evaluate(predictions_valid)
        print("Train acc:", train_acc)
        print("Valid acc:", valid_acc)


Adjusted Accuracy
------------------------------
However, some papers have more than one field of study, but only one field of study of the paper in sampled to the test set.
So the accuracy above might miss some field of study. We propose a metric called adjusted accuracy. Specifically, for each abstract
in the test set, we still assign one label based on the classification. If the the assigned label in the set of true label, we count one.
The calculation is as followed:

:math:`Adjusted Accuracy = \frac{\sum_i I(if predicted_i \in S_i)}{number of all paper}`,

where where :math:`I` is an indicator function,
:math:`predicted_i` is the predicted field of study of abstract i and :math:`S_i` is all the field of study of abstract.

    ::

        >> predicted_label = [[1],[3]]
        >> true_label = [[0,3,4],[0,1,2,3,4]]
        >> accuracy_0 = 0
        >> accuracy_1 = 1
        >> average_accuracy = (0 + 1)/2 = 0.5

    ::

        def Average_acc(pred, true):
            accuracy = []
            true_labels = true.groupby("paperid")["label"].apply(list).to_dict()
            pred_labels = dict(zip(pred.paperid, pred.prediction))
            for key, val in true_labels.items():
                pred = pred_labels[key]
                accuracy.append(pred in val)
            return np.mean(accuracy)

         avg_acc = Average_acc(pred = predictions_valid_label_df, true = true_valid_label_df)
         print("Average accuracy:", avg_acc)

Average Accuracy
------------------------------

Sometimes it is not appropriate to predict only one label, so e also propose a multi-label prediction and evaluate it with average accuracy.
For each abstract, we assign top :math:`k` labels to it based on our classification probability where :math:`k \leq 5`. Then for each paper,
we calculate how many predicted label are in the set of true label.

    ::

        >> predicted_label = [[0,1,2],[0,1,2,3,4]]
        >> true_label = [[0,3,4],[0,1,2,3,4]]
        >> accuracy_0 = 1/3
        >> accuracy_1 = 4/5
        >> average_accuracy = (1/3 + 4/5)/2 = 0.5





