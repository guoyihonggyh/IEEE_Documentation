Classification model
========================================

Logistic Regression
-----------------------
With the feature generated in the previous section and the corresponding field of study generated in the sample
creation, we can train a classification model for tagging. Specifically, we employ the multinomial
logistic regression with L1 and L2 regularization, where :math:`\alpha` and :math:`\lambda` are hyper-parameters.

:math:`\alpha(\lambda ||w||_1) + (1-\alpha) (\frac{\lambda}{2}||w||^2_2) ,\alpha \in [0,1] ,\lambda \geq 1`

We tune the :math:`\alpha` and :math:`\lambda` for logistic regression.


train test data split
-----------------------

We have two set of data, both of which are pre-processed with feature engineering and the two set to data has totally
different field of study set with hierarchy relation. So we decided to train two model separately.

#. level 0&1 sample

#. level 2 sample

The example code:










