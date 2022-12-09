Feature Engineering
========================================

Method
-------
To make the feature more representative, we combine the output of LDA and
BERT output to obtain a new feature. We concatenate the vector of LDA and
the embedding of Universal sentence encoder (USE). Specifically, for LDA with $$k$$ topic, we have a
k dimension topic distribution for each abstract. For USE, we have a 512 dimension sentence embedding.
And we concatenate this two part of vector, represented as follows:


:math:`v^i = [v^{LDA}_i, v^{USE}_i]`

where :math:`v^i` is the final feature representation of an abstract :math:`i`, :math:`v^{LDA}_i` is the
topic distribution vector and :math:`v^{USE}_i` is the embedding from USE. The example code is shown below:


     ::

          assembler = VectorAssembler().setInputCols(['bert', "lda"]).setOutputCol('features')
          train_df = assembler.transform(trans_train)
          valid_df = assembler.transform(trans_valid)










