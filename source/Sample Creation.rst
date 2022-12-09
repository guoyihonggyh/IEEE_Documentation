Sample Data Creation
========================

Data Description
-------------------------------

IEEE Xplore is one of the most popular digital libraries provided by the Institute of Electrical and Electronics Engineers （IEEE）. It hosts more than 5.8M+ publications for engineering, computing, and technology information around the globe.

However, the high volume and speed of influx of the scholarly publications requires a more "intelligent" library system that can automatically comprehend and identify topics from the publications, which lays a foundation for facilitating efficient searches on library.

The data contains 714,971 field of study(FOS) and is devided into 6 level, where as the level rise up, the field of study would be more specific. Since the size of the raw data are significantly large, it is not possible for one to train or test on the raw data. Then we need to sample the data from the database.

In practical, we would like to sample data level 0&1 and level 2.

Sample data creation
-------------------------------

Mainly, there are four steps to create sample data:


    Extract FOS and corresponding papers with constraint on level, score in database.

    Load sample data from database; Convert invert indexed abstract into in-reading order abstract.

        Originally, the inv_abstract was a dictionary whose keys are words and values are the index of the words. Then we put the key and value pairs into a tuple and sort by the index, which would make the words listed in the right sequence. In the end, we could convert the words tuple to a string, which is exactly the abstract we want. Meanwhile, we also remove the stop words in abstract as well.

        ::

            def inverted_to_text_og(dictionary):
                dictionary = json.loads(dictionary)
                tuples = []
                for k, vals in dictionary['InvertedIndex'].items():
                    for v in vals:
                        tuples.append((k, v))
                abstact_tuples = sorted(tuples, key = lambda x: x[1])
                abstract = ' '.join([x[0] for x in abstact_tuples])
                return abstract

    Remove non-English abstracts using spaCy.

        "We can add code here if we want."

    Conduct stratified train validation split w.r.t FOS.


Level 0&1 sample method (s1)
++++++++++++++++++++++++++++++++++

In level 0&1, there are in total 310 unique FOS. And we sample 10 papers for each FOS, which would give us 3100 papers in total.

::

    create table fos_modeling.sample_paper_level_01 as
    select a.paperid, a.fieldofstudyid, p3.indexedabstract 
    from (
        select p.paperid, p.fieldofstudyid, row_number() over (partition by p.fieldofstudyid order by p.score desc) as row_num 
        from paperfieldsofstudy p 
        left join paperabstractsinvertedindex p2 on p2.paperid = p.paperid
        where p.fieldofstudyid in (select f.fieldofstudyid from fieldsofstudy f where f.level in (0, 1))
        and p.score >= 0.45 and p2.paperid is not null) a
    left join paperabstractsinvertedindex p3 on p3.paperid = a.paperid
    where a.row_num <= 10;

Level 2 sample method (s2)
++++++++++++++++++++++++++++++++++

The sample method is the same with level 0&1. In level 2, there are 25,472 unique FOS. And simially, we sample 10 papers for each FOS, which would return 254,720 papers.

::

    create table fos_modeling.sample_paper_level_2 as
    select b.paperid, b.fos, b.indexedabstract 
    from(
        select a.paperid, a.fos, p2.indexedabstract, row_number() over (partition by a.fos order by a.score desc, a.paperid) as row_num
        from ( 
            select p.paperid, pl.fos, p.score
            from paperfieldsofstudy p 
            inner join fos_modeling.papercountperfos_l2_v2 pl on pl.fos = p.fieldofstudyid 
            where pl.paper_cnt >= 10 and p.score > 0.6 ) as a 
        inner join paperabstractsinvertedindex p2 on p2.paperid = a.paperid
            ) as b
    where b.row_num <= 10;


    create table fos_modeling.sample_paper_level_2_more as
    select b.paperid, b.fos, b.indexedabstract 
    from(
        select a.paperid, a.fos, p2.indexedabstract, row_number() over (partition by a.fos order by a.score desc, a.paperid) as row_num
        from ( 
            select p.paperid, pl.fos, p.score
            from paperfieldsofstudy p 
            inner join fos_modeling.papercountperfos_l2_v2 pl on pl.fos = p.fieldofstudyid 
            where pl.paper_cnt >= 10 and p.score > 0.6 ) as a 
        inner join paperabstractsinvertedindex p2 on p2.paperid = a.paperid
            ) as b
    where b.row_num > 10 and b.row_num <= 25;

Another level 2 sample method (s3)
++++++++++++++++++++++++++++++++++

According to the previous method, the sample data size of level 2 is still very large, which would slow down the training time and the result is not good as well. Due to the result of predictive accuracy for the training set, we think the model does not train sufficient on the training set. 

Hence we choose the FOS which corresponding to more than 500 papers with score >0.6. It will return 4,998 FOS and we pick 20 papers for each FOS.

::

    create table fos_modeling.sample_paper_level_2_500 as
    select b.paperid, b.fos, b.indexedabstract 
    from(
        select a.paperid, a.fos, p2.indexedabstract, row_number() over (partition by a.fos order by a.score desc, a.paperid) as row_num
        from ( 
            select p.paperid, pl.fos, p.score
            from paperfieldsofstudy p 
            inner join fos_modeling.papercountperfos_l2_v2 pl on pl.fos = p.fieldofstudyid 
            where pl.paper_cnt >= 500 and p.score > 0.6 ) as a 
        inner join paperabstractsinvertedindex p2 on p2.paperid = a.paperid
            ) as b
    where b.row_num <= 20;

Result
-----------

.. table:: Sample data description

    ==================    ==========   ======================     ====================
    File Name             FOS level    Number of FOS Classes      Train/Validation
    ==================    ==========   ======================     ====================
    sample_creation_s1    level 0&1    310                        8:2
    sample_creation_s2    level 2      25472                      7:3
    sample_creation_s3    level 2      4988                       7:3
    ==================    ==========   ======================     ====================






