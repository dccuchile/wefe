=============================================
Multilingual Gender Bias Measurement Examples
=============================================

The notebooks located in 
`multilingual examples folder <https://github.com/dccuchile/wefe/tree/master/examples/multilingual>`_ 
show how to measure gender bias in static embeddings from gender queries in different
languages.

The word embedding models in different languages are obtained from the flair library. 
The notebooks are self-contained: they contain everything necessary to load the
embeddings and execute the queries and allow flair to be installed in case it is not
already installed.

Available languages are English, Spanish, French, German, Italian, Spanish,
Swedish, Dutch.

.. warning::

    The words sets used in the notebooks were translated using google translator. 
    Therefore, it is possible that some concepts may have been mistranslated and may
    require some correction. The original English concepts can be loaded using 
    `load_weat <https://wefe.readthedocs.io/en/latest/generated/dataloaders/wefe.load_weat.html#wefe.load_weat>`_
    util. 
    Read the instructions carefully and use the notebooks and its results with caution!

