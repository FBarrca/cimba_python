How-to guides
=============

Short, focused recipes. Each one assumes you know the basics from the
:doc:`../tutorial/index` and gets straight to the point.

.. list-table::
   :widths: 35 65

   * - :doc:`recorded_data`
     - Replay a recorded log in a model, and handle running out of data.
   * - :doc:`input_models`
     - Choose, fit and check an input model before trusting it.
   * - :doc:`comparing`
     - Compare scenarios and input models with paired statistics.
   * - :doc:`policies`
     - Swap decision logic with polymorphic ``@cb.function`` methods and
       compare policies in one experiment.
   * - :doc:`optimizing`
     - Tune parameter values, size the search batches, and read the chosen
       solution's independent estimate.
   * - :doc:`diagrams`
     - Draw a model's structure and its process interactions.
   * - :doc:`captures`
     - Keep raw histories and plot them.
   * - :doc:`debugging`
     - Find out why a trial failed, and reproduce it.
   * - :doc:`custom_sources`
     - Write your own resampling or generative input source.
   * - :doc:`performance`
     - Make big experiments fast and memory-friendly.

.. toctree::
   :hidden:

   recorded_data
   input_models
   comparing
   policies
   optimizing
   diagrams
   captures
   debugging
   custom_sources
   performance
