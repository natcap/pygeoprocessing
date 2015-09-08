=================================
Running tests for pygeoprocessing
=================================

Tests for pygeoprocessing are located in pygeoprocessing/tests
and in tests/.  Tests that require binary data will be skipped 
unless the appropriate SVN repository can be found.


Nosetests
^^^^^^^^^

.. code-block:: shell:

    $ python setup.py nosetests

.. code-block:: shell:

    $ nosetests

Python
^^^^^^

.. code-block:: python:

    >>> import pygeoprocessing
    >>> pygeoprocessing.test()

Tox
^^^

.. note::
    Calling ``tox`` will also fetch svn sampledata.

.. code-block:: shell:

    $ tox

To get binary test data
-----------------------
.. code-block:: shell

    $ python tests/get_data.py


