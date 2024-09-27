PopFrame
==========

.. logo-start

.. figure:: https://i.ibb.co/8bD3qr0/DALL-E-2024-05-22-16-49.png
   :alt: PopFrame logo

.. logo-end

|PythonVersion| |Black|

.. description-start

**PopFrame** is an open source library that includes methods for modeling the framework of a regional-level settlement system for assessing territories subject to reclamation/renovation, as well as modeling scenarios for changing regional facilities. The library is designed to form a universal information model of the region based on localities. 

.. description-end

Table of Contents
--------------------

- `Core features <Core features_>`_
- `Installation <Installation_>`_
- `Examples <Examples_>`_
- `Project Structure <Project Structure_>`_
- `Documentation <Documentation_>`_
- `Developing <Developing_>`_
- `License <License_>`_
- `Acknowledgments <Acknowledgments_>`_
- `Contacts <Contacts_>`_

Core features
-------------

.. features-start

The library also provides tools for working with the information model of the region, which allow:

-  Calculate indicators (population, birth rate, mortality) by municipal districts and municipalities.
-  Evaluate territories in relation to the framework of the settlement system.
-  Calculating the level of urbanization of the territory.
-  The construction of agglomerations based on the framework of the settlement system

.. features-end

Installation
------------

.. installation-start

**PopFrame** can be installed with ``pip``:

::

   pip install popframe

.. installation-end

Examples
------------
Describe examples how it should work and should be used.
Images, GIFs and code cells are welcome.


Project Structure
-----------------

The latest version of the library is available in the ``main`` branch.

The repository includes the following directories and modules:

-  `popframe <https://github.com/Mvin8/PopFrame/tree/main?tab=readme-ov-file>`__
   - directory with the library code:

   -  preprocessing - data preprocessing module
   -  models - entities' classes used in library
   -  method - library tool methods based on ``Region`` model
   -  utils - module containing utulity functions and consts

-  `tests <https://github.com/Mvin8/PopFrame/tree/main/tests>`__
   ``pytest`` testing
-  `examples <https://github.com/Mvin8/PopFrame/tree/main/examples>`__
   examples of how methods work
-  `docs <https://github.com/Mvin8/PopFrame/tree/main/docs>`__ -
   documentation sources


Documentation
-------------

Detailed information and description of BlocksNet is available in
`documentation <https://mvin8.github.io/PopFrame/>`__.


Developing
----------

.. developing-start

To start developing the library, one must perform following actions:

1. Clone the repository:
   ::

       $ git clone https://github.com/Mvin8/PopFrame

2. (Optional) Create a virtual environment as the library demands exact package versions:
   ::

       $ make venv

   Activate the virtual environment if you created one:
   ::

       $ source .venv/bin/activate

3. Install the library in editable mode with development dependencies:
   ::

       $ make install-dev

4. Install pre-commit hooks:
   ::

       $ pre-commit install

5. Create a new branch based on ``develop``:
   ::

       $ git checkout -b develop <new_branch_name>

6. Start making changes on your newly created branch, remembering to
   never work on the ``master`` branch! Work on this copy on your
   computer using Git to do the version control.

7. Update
   `tests <https://github.com/Mvin8/PopFrame/tree/main/tests>`__
   according to your changes and run the following command:

   ::

         $ make test

   Make sure that all tests pass.

8. Update the
   `documentation <https://github.com/Mvin8/PopFrame/tree/main/docs>`__
   and **README** according to your changes.

11. When you're done editing and local testing, run:

   ::

         $ git add modified_files
         $ git commit

   to record your changes in Git, then push them to GitHub with:

   ::

            $ git push -u origin my-contribution

   Finally, go to the web page of your fork of the BlocksNet repo, and click
   'Pull Request' (PR) to send your changes to the maintainers for review.

.. developing-end

Check out the...


License
-------

The project has `BSD-3-Clause license <./LICENSE>`__

Acknowledgments
---------------

.. acknowledgments-start

The library was developed as the main part of the ITMO University
project...


Contacts
--------

.. contacts-start

You can contact us:

-  `NCCR <https://actcognitive.org/o-tsentre/kontakty>`__ - National
   Center for Cognitive Research
-  `IDU <https://idu.itmo.ru/en/contacts/contacts.htm>`__ - Institute of
   Design and Urban Studies
-  `Maksim Natykin <https://t.me/Mvin98>`__ - lead software engineer

.. contacts-end





