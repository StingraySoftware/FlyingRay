FlyingRay
=========

**An interactive dashboard for the analysis and on-the-fly visualization of X-ray binary data, built with the Stingray library.**

FlyingRay is an interactive database and quick-look dashboard for X-ray observations, enabling the on-the-fly visualization of scientific data products. The system provides a unified interface to store, analyze, and organize key data products from multiple missions, allowing the astrophysics research community to efficiently track and predict the evolution of black hole binaries. By abstracting away the complexities of mission-specific data processing, FlyingRay simplifies the analysis workflow for researchers. Combining the use of the **Panel**, **Stingray**, **Plotly**, and **Matplotlib** packages, FlyingRay enables researchers to take full advantage of advanced timing analysis without needing to be well-versed in the underlying instrument-specific software.

Setup Instructions
-----------------

Step 1: Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by cloning the project from the GitHub repository:

.. code-block:: bash

   git clone https://github.com/StingraySoftware/FlyingRay.git
   cd FlyingRay

Step 2: Create and Activate a Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure you're using a compatible version of Python (e.g., Python 3.12).

.. code-block:: bash

   # Create a virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate

Step 3: Install Required Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This single command will install your project and all of its Python dependencies listed in the ``setup.py`` file.

.. code-block:: bash

   pip install -e .

Step 4: Initialize HEASoft & Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the same terminal, initialize your HEASoft environment and then start the dashboard.

.. code-block:: bash

   # Initialize HEASoft
   heainit

   # Run the dashboard
   panel serve flyingray.py --autoreload --static-dirs assets=assets

Navigate to **http://localhost:5006** in your browser, as the application will be available at this address.

Contributing to FlyingRay
------------------------

Reporting Bugs & Feature Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter a bug or have an idea for a new feature, please report it in the `GitHub Issues <https://github.com/your-username/FlyingRay/issues>`_ section. Describe how to reproduce the bug, including the mission, OBSID, and any custom flags used. We also welcome pull requests for bug fixes.

Support Channels
~~~~~~~~~~~~~~~~

For questions or direct contact, you can reach out through the following channels:

* **GitHub Issues:** For all bug reports and feature requests
* **Email:** `adnanmoahammmad6002@gmail.com <mailto:adnanmoahammmad6002@gmail.com>`_
* **Stingray Slack:** Find me with the username **@Adnan**

License and Acknowledgments
--------------------------

License
~~~~~~~

The FlyingRay dashboard is licensed under the terms of the MIT license. See the ``LICENSE`` file for details.

Credits & Acknowledgments
~~~~~~~~~~~~~~~~~~~~~~~~~

FlyingRay is built upon and depends on several powerful open-source projects. We gratefully acknowledge their developers:

* **Stingray Library:** For core astronomical time-series analysis
* **heasarc_retrieve_pipeline:** For programmatic data retrieval and processing
* **Panel & HoloViz:** For the interactive dashboard framework and GUI
* **Plotly & Matplotlib:** For generating interactive and static visualizations
* **Prefect:** For workflow orchestration

Acknowledgments
~~~~~~~~~~~~~~~

* The development teams of Stingray, HoloViz, and Prefect
* The broader X-ray astronomy and open-source scientific Python communities
* Matteo Bachetti for his work on the FAD dead-time correction code and the HEASARC retrieval pipeline