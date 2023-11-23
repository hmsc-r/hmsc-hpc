====
HMSC
====

Hierarchical modelling of species communicaties (HMSC) is a flexible framework for joint species distribution modelling (JSDM).

Program setup
-------------

#.	Clone hmsc-hpc repository

        .. code-block:: sh
    
	        git clone https://github.com/aniskhan25/hmsc-hpc.git
		cd hmsc-hpc

#.	Create and activate a virtual environment

	.. code-block:: sh

		python -m venv venv
		source venv/bin/activate
		export PYTHONPATH=$PYTHONPATH:$(pwd)

#.	Install dependences

	.. code-block:: sh

	        pip install --upgrade pip
		pip install -r requirements_dev.txt

Program options
---------------
   
.. code-block:: sh

	python hmsc/run_gibbs_sampler.py [ --help ] 
	   [ --samples n ] [ --transient n ] [ --thin n ] [ --chains n ]
	   [ --input path ] [ --output path ]
	   [ --tnlib ] [ --verbose ] [ --fse ] [ --profile ]


Input arguments
---------------

.. role:: bash(code)
   :language: bash
   
*	:bash:`-s [ --samples ] n`
  		number of samples obtained per chains

* 	:bash:`-b [ --transient ] n`
		number of samples discarded before recording posterior samples
    
* 	:bash:`-t [ --thin ] n`
		number of samples between each recording of posterior samples
    
* 	:bash:`-c [ --chains ] n`
	indices of chains to fit
    
* 	:bash:`-i [ --input ] path`
		input RDS file with parameters for model initialization",
    
* 	:bash:`-o [ --output ] path`
		output RDS file with recorded posterier samples",
    
* 	:bash:`-v [ --verbose ]`
		print out information meassages and progress status
    
* 	:bash:`--tnlib`
		which library is used for sampling trunacted normal: scipy, tf or tfd
    
* 	:bash:`--fse`
		whether to save Eta posterior
    
* 	:bash:`--profile`
		whether to run profiler alongside sampling

Example usage
-------------

.. code-block:: sh

	python hmsc/run_gibbs_sampler.py 
   	   --input <input path>
	   --output <output path> 
	   --samples 25 --transient 0 --thin 1
