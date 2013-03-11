Last modified: 2012-09-18 17:13:16 EEST
-----------------------------------------------------------------

GPstuff: Gaussian process models for Bayesian analysis 3.4 (r1086)

Maintainers: Aki Vehtari <aki.vehtari@aalto.fi>
             Jarno Vanhatalo <jarno.vanhatalo@helsinki.fi>
             

This software is distributed under the GNU General Public Licence
(version 3 or later); please refer to the file Licence.txt,
included with the software, for details.


Table of contents:

1. INTRODUCTION
2. INSTALLING THE TOOLBOX
3. CONTENTS
4. TESTING THE INSTALLATION
5. USER GUIDE (VERY SHORT)
6. KNOWN PROBLEMS WITH INSTALLING SUITESPARSE AND SOLUTIONS TO THEM


    ------------------------------------------
1. INTRODUCTION

  GPstuff is a collection of Matlab functions to build and analyze
  Bayesian models build over Gaussian processes. The toolbox is tested
  with Matlab 7.9 in 64bit Windows and Linux environments (it should
  work in the 32bit versions as well but they are not tested
  properly).

  The code for GPstuff can be found from GPstuff folder and its
  subfolders. The SuiteSparse folder contains the SuiteSparse toolbox
  by Tim Davis:
  http://www.cise.ufl.edu/research/sparse/SuiteSparse/current/SuiteSparse/
  The SuiteSparse is needed when using compactly supported covariance
  functions.

2. INSTALLING THE TOOLBOX

  Some of the functions in GPstuff are implemented using C in order to
  make the computations faster. In order to use these functions you
  need to compile them first. There are two ways to do that:

  1) Basic installation without compactly supported covariance
     functions

  * Install the GPstuff package by running matlab_install in this
    folder

  * With this option you are able to use all the other functions
    except for gpcf_ppcs*


  2) Installation with compactly supported covariance functions
  
  Compactly supported (CS) covariance functions are functions that
  produce sparse covariance matrices (matrices with zero elements). To
  use these functions (gpcf_ppcs*) you need the sparse GP
  functionalities in GPstuff which are build over SuiteSparse
  toolbox. To take full advantage of the CS covariance functions
  install GPstuff by running matlab_install('SuiteSparseOn' ) in the
  present directory.

    The function matlab_install compiles the mex-files and prints on
    the screen, which directories should be added to Matlab paths. 
    
3. CONTENTS
   
   The GPstuff packge contains the following subdirectories:
   diag  dist  gp  mc  misc  optim  xunit

   Each folder contains Contents.m, which summarizes the functions
   in the folder. 

   From the above 'gp' folder contains the main functionalities and
   demonstration programs. Other folders contain additional help
   functions.

4. TESTING THE INSTALLATION

   Installation can be tested by running command test_all, which
   runs all demos and compares the computed results to pre-saved
   results. Running test_all takes about one hour and it requires
   that 'xunit' toolbox is in the Matlab path. xunit package can be
   downloaded from
   http://www.mathworks.com/matlabcentral/fileexchange/22846-matlab-xunit-test-framework

5. USER QUIDE (VERY SHORT)

   It easiest to learn to use the package by running the demos. It is
   advisable to open the demo files in text editor and run them line
   by line. The demos are documented so that user can follow what
   happens on each line.

   The basic structure of the program is as follows. The program
   consist of separate blocks, which are:

      Gaussian process model structure (GP):
                      This is a structure that contains all the
                      model information (see gp_init) and information
                      on, which inference scheme is used. 

                      GP structure contains covariance function
                      structures (GPCF_*) and likelihood structures
                      (LIKELIH_*). 

      Covariance function structure (GPCF):
                      This is a structure that contains all of the
                      covariance function information (see
                      e.g. gpcf_sexp). The structure contains the
                      hyperparameter values, pointers to nested
                      functions that are related to the covariance
                      function (e.g. function to evaluate covariance
                      matrix) and hyperprior structure.

      likelihood structure:
                      This is a structure that contains all of the
                      likelihood function information (see
                      e.g. likelih_probit). The structure contains the
                      likelihood parameter values and pointers to
                      nested functions that are related to the
                      likelihood function (e.g. log likelihood and its
                      derivatives).

      Inference utilities:
                      Inference utilities consist of functions that
                      are needed to make the posterior inference and
                      predictions. These include, among others,
		        GP_E - Evaluate conditional log posterior
 		               density
                        GP_G - Evaluate gradient of conditional log
                               posterior 
			EP_PRED - Predictions with Gaussian Process EP
                        GP_MC - Markov chain Monte Carlo sampling
