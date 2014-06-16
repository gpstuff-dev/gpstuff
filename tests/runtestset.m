function log = runtestset(mode)
%RUNTESTSET  Run a set of tests in GPstuff.
%
%  Description
%    LOG = RUN_GP_TESTS(MODE) runs a set of tests in tests folder and
%    returns the test log. Each of the tests runs corresponding demo
%    and compares the values to previously saved ones for errors. This is
%    useful e.g. in case user modifies functions provided by GPstuff.
%
%    Possible values for parameter MODE:
%      'fast' - runs a selection of test taking about one hour (default)
%      'all'  - runs all the tests
%
%    Can be used with both xUnit Test Framework package by Steve Eddins
%    and the built-in Unit Testing Framework.
%
%  See also
%    Readme.txt
%
% Copyright (c) 2011-2012 Ville Tolvanen
% Copyright (c) 2014 Tuomas Sivula

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 1
  mode = 'fast';
end

% Define the test to run on fast mode
fast_tests = {'test_binomial1' ...
'test_binomial_apc' ...
'test_classific' ...
'test_derivativeobs' ...
'test_hierprior' ...
'test_hurdle' ...
'test_improvemarginals' ...
'test_kalman1' ...
'test_kalman2' ...
'test_lgcp' ...
'test_loopred' ...
'test_memorysave' ...
'test_modelassesment1' ...
'test_monotonic2' ...
'test_multiclass_nested_ep' ...
'test_multinom' ...
'test_neuralnetcov' ...
'test_periodic' ...
'test_quantilegp' ...
'test_regression1' ...
'test_regression_additive1' ...
'test_regression_additive2' ...
'test_regression_hier' ...
'test_regression_meanf' ...
'test_regression_robust' ...
'test_regression_sparse1' ...
'test_regression_sparse2' ...
'test_spatial1' ...
'test_survival_aft' ...
'test_survival_coxph' ...
'test_zinegbin'}';

% Path to the unittest folder
path = mfilename('fullpath');
path = path(1:end-length(mfilename));

% Check which test package to use
runtests_path = which('runtests');
if exist([runtests_path(1:end-10) 'initTestSuite.m'], 'file')
  
  % xUnit Test Framework
  if strcmp(mode, 'all')
    % Add all the tests
    suite = TestSuite.fromName(path);
  elseif strcmp(mode, 'fast')
    % Add the selected tests
    suite = TestSuite;
    for i = 1:length(fast_tests)
      suite.add(TestSuite(fast_tests{i}));
    end
  else
    error('Unsupported mode')
  end
  % Run the tests
  log = TestRunLogger;
  suite.run(log);
  
else
  
  % Built-in test framework
  if strcmp(mode, 'all')
    % Run all the tests
    log = runtests(path);
  elseif strcmp(mode, 'fast')
    % Run the selected tests
    log = runtests(fast_tests);
  else
    error('Unsupported mode')
  end
  
end
