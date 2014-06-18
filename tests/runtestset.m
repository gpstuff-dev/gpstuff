function log = runtestset(mode)
%RUNTESTSET  Run a set of tests in GPstuff.
%
%  Description
%    LOG = RUN_GP_TESTS(MODE) runs a collection of tests in tests folder
%    and returns the test log. Each of the tests runs corresponding demo
%    and compares the values to previously saved ones for errors. This is
%    useful e.g. in case user modifies functions provided by GPstuff.
%
%    Possible values for parameter MODE:
%      'fast' - runs a selection of test taking about 2.5 min (default)
%      'hour' - runs a selection of test taking about one hour
%      'all'  - runs all the tests taking about 3.7 hours
%
%    Can be used with both xUnit Test Framework package by Steve Eddins
%    and the built-in Unit Testing Framework.
%
%    N.B. The time estimates assumes that SuiteSparse is used. The
%    estimated runtime of the individual demos (min):
%       binomial1              0.0570
%       binomial2              2.8533
%       binomial_apc           0.2244
%       classific              1.0576
%       derivativeobs          0.0201
%       hierprior              2.7335
%       hurdle                 2.2864
%       improvemarginals       0.0426
%       kalman1                0.0442
%       kalman2                6.2701
%       lgcp                   0.0692
%       loopred                1.0173
%       memorysave             0.7269
%       modelassesment1       12.8916
%       modelassesment2       45.4903
%       monotonic2             0.4059
%       multiclass            42.3722
%       multiclass_nested_ep   1.0884
%       multinom               1.0269
%       neuralnetcov           0.0757
%       periodic               0.3800
%       quantilegp             2.8618
%       regression1            1.0282
%       regression_additive1   0.2578
%       regression_additive2   0.1123
%       regression_hier        0.4091
%       regression_meanf       0.0147
%       regression_ppcs       58.4807
%       regression_robust      1.3518
%       regression_sparse1     0.2732
%       regression_sparse2     0.0444
%       spatial1               4.4727
%       spatial2               8.2087
%       survival_aft           6.5190
%       survival_coxph        15.8717
%       zinegbin               1.7628
%
%  See also
%    Readme.txt

% Copyright (c) 2011-2012 Ville Tolvanen
% Copyright (c) 2014 Tuomas Sivula

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 1
  mode = 'fast';
end

% Define tests to run on fast mode
fast_tests = {'test_binomial1' ...
'test_binomial_apc' ...
'test_derivativeobs' ...
'test_improvemarginals' ...
'test_kalman1' ...
'test_lgcp' ...
'test_monotonic2' ...
'test_neuralnetcov' ...
'test_periodic' ...
'test_regression_additive1' ...
'test_regression_additive2' ...
'test_regression_hier' ...
'test_regression_meanf' ...
'test_regression_sparse1' ...
'test_regression_sparse2'}';

% Define tests to run on hour mode
hour_tests = {'test_binomial1' ...
'test_binomial2' ...
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
'test_spatial2' ...
'test_survival_aft' ...
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
  elseif strcmp(mode, 'hour')
    % Add the selected tests
    suite = TestSuite;
    for i = 1:length(hour_tests)
      suite.add(TestSuite(hour_tests{i}));
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
  elseif strcmp(mode, 'hour')
    % Run the selected tests
    log = runtests(hour_tests);
  else
    error('Unsupported mode')
  end
  
end
