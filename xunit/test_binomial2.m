function test_suite = test_binomial2

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_BINOMIAL2
%
% Copyright (c) 2011-2012 Ville Tolvanen

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

initTestSuite;


  function testDemo
    % Set random number stream so that the test failing isn't because
    % randomness. Run demo & save test values.
    prevstream=setrandstream(0);    
    disp('Running: demo_binomial2')
    demo_binomial2
    path = which('test_binomial2');
    path = strrep(path,'test_binomial2.m', 'testValues');
    if ~(exist(path, 'dir') == 7)
      mkdir(path)
    end
    path = strcat(path, '/testBinomial2');
    save(path, 'pbetapositive_la','Eft_la','Varft_la','Eft_la_sampled','Varft_la_sampled', ...
      'pbetapositive_ep','Eft_ep','Varft_ep','Eft_ep_sampled','Varft_ep_sampled', ...
      'pbetapositive_mcmc','Eft_mcmc');
    
    % Set back initial random stream
    setrandstream(prevstream);
    drawnow;clear;close all


% Test predictive mean and variance
  
  function testPredictiveMeanAndVariance_Laplace
    values.real = load('realValuesBinomial2.mat', ...
      'pbetapositive_la','Eft_la','Varft_la','Eft_la_sampled','Varft_la_sampled');
    values.test = load(strrep(which('test_binomial2.m'), 'test_binomial2.m', 'testValues/testBinomial2.mat'), ...
      'pbetapositive_la','Eft_la','Varft_la','Eft_la_sampled','Varft_la_sampled');
    tol = 0.05; % Tolerance
    assertElementsAlmostEqual(values.real.pbetapositive_la, values.test.pbetapositive_la, 'relative', tol);
    assertElementsAlmostEqual(values.real.Eft_la, values.test.Eft_la, 'relative', tol);
    assertElementsAlmostEqual(values.real.Varft_la, values.test.Varft_la, 'relative', tol);
    assertElementsAlmostEqual(values.real.Eft_la_sampled, values.test.Eft_la_sampled, 'relative', tol);
    assertElementsAlmostEqual(values.real.Varft_la_sampled, values.test.Varft_la_sampled, 'relative', tol);

  function testPredictiveMeanAndVariance_EP
    values.real = load('realValuesBinomial2.mat', ...
      'pbetapositive_ep','Eft_ep','Varft_ep','Eft_ep_sampled','Varft_ep_sampled');
    values.test = load(strrep(which('test_binomial2.m'), 'test_binomial2.m', 'testValues/testBinomial2.mat'), ...
      'pbetapositive_ep','Eft_ep','Varft_ep','Eft_ep_sampled','Varft_ep_sampled');
    tol = 0.05; % Tolerance
    assertElementsAlmostEqual(values.real.pbetapositive_ep, values.test.pbetapositive_ep, 'relative', tol);
    assertElementsAlmostEqual(values.real.Eft_ep, values.test.Eft_ep, 'relative', tol);
    assertElementsAlmostEqual(values.real.Varft_ep, values.test.Varft_ep, 'relative', tol);
    assertElementsAlmostEqual(values.real.Eft_ep_sampled, values.test.Eft_ep_sampled, 'relative', tol);
    assertElementsAlmostEqual(values.real.Varft_ep_sampled, values.test.Varft_ep_sampled, 'relative', tol);

  function testPredictiveMeanAndVariance_MCMC
    values.real = load('realValuesBinomial2.mat', ...
      'pbetapositive_mcmc','Eft_mcmc');
    values.test = load(strrep(which('test_binomial2.m'), 'test_binomial2.m', 'testValues/testBinomial2.mat'), ...
      'pbetapositive_mcmc','Eft_mcmc');
    tol = 0.10; % Tolerance
    assertElementsAlmostEqual(values.real.pbetapositive_mcmc, values.test.pbetapositive_mcmc, 'relative', 0.02);
    
    % Here the results are compared in L2-norm sense in order to be more
    % reasonable (using assertVectorsAlmostEqual). This type of comparison
    % should be used in the other asserts also.
    assertVectorsAlmostEqual(values.real.Eft_mcmc, values.test.Eft_mcmc, 'relative', tol);

    