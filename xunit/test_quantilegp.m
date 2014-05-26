function test_suite = test_quantilegp

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_QUANTILEGP
%
% Copyright (c) 2011-2012 Ville Tolvanen

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

initTestSuite;


  function testDemo
    % Set random number stream so that the test failing isn't because
    % randomness. Run demo & save test values.
    
    disp('Running: demo_quantilegp')
    demo_quantilegp
    path = which('test_quantilegp');
    path = strrep(path,'test_quantilegp.m', 'testValues');
    if ~(exist(path, 'dir') == 7)
      mkdir(path)
    end
    path = strcat(path, '/testQuantilegp');
    save(path, 'Ef1','Ef2','Ef3','Ef4','Ef5','Varf1','Varf2','Varf3','Varf4','Varf5');
    
    drawnow;clear;close all


% Test predictive mean and variance
  
  function testPredictiveMeanAndVariance_EP
    values.real = load('realValuesQuantilegp.mat', 'Ef4','Ef5','Varf4','Varf5');
    values.test = load(strrep(which('test_quantilegp.m'), 'test_quantilegp.m', 'testValues/testQuantilegp.mat'), ...
      'Ef4','Ef5','Varf4','Varf5');
    tol = 0.10; % Tolerance
    assertElementsAlmostEqual(values.real.Ef4, values.test.Ef4, 'relative', tol);
    assertElementsAlmostEqual(values.real.Varf4, values.test.Varf4, 'relative', tol);
    assertElementsAlmostEqual(values.real.Ef5, values.test.Ef5, 'relative', tol);
    assertElementsAlmostEqual(values.real.Varf5, values.test.Varf5, 'relative', tol);

  function testPredictiveMeanAndVariance_MCMC
    values.real = load('realValuesQuantilegp.mat', 'Ef1','Ef2','Ef3');
    values.test = load(strrep(which('test_quantilegp.m'), 'test_quantilegp.m', 'testValues/testQuantilegp.mat'), ...
      'Ef1','Ef2','Ef3');
    tol = 0.10; % Tolerance
    % Here the results are compared in L2-norm sense in order to be more
    % reasonable (using assertVectorsAlmostEqual). This type of comparison
    % should be used in the other asserts also.
    assertVectorsAlmostEqual(values.real.Ef1, values.test.Ef1, 'relative', tol);
    %assertVectorsAlmostEqual(values.real.Varf1, values.test.Varf1, 'relative', tol);
    assertVectorsAlmostEqual(values.real.Ef2, values.test.Ef2, 'relative', tol);
    %assertVectorsAlmostEqual(values.real.Varf2, values.test.Varf2, 'relative', tol);
    assertVectorsAlmostEqual(values.real.Ef3, values.test.Ef3, 'relative', tol);
    %assertVectorsAlmostEqual(values.real.Varf3, values.test.Varf3, 'relative', tol);

    