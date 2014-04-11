function test_suite = test_hierarchial

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_HIERARCHIAL
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
    disp('Running: demo_hierarchial')
    demo_hierarchial
    path = which('test_hierarchial');
    path = strrep(path,'test_hierarchial.m', 'testValues');
    if ~(exist(path, 'dir') == 7)
      mkdir(path)
    end
    path = strcat(path, '/testHierarchial');
    save(path, 'derivs');
    
    % Set back initial random stream
    setrandstream(prevstream);
    drawnow;clear;close all


% Test predictive mean, variance and density      
        
  function testDerivatesRegression
    values.real = load('realValuesHierarchial.mat','derivs');
    values.test = load(strrep(which('test_hierarchial.m'), 'test_hierarchial.m', 'testValues/testHierarchial.mat'),'derivs');
    assertElementsAlmostEqual(values.real.derivs{1}, (values.test.derivs{1}), 'relative', 0.05);
    assertElementsAlmostEqual((values.real.derivs{2}), (values.test.derivs{2}), 'relative', 0.05);
    
  function testDerivatesLaplace
    values.real = load('realValuesHierarchial.mat','derivs');
    values.test = load(strrep(which('test_hierarchial.m'), 'test_hierarchial.m', 'testValues/testHierarchial.mat'),'derivs');
    assertElementsAlmostEqual(values.real.derivs{3}, (values.test.derivs{3}), 'relative', 0.05);
    assertElementsAlmostEqual((values.real.derivs{4}), (values.test.derivs{4}), 'relative', 0.05);
    
  function testDerivatesEP
    values.real = load('realValuesHierarchial.mat','derivs');
    values.test = load(strrep(which('test_hierarchial.m'), 'test_hierarchial.m', 'testValues/testHierarchial.mat'),'derivs');
    assertElementsAlmostEqual(values.real.derivs{5}, (values.test.derivs{5}), 'relative', 0.05);
    assertElementsAlmostEqual((values.real.derivs{6}), (values.test.derivs{6}), 'relative', 0.05);


