function test_suite = test_monotonic2

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_MONOTONIC2
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
    disp('Running: demo_monotonic2')
    demo_monotonic2
    path = which('test_monotonic2');
    path = strrep(path,'test_monotonic2.m', 'testValues');
    if ~(exist(path, 'dir') == 7)
      mkdir(path)
    end
    path = strcat(path, '/testMonotonic2');
    save(path, 'Eft', 'Varft', 'Eftm', 'Varftm');
    
    % Set back initial random stream
    setrandstream(prevstream);
    drawnow;clear;close all


% Test predictive mean and variance for non-monotonic and monotonic model 
% with 5% tolerance.
  
  function testPredictiveMeanAndVariance
    values.real = load('realValuesMonotonic2.mat','Eft','Varft','Eftm','Varftm');
    values.test = load(strrep(which('test_monotonic2.m'), 'test_monotonic2.m', 'testValues/testMonotonic2.mat'),'Eft','Varft','Eftm','Varftm');
    assertElementsAlmostEqual((values.real.Eft), (values.test.Eft), 'relative', 0.05);
    assertElementsAlmostEqual((values.real.Varft), (values.test.Varft), 'relative', 0.05);
    assertElementsAlmostEqual((values.real.Eftm), (values.test.Eftm), 'relative', 0.05);
    assertElementsAlmostEqual((values.real.Varftm), (values.test.Varftm), 'relative', 0.05);
    


