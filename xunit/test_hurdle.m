function test_suite = test_hurdle

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_HURDLE
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
    disp('Running: demo_hurdle')
    demo_hurdle
    path = which('test_hurdle');
    path = strrep(path,'test_hurdle.m', 'testValues');
    if ~(exist(path, 'dir') == 7)
      mkdir(path)
    end
    path = strcat(path, '/testHurdle');
    save(path, 'Efz','Varfz','Efc','Varfc');
    
    % Set back initial random stream
    setrandstream(prevstream);
    drawnow;clear;close all


% Test predictive mean and variance with 5% tolerance.
  
  function testPredictiveMeanAndVariance_LatentZeroProcess
    values.real = load('realValuesHurdle.mat','Efz','Varfz');
    values.test = load(strrep(which('test_hurdle.m'), 'test_hurdle.m', 'testValues/testHurdle.mat'),'Efz','Varfz');
    assertElementsAlmostEqual(values.real.Efz, values.test.Efz, 'relative', 0.05);
    assertElementsAlmostEqual(values.real.Varfz, values.test.Varfz, 'relative', 0.05);

  function testPredictiveMeanAndVariance_LatentCountProcess
    values.real = load('realValuesHurdle.mat','Efc','Varfc');
    values.test = load(strrep(which('test_hurdle.m'), 'test_hurdle.m', 'testValues/testHurdle.mat'),'Efc','Varfc');
    assertElementsAlmostEqual(values.real.Efc, values.test.Efc, 'relative', 0.05);
    assertElementsAlmostEqual(values.real.Varfc, values.test.Varfc, 'relative', 0.05);
