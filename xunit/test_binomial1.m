function test_suite = test_binomial1

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_BINOMIAL1
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
    disp('Running: demo_binomial1')
    demo_binomial1
    path = which('test_binomial1');
    path = strrep(path,'test_binomial1.m', 'testValues');
    if ~(exist(path, 'dir') == 7)
      mkdir(path)
    end
    path = strcat(path, '/testBinomial1');
    save(path, 'Eyt_la', 'Varyt_la', 'lpyt_la');
    
    % Set back initial random stream
    setrandstream(prevstream);
    drawnow;clear;close all


% Test predictive mean, variance and density for binomial model with 10% tolerance.
        
  function testPredictiveMeanAndVariance
    values.real = load('realValuesBinomial1.mat','Eyt_la','Varyt_la');
    values.test = load(strrep(which('test_binomial1.m'), 'test_binomial1.m', 'testValues/testBinomial1.mat'),'Eyt_la','Varyt_la');
    assertElementsAlmostEqual((values.real.Eyt_la), (values.test.Eyt_la), 'relative', 0.1);
    assertElementsAlmostEqual((values.real.Varyt_la), (values.test.Varyt_la), 'relative', 0.1);


  function testPredictiveDensity
    values.real = load('realValuesBinomial1.mat','lpyt_la');
    values.test = load(strrep(which('test_binomial1.m'), 'test_binomial1.m', 'testValues/testBinomial1.mat'),'lpyt_la');
    assertElementsAlmostEqual((values.real.lpyt_la), (values.test.lpyt_la), 'relative', 0.1);


