function test_suite = test_kalman1

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_ZINEGBIN

% Copyright (c) 2011-2012 Ville Tolvanen

initTestSuite;

function testDemo
% Set random number stream so that failing isn't because randomness. Run
% demo & save test values.
prevstream=setrandstream(0);

disp('Running: demo_kalman1')
demo_kalman1;
path = which('test_kalman1.m');
path = strrep(path,'test_kalman1.m', 'testValues');
if ~(exist(path, 'dir') == 7)
    mkdir(path)
end
path = strcat(path, '/testKalman1'); 
save(path, 'Ef', 'Varf', 'Covft');

% Set back initial random stream
setrandstream(prevstream);
drawnow;clear;close all

% Compare test values to real values.

function testPredictionsKalman1
values.real = load('realValuesKalman1', 'Ef', 'Varf','Covft');
values.test = load(strrep(which('test_kalman1.m'), 'test_kalman1.m', 'testValues/testKalman1'), 'Ef', 'Varf','Covft');
assertElementsAlmostEqual(values.real.Ef, values.test.Ef, 'relative', 0.10);
assertElementsAlmostEqual(values.real.Varf, values.test.Varf, 'relative', 0.10);
assertElementsAlmostEqual(values.real.Covft, values.test.Covft, 'relative', 0.10);
