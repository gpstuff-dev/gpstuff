function test_suite = test_kalman2

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

disp('Running: demo_kalman2')
demo_kalman2;
path = which('test_kalman2.m');
path = strrep(path,'test_kalman2.m', 'testValues');
if ~(exist(path, 'dir') == 7)
    mkdir(path)
end
path = strcat(path, '/testKalman2'); 
save(path, 'Eft', 'Varft');

% Set back initial random stream
setrandstream(prevstream);
drawnow;clear;close all

% Compare test values to real values.

function testPredictionsKalman2
values.real = load('realValuesKalman2', 'Eft', 'Varft');
values.test = load(strrep(which('test_kalman2.m'), 'test_kalman2.m', 'testValues/testKalman2'), 'Eft', 'Varft');
assertElementsAlmostEqual(values.real.Eft, values.test.Eft, 'relative', 0.10);
assertElementsAlmostEqual(values.real.Varft, values.test.Varft, 'relative', 0.10);
