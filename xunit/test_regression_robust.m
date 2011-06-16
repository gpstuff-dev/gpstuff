function test_suite = test_regression_robust
initTestSuite;

% Set random number stream so that failing isn't because randomness. Run
% demo & save test values.

function testDemo
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
disp('Running: demo_regression_robust')
demo_regression_robust
path = which('test_regression_robust.m');
path = strrep(path,'test_regression_robust.m', 'testValues');
if ~(exist(path, 'dir') == 7)
    mkdir(path)
end
path = strcat(path, '/testRegression_robust'); 
save(path, 'Eft', 'Varft')
RandStream.setDefaultStream(prevstream);
drawnow;clear;close all

% Compare test values to real values.

function testPredictionEP
values.real = load('realValuesRegression_robust', 'Eft', 'Varft');
values.test = load(strrep(which('test_regression_robust.m'), 'test_regression_robust.m', 'testValues/testRegression_robust'), 'Eft', 'Varft');
assertElementsAlmostEqual(mean(values.real.Eft), mean(values.test.Eft), 'relative', 0.05);
assertElementsAlmostEqual(mean(values.real.Varft), mean(values.test.Varft), 'relative', 0.05);

