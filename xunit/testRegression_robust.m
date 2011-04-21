function test_suite = testRegression_robust
initTestSuite;

% Set random number stream so that failing isn't because randomness. Run
% demo & save test values.

function testDemo
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
disp('Running: demo_regression_robust')
demo_regression_robust
path = which('testRegression_robust.m');
path = strrep(path,'testRegression_robust.m', 'testValues/testRegression_robust');
save(path, 'Eft', 'Varft')
RandStream.setDefaultStream(prevstream);
drawnow;clear;close all

% Compare test values to real values.

function testPredictionEP
values.real = load('realValuesRegression_robust', 'Eft', 'Varft');
values.test = load('testValues/testRegression_robust', 'Eft', 'Varft');
assertElementsAlmostEqual(mean(values.real.Eft), mean(values.test.Eft), 'relative', 0.05);
assertElementsAlmostEqual(mean(values.real.Varft), mean(values.test.Varft), 'relative', 0.05);

