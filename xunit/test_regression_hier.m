function test_suite = test_regression_hier
initTestSuite;

% Set random number stream so that test failing isn't because randomness.
% Run demo & save test values.

function testDemo
stream0 = RandStream('mt19937ar','Seed',0);
RandStream.setDefaultStream(stream0)
disp('Running: demo_regression_hier')
demo_regression_hier
path = which('test_regression_hier.m');
path = strrep(path,'test_regression_hier.m', 'testValues/testRegression_hier');
save(path, 'Eff');
drawnow;clear;close all


function testPredictionMissingData
values.real = load('realValuesRegression_hier', 'Eff');
values.test = load('testValues/testRegression_hier', 'Eff');
assertVectorsAlmostEqual(mean(values.real.Eff), mean(values.test.Eff), 'relative', 0.01);

