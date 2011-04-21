function test_suite = testRegression_meanf
initTestSuite;

% Set random number stream so that test failing isn't because randomness.
% Run demo & save test values.

function testDemo
stream0 = RandStream('mt19937ar','Seed',0);
stream = RandStream.setDefaultStream(stream0);
disp('Running: demo_regression_meanf')
demo_regression_meanf
path = which('testRegression_meanf.m');
path = strrep(path,'testRegression_meanf.m', 'testValues/testRegression_meanf');
save(path, 'Eft', 'Varft');
RandStream.setDefaultStream(stream);
drawnow;clear;close all


function testPredictions
values.real = load('realValuesRegression_meanf.mat', 'Eft', 'Varft');
values.test = load('testValues/testRegression_meanf.mat', 'Eft', 'Varft');
assertElementsAlmostEqual(mean(values.real.Eft), mean(values.test.Eft), 'relative', 0.05);
assertElementsAlmostEqual(mean(values.real.Varft), mean(values.test.Varft), 'relative', 0.05);

