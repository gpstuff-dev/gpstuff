function test_suite = test_regression_meanf
initTestSuite;

% Set random number stream so that test failing isn't because randomness.
% Run demo & save test values.

function testDemo
stream0 = RandStream('mt19937ar','Seed',0);
stream = RandStream.setDefaultStream(stream0);
disp('Running: demo_regression_meanf')
demo_regression_meanf
path = which('test_regression_meanf.m');
path = strrep(path,'test_regression_meanf.m', 'testValues');
if ~(exist(path, 'dir') == 7)
    mkdir(path)
end
path = strcat(path, '/testRegression_meanf'); 
save(path, 'Eft', 'Varft');
RandStream.setDefaultStream(stream);
drawnow;clear;close all


function testPredictions
values.real = load('realValuesRegression_meanf.mat', 'Eft', 'Varft');
values.test = load(strrep(which('test_regression_meanf.m'), 'test_regression_meanf.m', 'testValues/testRegression_meanf.mat'), 'Eft', 'Varft');
assertElementsAlmostEqual(mean(values.real.Eft), mean(values.test.Eft), 'relative', 0.10);
assertElementsAlmostEqual(mean(values.real.Varft), mean(values.test.Varft), 'relative', 0.10);

