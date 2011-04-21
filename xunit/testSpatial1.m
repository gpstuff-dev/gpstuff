function test_suite = testSpatial1
initTestSuite;


% Set random number stream so that failing isn't because randomness. Run
% demo & save test values.

function testDemo
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
disp('Running: demo_spatial1')
demo_spatial1
Ef = Ef(1:100);
Varf = Varf(1:100);
path = which('testSpatial1.m');
path = strrep(path,'testSpatial1.m', 'testValues/testSpatial1');
save(path, 'Elth', 'Elth2', 'Ef', 'Varf');
RandStream.setDefaultStream(prevstream);
drawnow;clear;close all

% Compare test values to real values.

function testEstimatesIA
values.real = load('realValuesSpatial1.mat', 'Elth', 'Elth2');
values.test = load('testValues/testSpatial1.mat', 'Elth', 'Elth2');
assertElementsAlmostEqual(values.real.Elth, values.test.Elth, 'relative', 0.01);
assertElementsAlmostEqual(values.real.Elth2, values.test.Elth2, 'relative', 0.01);


function testPredictionIA
values.real = load('realValuesSpatial1.mat', 'Ef', 'Varf');
values.test = load('testValues/testSpatial1.mat', 'Ef', 'Varf');
assertElementsAlmostEqual(mean(values.real.Ef), mean(values.test.Ef), 'relative', 0.05);
assertElementsAlmostEqual(mean(values.real.Varf), mean(values.test.Varf), 'relative', 0.05);

