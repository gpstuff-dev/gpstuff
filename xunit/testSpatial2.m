function test_suite = testSpatial2
initTestSuite;


% Set random number stream so that failing isn't because randomness. Run
% demo & save test values.

function testDemo
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
disp('Running: demo_spatial2')
demo_spatial2
Ef = Ef(1:100);
Varf = Varf(1:100);
C = C(1:50, 1:50);
path = which('testSpatial2.m');
path = strrep(path,'testSpatial2.m', 'testValues/testSpatial2');
save(path, 'Ef', 'Varf', 'C');
RandStream.setDefaultStream(prevstream);
drawnow;clear;close all

% Compare test values to real values.

function testPredictionsEP
values.real = load('realValuesSpatial2.mat', 'Ef', 'Varf');
values.test = load('testValues/testSpatial2.mat', 'Ef', 'Varf');
assertElementsAlmostEqual(mean(values.test.Ef), mean(values.real.Ef), 'relative', 0.05);
assertElementsAlmostEqual(mean(values.test.Varf), mean(values.real.Varf), 'relative', 0.05);


function testCovarianceMatrix
values.real = load('realValuesSpatial2.mat', 'C');
values.test = load('testValues/testSpatial2.mat', 'C');
assertElementsAlmostEqual(mean(values.real.C), mean(values.test.C), 'relative', 0.05);