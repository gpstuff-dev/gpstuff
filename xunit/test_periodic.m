function test_suite = test_periodic
initTestSuite;

% Set random number stream so that test failing isn't because randomness.
% Run demo & save test values.

function testDemo
stream0 = RandStream('mt19937ar','Seed',0);
RandStream.setDefaultStream(stream0)
disp('Running: demo_periodic')
demo_periodic
path = which('test_periodic.m');
path = strrep(path,'test_periodic.m', 'testValues');
if ~(exist(path, 'dir') == 7)
    mkdir(path)
end
path = strcat(path, '/testPeriodic'); 
save(path, 'Eft_full1', 'Varft_full1', 'Eft_full2', 'Varft_full2', ...
     'Eft_full', 'Varft_full');
drawnow;clear;close all

% Compare test values to real values.

function testPredictionsMaunaLoa
values.real = load('realValuesPeriodic.mat', 'Eft_full1', 'Eft_full2','Varft_full1','Varft_full2');
values.test = load('testValues/testPeriodic.mat', 'Eft_full1', 'Eft_full2','Varft_full1','Varft_full2');
assertElementsAlmostEqual(mean(values.real.Eft_full1), mean(values.test.Eft_full1), 'relative', 0.05);
assertElementsAlmostEqual(mean(values.real.Eft_full2), mean(values.test.Eft_full2), 'relative', 0.05);
assertElementsAlmostEqual(mean(values.real.Varft_full1), mean(values.test.Varft_full1), 'relative', 0.05);
assertElementsAlmostEqual(mean(values.real.Varft_full2), mean(values.test.Varft_full2), 'relative', 0.05);


function testPredictionsDrowning
values.real = load('realValuesPeriodic.mat', 'Eft_full', 'Varft_full');
values.test = load('testValues/testPeriodic.mat', 'Eft_full', 'Varft_full');
assertElementsAlmostEqual(mean(values.real.Eft_full), mean(values.test.Eft_full), 'relative', 0.05);
assertElementsAlmostEqual(mean(values.real.Varft_full), mean(values.test.Varft_full), 'relative', 0.05);



