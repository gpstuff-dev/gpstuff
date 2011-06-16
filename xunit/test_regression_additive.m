function test_suite = test_regression_additive
initTestSuite;


% Set random number stream so that test failing isn't because randomness.
% Run demo & save test values.

function testDemo
    stream0 = RandStream('mt19937ar','Seed',0);
    RandStream.setDefaultStream(stream0);
    disp('Running: demo_regression_additive')
    demo_regression_additive
    path = which('test_regression_additive.m');
    path = strrep(path,'test_regression_additive.m', 'testValues');
    if ~(exist(path, 'dir') == 7)
        mkdir(path)
    end
    path = strcat(path, '/testRegression_additive');     
    save(path, 'Eft_map');
    drawnow;clear;close all
    
% Compare test values to real values.    

function testNeuralNetworkCFPrediction
    values.real = load('realValuesRegression_additive.mat','Eft_map');
    values.test = load(strrep(which('test_regression_additive.m'), 'test_regression_additive.m', 'testValues/testRegression_additive.mat'),'Eft_map');
    assertElementsAlmostEqual(mean(values.real.Eft_map), mean(values.test.Eft_map), 'relative', 0.05);