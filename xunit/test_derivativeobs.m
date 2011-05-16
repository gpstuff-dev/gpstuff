function test_suite = test_derivativeobs
initTestSuite;

% Set random number stream so test failing isn't because randomness. Run
% demo & save test values.

    function testDemo
        stream0 = RandStream('mt19937ar','Seed',0);
        prevstream = RandStream.setDefaultStream(stream0);
        disp('Running: demo_derivativeobs')
        demo_derivativeobs
        path = which('test_derivativeobs.m');
        path = strrep(path,'test_derivativeobs.m', 'testValues');
        if ~(exist(path, 'dir') == 7)
            mkdir(path)
        end
        path = strcat(path, '/testDerivativeobs');           
        save(path);
        RandStream.setDefaultStream(prevstream);
        drawnow;clear;close all

% Compare test values to real values.
        
    function testPrediction
        values.real = load('realValuesDerivativeobs.mat','Eft','Varft');
        values.test = load('testValues/testDerivativeobs','Eft','Varft');
        assertElementsAlmostEqual(mean(values.real.Eft),mean(values.test.Eft),'relative', 0.05);
        assertElementsAlmostEqual(mean(values.real.Varft),mean(values.test.Varft),'relative', 0.05);
    
        
    function testPredictionDerivative
        values.real = load('realValuesDerivativeobs.mat','Eft2','Varft2');
        values.test = load('testValues/testDerivativeobs.mat','Eft2','Varft2');
        assertElementsAlmostEqual(mean(values.real.Eft2),mean(values.test.Eft2),'relative', 0.05);
        assertElementsAlmostEqual(mean(values.real.Varft2),mean(values.test.Varft2),'relative', 0.05);
        

