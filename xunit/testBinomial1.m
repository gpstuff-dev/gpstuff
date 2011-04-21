function test_suite = testBinomial1
initTestSuite;

% Set random number stream so that the test failing isn't because
% randomness. Run demo & save test values.

    function testDemo
        stream0 = RandStream('mt19937ar','Seed',0);
        prevstream = RandStream.setDefaultStream(stream0);
        disp('Running: demo_binomial1')
        demo_binomial1
        path = which('testBinomial1');
        path = strrep(path,'testBinomial1.m', 'testValues/testBinomial1');
        save(path, 'Eyt_la', 'Varyt_la', 'pyt_la');
%         save('testValues/testBinomial1', 'Eyt_la', 'Varyt_la', 'pyt_la');
        RandStream.setDefaultStream(prevstream);
        drawnow;clear;close all


% Test predictive mean, variance and density for binomial model with 5% tolerance.        
        
    function testPredictiveMeanAndVariance
        values.real = load('realValuesBinomial1.mat','Eyt_la','Varyt_la');
        values.test = load('testValues/testBinomial1.mat','Eyt_la','Varyt_la');
        assertElementsAlmostEqual(mean(values.real.Eyt_la), mean(values.test.Eyt_la), 'relative', 0.05);
        assertElementsAlmostEqual(mean(values.real.Varyt_la), mean(values.test.Varyt_la), 'relative', 0.05);


    function testPredictiveDensity
        values.real = load('realValuesBinomial1.mat','pyt_la');
        values.test = load('testValues/testBinomial1.mat','pyt_la');
        assertElementsAlmostEqual(mean(values.real.pyt_la), mean(values.test.pyt_la), 'relative', 0.05);


