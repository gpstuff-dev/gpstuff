function test_suite = test_binomial_apc
initTestSuite;

% Set random number stream so that the test failing isn't because
% randomness. Run demo & save test values

    function testDemo
        stream0 = RandStream('mt19937ar','Seed',0);
        prevstream = RandStream.setDefaultStream(stream0);
        disp('Running: demo_binomial_apc')
        demo_binomial_apc
        path = which('test_binomial_apc.m');
        path = strrep(path,'test_binomial_apc.m', 'testValues/testBinomial_apc');
        save(path, 'Eft', 'Varft', 'Eft_3', 'Varft_3');
        RandStream.setDefaultStream(prevstream);
        drawnow;clear;close all
        
% Compare test values to real values.

    function testPredictionsAll
        values.real = load('realValuesBinomial_apc', 'Eft', 'Varft');
        values.test = load('testValues/testBinomial_apc', 'Eft', 'Varft');
        assertElementsAlmostEqual(mean(values.real.Eft), mean(values.test.Eft), 'relative', 0.05);
        assertElementsAlmostEqual(mean(values.real.Varft), mean(values.test.Varft), 'relative', 0.05);

    function testPredictionsCohort
        values.real = load('realValuesBinomial_apc', 'Eft_3', 'Varft_3');
        values.test = load('testValues/testBinomial_apc', 'Eft_3', 'Varft_3');
        assertElementsAlmostEqual(mean(values.real.Eft_3), mean(values.test.Eft_3), 'relative', 0.05);
        assertElementsAlmostEqual(mean(values.real.Varft_3), mean(values.test.Varft_3), 'relative', 0.05);
