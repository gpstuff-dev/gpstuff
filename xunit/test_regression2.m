function test_suite = test_regression2
initTestSuite;

% Set random number stream so that test failing isn't because randomness.
% Run demo & save test values.

function testDemo
    stream0 = RandStream('mt19937ar','Seed',0);
    RandStream.setDefaultStream(stream0);
    disp('Running: demo_regression2')
    demo_regression2
    path = which('test_regression2.m');
    path = strrep(path,'test_regression2.m', 'testValues');
    if ~(exist(path, 'dir') == 7)
        mkdir(path)
    end
    path = strcat(path, '/testRegression2');     
    save(path, 'Eft_fic', 'Varft_fic', 'Eft_pic', 'Varft_pic', ...
         'Eft_csfic', 'Varft_csfic');
    drawnow;clear;close all
    
% Compare test values to real values.

function testPredictionsFIC
    values.real = load('realValuesRegression2.mat', 'Eft_fic', 'Varft_fic');
    values.test = load('testValues/testRegression2.mat', 'Eft_fic', 'Varft_fic');
    assertElementsAlmostEqual(mean(values.real.Eft_fic), mean(values.test.Eft_fic), 'relative', 0.05);
	assertElementsAlmostEqual(mean(values.real.Varft_fic), mean(values.test.Varft_fic), 'relative', 0.05);
    
function testPredictionsPIC 
    values.real = load('realValuesRegression2.mat', 'Eft_pic', 'Varft_pic');
    values.test = load('testValues/testRegression2.mat', 'Eft_pic', 'Varft_pic');
    assertElementsAlmostEqual(mean(values.real.Eft_pic), mean(values.test.Eft_pic), 'relative', 0.05);
	assertElementsAlmostEqual(mean(values.real.Varft_pic), mean(values.test.Varft_pic), 'relative', 0.05);
    

function testPredictionsSparse
    values.real = load('realValuesRegression2.mat', 'Eft_csfic', 'Varft_csfic');
    values.test = load('testValues/testRegression2.mat', 'Eft_csfic', 'Varft_csfic');
    assertElementsAlmostEqual(mean(values.real.Eft_csfic), mean(values.test.Eft_csfic), 'relative', 0.05);
	assertElementsAlmostEqual(mean(values.real.Varft_csfic), mean(values.test.Varft_csfic), 'relative', 0.05);