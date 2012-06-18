function test_suite = test_regression_additive1

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_REGRESSION_ADDITIVE1

initTestSuite;

function testDemo
   % Set random number stream so that test failing isn't because randomness.
   % Run demo & save test values.
   stream0 = RandStream('mt19937ar','Seed',0);
   if str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
     prevstream = RandStream.setDefaultStream(stream0);
   else
     prevstream = RandStream.setGlobalStream(stream0);
   end
   
   disp('Running: demo_regression_additive1')
   demo_regression_additive1
   path = which('test_regression_additive1.m');
   path = strrep(path,'test_regression_additive1.m', 'testValues');
   if ~(exist(path, 'dir') == 7)
     mkdir(path)
   end
   path = strcat(path, '/testRegression_additive1');
   save(path, 'Eft_fic', 'Varft_fic', 'Eft_pic', 'Varft_pic', ...
     'Eft_csfic', 'Varft_csfic');
   
   % Set back initial random stream
   if str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
     RandStream.setDefaultStream(prevstream);
   else
     RandStream.setGlobalStream(prevstream);
   end
   drawnow;clear;close all
    
% Compare test values to real values.

function testPredictionsFIC
    values.real = load('realValuesRegression_additive1.mat', 'Eft_fic', 'Varft_fic');
    values.test = load(strrep(which('test_regression_additive1.m'), 'test_regression_additive1.m', 'testValues/testRegression_additive1.mat'), 'Eft_fic', 'Varft_fic');
    assertElementsAlmostEqual((values.real.Eft_fic), (values.test.Eft_fic), 'absolute', 1);
    assertElementsAlmostEqual(mean(values.real.Varft_fic), mean(values.test.Varft_fic), 'relative', 0.05);
    
function testPredictionsPIC 
    values.real = load('realValuesRegression_additive1.mat', 'Eft_pic', 'Varft_pic');
    values.test = load(strrep(which('test_regression_additive1.m'), 'test_regression_additive1.m', 'testValues/testRegression_additive1.mat'), 'Eft_pic', 'Varft_pic');
    assertElementsAlmostEqual(mean(values.real.Eft_pic), mean(values.test.Eft_pic), 'relative', 0.05);
    assertElementsAlmostEqual(mean(values.real.Varft_pic), mean(values.test.Varft_pic), 'relative', 0.05);
    

function testPredictionsSparse
    values.real = load('realValuesRegression_additive1.mat', 'Eft_csfic', 'Varft_csfic');
    values.test = load(strrep(which('test_regression_additive1.m'), 'test_regression_additive1.m', 'testValues/testRegression_additive1.mat'), 'Eft_csfic', 'Varft_csfic');
    assertElementsAlmostEqual(mean(values.real.Eft_csfic), mean(values.test.Eft_csfic), 'relative', 0.05);
    assertElementsAlmostEqual(mean(values.real.Varft_csfic), mean(values.test.Varft_csfic), 'relative', 0.05);