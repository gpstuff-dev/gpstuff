function test_suite = test_regression_sparse1

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_REGRESSION_SPARSE1

initTestSuite;


function testDemo
% Set random number stream so that failing isn't because randomness. Run
% demo & save test values.
stream0 = RandStream('mt19937ar','Seed',0);
if str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
  prevstream = RandStream.setDefaultStream(stream0);
else
  prevstream = RandStream.setGlobalStream(stream0);
end

disp('Running: demo_regression_sparse1')
demo_regression_sparse1
path = which('test_regression_sparse1');
path = strrep(path,'test_regression_sparse1.m', 'testValues');
if ~(exist(path, 'dir') == 7)
    mkdir(path)
end
path = strcat(path, '/testRegression_sparse1'); 
save(path, 'Eft_fic', 'Eft_pic', 'Eft_var', 'Eft_dtc', 'Eft_cs');

% Set back initial random stream
if str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
  RandStream.setDefaultStream(prevstream);
else
  RandStream.setGlobalStream(prevstream);
end
drawnow;clear;close all

% Compare test values to real values.

function testPredictionsCS
values.real = load('realValuesRegression_sparse1', 'Eft_cs');
values.test = load(strrep(which('test_regression_sparse1.m'), 'test_regression_sparse1.m', 'testValues/testRegression_sparse1'), 'Eft_cs');
assertElementsAlmostEqual(mean(values.real.Eft_cs), mean(values.test.Eft_cs), 'relative', 0.1);


function testPredictionsFIC
values.real = load('realValuesRegression_sparse1', 'Eft_fic');
values.test = load(strrep(which('test_regression_sparse1.m'), 'test_regression_sparse1.m', 'testValues/testRegression_sparse1'), 'Eft_fic');
assertElementsAlmostEqual(mean(values.real.Eft_fic), mean(values.test.Eft_fic), 'relative', 0.1);


function testPredictionsPIC
values.real = load('realValuesRegression_sparse1', 'Eft_pic');
values.test = load(strrep(which('test_regression_sparse1.m'), 'test_regression_sparse1.m', 'testValues/testRegression_sparse1'), 'Eft_pic');
assertElementsAlmostEqual(mean(values.real.Eft_pic), mean(values.test.Eft_pic), 'relative', 0.1);


function testPredictionsVAR
values.real = load('realValuesRegression_sparse1', 'Eft_var');
values.test = load(strrep(which('test_regression_sparse1.m'), 'test_regression_sparse1.m', 'testValues/testRegression_sparse1'), 'Eft_var');
assertElementsAlmostEqual(mean(values.real.Eft_var), mean(values.test.Eft_var), 'relative', 0.1);


function testPredictionsDTC
values.real = load('realValuesRegression_sparse1', 'Eft_dtc');
values.test = load(strrep(which('test_regression_sparse1.m'), 'test_regression_sparse1.m', 'testValues/testRegression_sparse1'), 'Eft_dtc');
assertElementsAlmostEqual(mean(values.real.Eft_dtc), mean(values.test.Eft_dtc), 'relative', 0.1);
