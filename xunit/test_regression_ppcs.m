function test_suite = test_regression_ppcs

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_REGRESSION_PPCS

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

disp('Running: demo_regression_ppcs')
demo_regression_ppcs
K = K(1:50, 1:50);
Ef = Ef(1:100);
path = which('test_regression_ppcs.m');
path = strrep(path,'test_regression_ppcs.m', 'testValues');
if ~(exist(path, 'dir') == 7)
    mkdir(path)
end
path = strcat(path, '/testRegression_ppcs'); 
save(path, 'K', 'Ef')

% Set back initial random stream
if str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
  RandStream.setDefaultStream(prevstream);
else
  RandStream.setGlobalStream(prevstream);
end
drawnow;clear;close all

% Compare test values to real values.

function testCovarianceMatrix
values.real = load('realValuesRegression_ppcs.mat', 'K');
values.test = load(strrep(which('test_regression_ppcs.m'), 'test_regression_ppcs.m', 'testValues/testRegression_ppcs.mat'), 'K');
assertElementsAlmostEqual(mean(full(values.real.K)), mean(full(values.test.K)), 'relative', 0.1)


function testPrediction
values.real = load('realValuesRegression_ppcs.mat', 'Ef');
values.test = load(strrep(which('test_regression_ppcs.m'), 'test_regression_ppcs.m', 'testValues/testRegression_ppcs.mat'), 'Ef');
assertElementsAlmostEqual(mean(values.real.Ef), mean(values.test.Ef), 'relative', 0.1);

