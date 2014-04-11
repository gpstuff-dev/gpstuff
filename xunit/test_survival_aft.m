function test_suite = test_survival_aft

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_SURVIVAL_WEIBULL
%
% Copyright (c) 2011-2012 Ville Tolvanen

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

initTestSuite;

function testDemo
% Set random number stream so that failing isn't because randomness. Run
% demo & save test values.
prevstream=setrandstream(0);

disp('Running: demo_survival_aft')
demo_survival_aft;
path = which('test_survival_aft.m');
path = strrep(path,'test_survival_aft.m', 'testValues');
if ~(exist(path, 'dir') == 7)
    mkdir(path)
end
path = strcat(path, '/testSurvival_aft'); 
save(path, 'pmu');

% Set back initial random stream
setrandstream(prevstream);
drawnow;clear;close all

% Compare test values to real values.

function testPredictions
values.real = load('realValuesSurvival_aft', 'pmu');
values.test = load(strrep(which('test_survival_aft.m'), 'test_survival_aft.m', 'testValues/testSurvival_aft'), 'pmu');
assertVectorsAlmostEqual(values.real.pmu(:,1), values.test.pmu(:,1), 'relative', 0.05);
assertVectorsAlmostEqual(values.real.pmu(:,2), values.test.pmu(:,2), 'relative', 0.05);
assertVectorsAlmostEqual(values.real.pmu(:,3), values.test.pmu(:,3), 'relative', 0.05);

