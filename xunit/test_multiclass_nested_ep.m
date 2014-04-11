function test_suite = test_multiclass_nested_ep
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

disp('Running: demo_multiclass_nested_ep')
demo_multiclass_nested_ep;
path = which('test_multiclass_nested_ep.m');
path = strrep(path,'test_multiclass_nested_ep.m', 'testValues');
if ~(exist(path, 'dir') == 7)
    mkdir(path)
end
path = strcat(path, '/testMulticlass_nested_ep'); 
save(path, 'Eft', 'Covft', 'lpg');

% Set back initial random stream
setrandstream(prevstream);
drawnow;clear;close all

% Compare test values to real values.

function testPredictionsMulticlassEP
values.real = load('realValuesMulticlass_nested_ep', 'Eft', 'Covft', 'lpg');
values.test = load(strrep(which('test_multiclass_nested_ep.m'), 'test_multiclass_nested_ep.m',...
                 'testValues/testMulticlass_nested_ep'),'Eft', 'Covft', 'lpg');
assertElementsAlmostEqual(values.real.Eft, values.test.Eft, 'relative', 0.10);
assertElementsAlmostEqual(values.real.Covft, values.test.Covft, 'relative', 0.10);
assertElementsAlmostEqual(values.real.lpg, values.test.lpg, 'relative', 0.10);
