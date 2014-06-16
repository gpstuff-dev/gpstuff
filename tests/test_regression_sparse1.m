function test_suite = test_regression_sparse1

%   Run specific demo, save values and compare the results to the expected.
%   Works for both xUnit Test Framework package by Steve Eddins and for
%   the built-in Unit Testing Framework (as of Matlab version 2013b).
%
%   See also
%     TEST_ALL, DEMO_REGRESSION_SPARSE1
%
% Copyright (c) 2011-2012 Ville Tolvanen
% Copyright (c) 2014 Tuomas Sivula

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.
  
  % Check which package to use
  if exist('initTestSuite', 'file')
    initTestSuite_path = which('initTestSuite');
  else
    initTestSuite_path = '';
  end
  if ~isempty(initTestSuite_path) && ...
     exist([initTestSuite_path(1:end-15) 'runtests'], 'file')
    % xUnit package
    initTestSuite;
  else
    % Built-in package
    % Use all functions except the @setup
    tests = localfunctions;
    tests = tests(~cellfun(@(x)strcmp(func2str(x),'setup'),tests));
    test_suite = functiontests(tests);
  end
end


% -------------
%     Tests
% -------------

function testRunDemo(testCase)
  % Run the correspondin demo and save the values. Note this test has to
  % be run at lest once before the other test may succeed.
  rundemo(getName())
end

function testPredictionsCS(testCase)
  verifyVarsEqual(testCase, getName(), {'Eft_cs'}, ...
    'RelTolRange', 0.02)
end

function testPredictionsFIC(testCase)
  verifyVarsEqual(testCase, getName(), {'Eft_fic'}, ...
    'RelTolRange', 0.02)
end

function testPredictionsPIC(testCase)
  verifyVarsEqual(testCase, getName(), {'Eft_pic'}, ...
    'RelTolRange', 0.02)
end

function testPredictionsVAR(testCase)
  verifyVarsEqual(testCase, getName(), {'Eft_var'}, ...
    'RelTolRange', 0.02)
end

function testPredictionsDTC(testCase)
  verifyVarsEqual(testCase, getName(), {'Eft_dtc'}, ...
    'RelTolRange', 0.02)
end


% ------------------------
%     Helper functions
% ------------------------

function testCase = setup
  % Helper function to suply empty array into variable testCase as an
  % argument for each test function, if using xUnit package. Not to be
  % used with built-in test framework.
  testCase = [];
end

function name = getName
  % Helperfunction that returns the name of the demo, e.g. 'binomial1'.
  name = mfilename;
  name = name(6:end);
end

