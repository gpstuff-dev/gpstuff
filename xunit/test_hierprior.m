function test_suite = test_hierprior

%   Run specific demo, save values and compare the results to the expected.
%   Works for both xUnit Test Framework package by Steve Eddins and for
%   the built-in Unit Testing Framework (as of Matlab version 2013b).
%
%   See also
%     TEST_ALL, DEMO_HIERPRIOR
%
% Copyright (c) 2014 Tuomas Sivula

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.
  
  % Check if the caller was the xUnit package or the built-in test framework
  c_stack = dbstack('-completenames');
  if exist([c_stack(2).file(1:end-11) 'initTestSuite'], 'file')
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

function testDerivatesRegression(testCase)
  verifyVarsEqual(testCase, getName(), {'derivs'}, {@(x)x{1}}, ...
    'RelTolElement', 0.05, 'RelTolRange', 0.01)
  verifyVarsEqual(testCase, getName(), {'derivs'}, {@(x)x{2}}, ...
    'RelTolElement', 0.05, 'RelTolRange', 0.01)
end

function testDerivatesLaplace(testCase)
  verifyVarsEqual(testCase, getName(), {'derivs'}, {@(x)x{3}}, ...
    'AbsTol', 1e-5)
  verifyVarsEqual(testCase, getName(), {'derivs'}, {@(x)x{4}}, ...
    'AbsTol', 1e-5)
end

function testDerivatesEP(testCase)
  verifyVarsEqual(testCase, getName(), {'derivs'}, {@(x)x{5}}, ...
    'AbsTol', 1e-5)
  verifyVarsEqual(testCase, getName(), {'derivs'}, {@(x)x{6}}, ...
    'AbsTol', 1e-5)
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

