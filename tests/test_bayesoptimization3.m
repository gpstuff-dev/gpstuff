function test_suite = test_bayesoptimization3
    %   Run specific demo, save values and compare the results to the expected.
    %   Works for both xUnit Test Framework package by Steve Eddins and for
    %   the built-in Unit Testing Framework (as of Matlab version 2013b).
    %
    %   See also
    %     TEST_ALL, DEMO_BAYEESOPTIMIZATION
    %
    %  Copyright (c) 2015 Jarno Vanhatalo
    %  Copyright (c) 2017 Eero Siivola

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


function testRunDemo(testCase)
  % Run the correspondin demo and save the values. Note this test has to
  % be run at lest once before the other test may succeed.
  
  % Suppress an expected warning
  Sw = warning('off','MATLAB:nearlySingularMatrix');
  rundemo(getName())
  warning(Sw); % Return the original warning visibility status
end

function testValues(testCase)
  % Test predictive mean and variance
  verifyVarsEqual(testCase, getName(), ...
    {'x'}, ...
    'RelTolElement', 0.01, 'RelTolRange', 0.02)
end

function testEI(testCase)
  % Test predictive mean and variance
  verifyVarsEqual(testCase, getName(), ...
    {'EI'}, ...
    'RelTolElement', 0.01, 'RelTolRange', 0.02)
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