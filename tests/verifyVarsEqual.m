function verifyVarsEqual(testCase, name, varargin)
%verifyVarsEqual  Verify the equality of demo results within tolerance.
%
%  Description
%    verifyVarsEqual(testCase, name, vars, funcs, OPTIONS)
%    compares the elements in the results of a demo into the expected ones
%    using relative tolerance to each element individually and/or to the
%    range of the expected array and/or using absolute tolerance. The data
%    is loaded from the 'realValues' and 'testValues' folders. Used by the
%    gpstuff tests. Can be used with xUnit Test Framework package by Steve
%    Eddins or with the Matlab's built-in Unit Testing Framework (as of
%    version 2013b).
%
%    Parameters:
%      testCase
%        The testCase object, if using the integrated unit test framework,
%        or empty array, if using the xUnit package.
%      name
%        The name of the demo.
%      vars (optional)
%        Cell array of strings defining the numerical array variables that
%        are compared. Giving string 'same' (default) looks the names of
%        the saved variables in the file 'realValues/<nameOfTheDemo>.mat'
%        and compares them.
%      funcs (optional)
%        Cell array of function handles that are applied into the
%        corresponding actual and expected arrays before comparison. Cell
%        containing an empty array indicates that no function is used for
%        that variable. Empty array in general indicates that no function
%        is applied for any variable. Providing a single function handle
%        applies the function to all variables.
%
%    OPTIONS is optional parameter-value pair
%      RelTolElement - tolerance relative to the magnitude of each element
%                      (default 0.05)
%      RelTolRange   - tolerance relative to the range of the elements
%                      (default 0.01)
%      AbsTol        - absolute tolerance (default 0)
%
%   See also
%     TEST_*
%
% Copyright (c) 2014 Tuomas Sivula

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% Parse inputs
ip = inputParser;
ip.FunctionName = 'verifyVarsEqual';
ip.addRequired('testCase',@(x) isempty(x) ...
    || isa(testCase, 'matlab.unittest.FunctionTestCase'))
ip.addRequired('name', @ischar)
ip.addOptional('vars', 'same', @(x) iscellstr(x) ...
    || (ischar(x) && strcmp(x,'same')))
ip.addOptional('funcs', [], @(x) isempty(x) || ...
    isa(x,'function_handle') || (iscell(x) && all( ...
    cellfun(@(y)isa(y,'function_handle')||isempty(y),x) )))
ip.addParamValue('RelTolElement', 0, ...
  @(x) isnumeric(x) && isscalar(x) && isreal(x) && x >= 0)
ip.addParamValue('RelTolRange', 0, ...
  @(x) isnumeric(x) && isscalar(x) && isreal(x) && x >= 0)
ip.addParamValue('AbsTol', 0, ...
  @(x) isnumeric(x) && isscalar(x) && isreal(x) && x >= 0)
ip.parse(testCase,name,varargin{:});
testCase = ip.Results.testCase;
name = ip.Results.name;
vars = ip.Results.vars;
funcs = ip.Results.funcs;
tol_el = ip.Results.RelTolElement;
tol_range = ip.Results.RelTolRange;
tol_abs = ip.Results.AbsTol;

% Handle parameter vars
if ischar(vars) && strcmp(vars, 'same')
  % Check the variable names from 'realValues/<nameOfTheDemo>.mat'
  vars = whos(matfile(['realValues/' name '.mat']));
  vars = {vars.name};
end

% Validate parameter funcs
if ~isempty(funcs) && iscell(funcs) && length(funcs) ~= length(vars)
  error('Parameter funcs size mismatch')
end

% Find the path to the test folder
path = mfilename('fullpath');
path = path(1:end-length(mfilename));

% For every variable in vars
for i = 1:length(vars)
  
  % Load the variables
  if isempty(testCase)
    % xUnit package
    Sw = warning('off','MATLAB:load:variableNotFound')
    try
      actual = load([path, 'testValues/' name '.mat'], vars{i});
      if ~isfield(actual, vars{i})
        error(['Variable ' vars{i} ' not found in testValues/' name '.mat'])
      end
      expected = load([path, 'realValues/' name '.mat'], vars{i});
      if ~isfield(expected, vars{i})
        error(['Variable ' vars{i} ' not found in realValues/' name '.mat'])
      end
    catch err
      warning(Sw)
      rethrow(err)
    end
    warning(Sw)
  else
    % Built-in test framework
    actual = assertWarningFree(testCase, ...
      @() load([path, 'testValues/' name '.mat'], vars{i}));
    expected = assertWarningFree(testCase, ...
      @() load([path, 'realValues/' name '.mat'], vars{i}));
  end
  
  % Error message (if the variable should fail the validation)
  msg = ['The variable ' vars{i} ' failed the verification.'];
  
  % Apply funcs
  if isa(funcs,'function_handle')
    actual = funcs(actual.(vars{i}));
    expected = funcs(expected.(vars{i}));
    %Add info to the error message
    msg = sprintf('%s\nFunction applied:\n%s', msg, func2str(funcs));
  elseif ~isempty(funcs) && ~isempty(funcs{i})
    actual = funcs{i}(actual.(vars{i}));
    expected = funcs{i}(expected.(vars{i}));
    %Add info to the error message
    msg = sprintf('%s\nFunction applied:\n%s', msg, func2str(funcs{i}));
  else
    actual = actual.(vars{i});
    expected = expected.(vars{i});
  end
  
  
  
  if isempty(testCase)
    
    % xUnit package (N.B. using assertions instead of verifications)
    if tol_el == 0 && tol_range == 0 && tol_abs == 0
      % No tolerance
      assertEqual(actual, expected, msg)
    else
      % Manual assertion
      
      % Compare
      diff = abs(actual(:) - expected(:));
      viol_el = 1;
      viol_range = 1;
      viol_abs = 1;
      if tol_el > 0
        viol_el = diff > tol_el*abs(expected(:));
      end
      if tol_range > 0
        viol_range = diff > tol_range ...
          *( max(expected(:)) - min(expected(:)) );
      end
      if tol_abs > 0
        viol_abs = diff > tol_abs;
      end
      
      if any( viol_el & viol_range & viol_abs )
        
        % Compose the message
        msg = sprintf( ...
          ['%s\nInput elements are not all equal within tolerances\n' ...
           '(RelTolElement=%g, RelTolRange=%g, AbsTol=%g)\n\n'], ...
          msg, tol_el, tol_range, tol_abs);
        if tol_el > 0 && any(viol_el)
          msg = sprintf( ...
            ['%sThe following indexes fail the elementwise relative ' ...
             'tolerance:\n%s\n\n'], ...
            msg, xunit.utils.arrayToString(find(viol_el)'));
        end
        if tol_range > 0 && any(viol_range)
          msg = sprintf( ...
            ['%sThe following indexes fail the range relative ' ...
             'tolerance:\n%s\n\n'], ...
            msg, xunit.utils.arrayToString(find(viol_range)'));
        end
        if tol_abs > 0 && any(viol_abs)
          msg = sprintf( ...
            ['%sThe following indexes fail the absolute ' ...
             'tolerance:\n%s\n\n'], ...
            msg, xunit.utils.arrayToString(find(viol_abs)'));
        end
        msg = sprintf( ...
          ['%sThe following indexes fail the comparison(s):\n%s\n\n' ...
           'First input:\n%s\n\nSecond input:\n%s'], ...
          msg, ...
          xunit.utils.arrayToString(find(viol_el & viol_range & viol_abs)'), ...
          xunit.utils.arrayToString(actual), ...
          xunit.utils.arrayToString(expected) );
        
        % Raise error
        throwAsCaller(MException('assertVars:tolExceeded', msg));
        
      end
    end

  else
    
    if tol_el == 0 && tol_range == 0 && tol_abs == 0
      % No tolerance
      verifyEqual(testCase, actual, expected, msg)
    else
      % Construct tolerance object
      if tol_el > 0
        tolObj = matlab.unittest.constraints.RelativeTolerance(tol_el);
      end
      if tol_range > 0
        if exist('tolObj', 'var')
          tolObj = tolObj | matlab.unittest.constraints.AbsoluteTolerance( ...
            tol_range*( max(expected(:)) - min(expected(:)) ));
        else
          tolObj = matlab.unittest.constraints.AbsoluteTolerance( ...
            tol_range*( max(expected(:)) - min(expected(:)) ));
        end
      end
      if tol_abs > 0
        if exist('tolObj', 'var')
          tolObj = tolObj | matlab.unittest.constraints.AbsoluteTolerance( ...
            tol_abs);
        else
          tolObj = matlab.unittest.constraints.AbsoluteTolerance(tol_abs);
        end
      end
      
      % Perform the verification
      testCase.verifyThat(actual, matlab.unittest.constraints.IsEqualTo( ...
        expected, 'Within', tolObj), msg);
    
    end

  end
  
  % Clear variables
  clear actual expected tolObj
  
end

