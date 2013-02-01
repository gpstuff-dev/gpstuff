## Copyright (C) 2011-2012 Carnë Draug <carandraug+dev@gmail.com>
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

function inPar = subsref (inPar, idx)

  if ( !isa (inPar, 'inputParser') )
    error ("object must be of the inputParser class but '%s' was used", class (inPar) );
  elseif ( idx(1).type != '.' )
    error ("invalid index for class %s", class (inPar) );
  endif

  ## the following at the end may allow to use the obj.method notation one day
  ## jwe is very against this ugly hack
  ## what would happen if the user has the obj inside a struct? Bad things!
#  ori = inputname(1);
#  assignin('caller', ori, inPar);

  method = idx(1).subs;

  switch method
  case 'Results'
    inPar = retrieve_results (inPar, idx)
  case 'Parameters'
    inPar = inPar.Parameters;
  case 'parse'
    inPar = parse_args (inPar, idx);
  case 'Unmatched'
  case 'UsingDefaults'
  case {'addOptional', 'addParamValue', 'addRequired', 'addSwitch'}
    inPar = check_methods (inPar, idx);
  otherwise
    error ("invalid index for reference of class %s", class (inPar) );
  endswitch

  ## TODO we should make inPar an object of the inputParser class again. At
  ## least after running parse it becomes just a structure again. While that is
  ## bad, at least allows for easy access to the Results and Unmatched fields
  ## without extra coding.
#  inPar = class (inPar, 'inputParser');

endfunction

function out = retrieve_results (inPar, idx)

  if ( numel(idx) != 2 || idx(2).type != '.' )
    print_usage ("@inputParser/Results");
  endif

  out = inPar.Results.(idx(2).subs);

endfunction


## when parsing options, here's the principle: Required options have to be the
## first ones. They are followed by Optional if any. In the end come the
## ParamValue mixed with Switch. Any other order makes no sense
function inPar = parse_args (inPar, idx)

  ## syntax is inPar.parse (arguments)
  if ( numel(idx) != 2 || idx(2).type != '()' )
    print_usage ("@inputParser/parse");
  endif

  ## this makes it easier to read but may be memory instensive
  args = idx(2).subs;

  ## make copy of ordered list of Parameters to keep the original intact and readable
  inPar.copy = inPar.Parameters;

  if ( numel (fieldnames (inPar.Required)) > numel (args) )
    error("%sNot enough arguments", inPar.FunctionName);
  endif

  ## we take names out of 'copy' and values out of 'args', evaluate them and
  ## store them into 'Results'
  for i = 1 : numel (fieldnames (inPar.Required))
    [name, inPar.copy] = shift (inPar.copy);
    [value, args]      = shift (args);
    if ( !feval (inPar.Required.(name).validator, value) )
      error_invalid (inPar.FunctionName, name, inPar.Required.(name).validator);
    endif
    inPar.Results.(name) = value;
  endfor

  ## loop a maximum #times of the number of Optional, similarly to the required
  ## loop. Once ran out of 'args', move their name into usingDefaults, place
  ## their default values into 'Results', and break

  ## because if an argument is string and does not validate, should be considered
  ## a ParamValue key
  found_possible_key = false;

  for i = 1 : numel (fieldnames (inPar.Optional))
    if ( !numel (args) || found_possible_key)
      ## loops the number of Optional options minus the number of them already processed
      for n = 1 : (numel (fieldnames (inPar.Optional)) - i + 1 )
        [name, inPar.copy]   = shift (inPar.copy);
        inPar.UsingDefaults  = push (inPar.UsingDefaults, name);
        inPar.Results.(name) = inPar.Optional.(name).default;
      endfor
      break
    endif
    [name, inPar.copy] = shift (inPar.copy);
    [value, args]      = shift (args);
    if ( !feval (inPar.Optional.(name).validator, value) )
      if (ischar (value) )
        ## maybe the other optional are not defined, this can be Paramvalue
        ## place this one on defaults and go back to the top with note to clean loop
        inPar.UsingDefaults  = push (inPar.UsingDefaults, name);
        inPar.Results.(name) = inPar.Optional.(name).default;
        found_possible_key   = true;
        args = unshift (args, value);
        continue
      else
        error_invalid (inPar.FunctionName, name, inPar.Optional.(name).validator);
      endif
    else
      inPar.Results.(name) = value;
    endif
  endfor

  ## loop a maximum #times of the number of ParamValue, taking pairs of keys and
  ## values out 'args'. We no longer expect an order so we need the index in
  ## 'copy' to remove it from there. Once ran out of 'args', move their name
  ## into usingDefaults, place their default values into 'Results', and break
  for i = 1 : (numel (fieldnames (inPar.ParamValue)) + numel (fieldnames (inPar.Switch)))
    if ( !numel (args) )
      ## loops the number of times left in 'copy' since these are the last type
      for n = 1 : numel (inPar.copy)
        [name, inPar.copy]   = shift (inPar.copy);
        inPar.UsingDefaults  = push (inPar.UsingDefaults, name);
        if (isfield (inPar.ParamValue, name))
          inPar.Results.(name) = inPar.ParamValue.(name).default;
        else
          inPar.Results.(name) = inPar.Switch.(name).default;
        endif
      endfor
      break
    endif
    [key, args] = shift (args);
    if ( !ischar (key) )
      error("%sParameter/Switch names must be strings", inPar.FunctionName);
    endif
    if (inPar.CaseSensitive)
      index = find( strcmp(inPar.copy, key));
    else
      index = find( strcmpi(inPar.copy, key));
    endif
    ## we can't use isfield here to support case insensitive
    if (any (strcmpi (fieldnames (inPar.Switch), key)))
      value  = true;
      method = "Switch";
    else
      ## then it must be a ParamValue (even if unmatched), shift its value
      if (numel (args) < 1)
        error ("%sparameter '%s' does not have a value", inPar.FunctionName, key);
      endif
      [value, args] = shift (args);
      method = "ParamValue";
    endif

    ## empty index means no match so either return error or move them into 'Unmatched'
    if (!isempty (index))
      [name, inPar.copy] = shift (inPar.copy, index);
      if ( !feval (inPar.(method).(name).validator, value))
        error_invalid (inPar.FunctionName, key, inPar.(method).(name).validator);
      endif
      ## we use the name shifted from 'copy' instead of the key from 'args' in case
      ## the key is in the wrong case
      inPar.Results.(name) = value;
    elseif (isempty (index) && inPar.KeepUnmatched )
      inPar.Unmatched.(key) = value;
      i = i - 1; # this time didn't count, go back one
    else
      error ("%sargument '%s' did not match any valid parameter of the parser", inPar.FunctionName, key);
    endif
  endfor

  ## if there's leftovers they must be unmatched. Note that some unmatched can
  ## have already been processed in the ParamValue loop
  if ( numel (args) && inPar.KeepUnmatched )
    for i = 1 : ( numel(args) / 2 )
      [key, args]           = shift (args);
      [value, args]         = shift (args);
      inPar.Unmatched.(key) = value;
    endfor
  elseif ( numel (args) )
    error("%sfound unmatched parameters at end of arguments list", inPar.FunctionName);
  endif

  ## remove copied field, keep it clean
  inPar = rmfield (inPar, 'copy');

endfunction


function inPar = check_methods (inPar, idx)

  ## this makes it easier to read but is more memory intensive?
  method  = idx(1).subs;
  args    = idx(2).subs;
  func    = sprintf ("@inputParser/%s", method);

  if ( idx(2).type != '()' )
    print_usage (func);
  endif
  def_val      = @() true;
  [name, args] = shift (args);
  ## a validator is optional but that complicates handling all the parsing with
  ## few functions and conditions. If not specified @() true will always return
  ## true. Simply using true is not enough because if the argument is zero it
  ## return false and if it's too large, takes up memory
  switch method
  case {'addOptional', 'addParamValue'}
    if     ( numel (args) == 1 )
      args{2} = def_val;
    elseif ( numel (args) == 2 )
      args{2} = validate_validator (args{2});
    else
      print_usage(func);
    endif
    [def, val] = args{:};
  case {'addRequired'}
    if     ( numel (args) == 0 )
      val = def_val;
    elseif ( numel (args) == 1 )
      val = validate_validator (args{1});
    else
      print_usage(func);
    endif
    def = false;
  case {'addSwitch'}
    if ( numel (args) == 0 )
      val = def_val;
      def = false;
    else
      print_usage(func);
    endif
  otherwise
    error ("invalid index for reference of class %s", class (inPar) );
  endswitch

  inPar = validate_args (method(4:end), inPar, name, val, def);

endfunction

## because we are nice we also support using the name of a function and not only
## a function handle
function val = validate_validator (val)
  if ( ischar (val) )
    val = str2func (val);
  elseif ( !isa (val, 'function_handle') )
    error ("validator must be a function handle or the name of a valid function");
  end
endfunction

## to have a single function that handles them all, something must be done with
## def value, because addRequire does not have those. That's why the order of
## the last two args is different from the rest and why 'def' has a default value
function inPar = validate_args (method, inPar, name, val, def = false)

  if ( !strcmp (class (inPar), 'inputParser') )
    error ("object must be of the inputParser class but '%s' was used", class (inPar) );
  elseif ( !isvarname (name) )
    error ("invalid variable name in argname");
  endif

  ## because the order arguments are specified are the order they are expected,
  ## can't have ParamValue/Switch before Optional, and Optional before Required
  n_optional  = numel (fieldnames (inPar.Optional));
  n_params    = numel (fieldnames (inPar.ParamValue));
  n_switch    = numel (fieldnames (inPar.Switch));
  if     ( strcmp (method, 'Required') && ( n_optional || n_params || n_switch) )
    error ("Can't specify 'Required' arguments after Optional, ParamValue or Switch");
  elseif ( strcmp (method, 'Optional') && ( n_params || n_switch) )
    error ("Can't specify 'Optional' arguments after ParamValue or Switch");
  endif

  ## even if CaseSensitive is turned on, we still shouldn't have two args with
  ## the same. What if they decide to change in the middle of specifying them?
  if ( any (strcmpi (inPar.Parameters, name)) )
    error ("argname '%s' has already been specified", name);
  else
    inPar.Parameters = push (inPar.Parameters, name);
    inPar.(method).(name).default = def;
    inPar.(method).(name).validator = val;
  endif

  ## make sure that the given default value is actually valid
  ## TODO make sure that when using the default, it's only validated once
  if ( isa (val, 'function_handle') && !strcmpi (method, 'Required') && !feval (val, def) )
    error ("default value for '%s' failed validation with '%s'", name, func2str (val) );
  endif

endfunction

## this is just for consistency of error message
function error_invalid (prefix, name, val)
  error("%sargument '%s' failed validation %s", prefix, name, func2str (val));
endfunction

################################################################################
## very auxiliary functions
################################################################################

function [out, in] = shift (in, idx = 1)
  out     = in{idx};
  in(idx) = [];
endfunction

function [in] = unshift (in, add)
  if ( !iscell (add) )
    add = {add};
  endif
  in (numel(add) + 1 : end + numel(add)) = in;
  in (1:numel(add)) = add;
endfunction

function [in] = push (in, add)
  if ( !iscell (add) )
    add = {add};
  endif
  in( end+1 : end+numel(add) ) = add;
endfunction
