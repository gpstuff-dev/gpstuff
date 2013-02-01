## Copyright (C) 2011 Carnë Draug <carandraug+dev@gmail.com>
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

## -*- texinfo -*-
## @deftypefn {Function File} {@var{parser} =} addParamValue (@var{parser}, @var{argname}, @var{default})
## @deftypefnx {Function File} {@var{parser} =} @var{parser}.addParamValue (@var{argname}, @var{default})
## @deftypefnx {Function File} {@var{parser} =} addParamValue (@var{parser}, @var{argname}, @var{default}, @var{validator})
## @deftypefnx {Function File} {@var{parser} =} @var{parser}.addParamValue (@var{argname}, @var{default}, @var{validator})
## Add new parameter to the object @var{parser} of the class inputParser to implement
## a name/value pair type of API.
##
## @var{argname} must be a string with the name of the new parameter.
##
## @var{default} will be the value used when the parameter is not specified.
##
## @var{validator} is an optional function handle to validate the given values
## for the parameter with name @var{argname}. Alternatively, a function name
## can be used.
##
## See @command{help inputParser} for examples.
##
## @seealso{inputParser, @@inputParser/addOptional, @@inputParser/addSwitch,
## @@inputParser/addParamValue, @@inputParser/addRequired, @@inputParser/parse}
## @end deftypefn

function inPar = addParamValue (inPar, name, def, val)

  ## check @inputParser/subsref for the actual code
  if (nargin == 3)
    inPar = subsref (inPar, substruct(
                                      '.' , 'addParamValue',
                                      '()', {name, def}
                                      ));
  elseif (nargin == 4)
    inPar = subsref (inPar, substruct(
                                      '.' , 'addParamValue',
                                      '()', {name, def, val}
                                      ));
  else
    print_usage;
  endif

endfunction
