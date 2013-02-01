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

## -*- texinfo -*-
## @deftypefn {Function File} {@var{parser} =} addSwitch (@var{parser}, @var{argname})
## @deftypefnx {Function File} {@var{parser} =} @var{parser}.addSwitch (@var{argname})
## Add new switch type of argument to the object @var{parser} of inputParser class.
##
## This method belongs to the inputParser class and implements a switch
## arguments type of API.
##
## @var{argname} must be a string with the name of the new argument. Arguments
## of this type can be specified at the end, after @code{Required} and @code{Optional},
## and mixed between the @code{ParamValue}. They default to false. If one of the
## arguments supplied is a string like @var{argname}, then after parsing the value
## of @var{parse}.Results.@var{argname} will be true.
##
## See @command{help inputParser} for examples.
##
## @seealso{inputParser, @@inputParser/addOptional, @@inputParser/addParamValue
## @@inputParser/addParamValue, @@inputParser/addRequired, @@inputParser/parse}
## @end deftypefn

function inPar = addSwitch (inPar, name)

  ## check @inputParser/subsref for the actual code
  if (nargin == 2)
    inPar = subsref (inPar, substruct(
                                      '.' , 'addSwitch',
                                      '()', {name}
                                      ));
  else
    print_usage;
  endif

endfunction
