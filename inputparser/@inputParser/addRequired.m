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
## @deftypefn {Function File} {@var{parser} =} addRequired (@var{parser}, @var{argname})
## @deftypefnx {Function File} {@var{parser} =} @var{parser}.addRequired (@var{argname})
## @deftypefnx {Function File} {@var{parser} =} addRequired (@var{parser}, @var{argname}, @var{validator})
## @deftypefnx {Function File} {@var{parser} =} @var{parser}.addRequired (@var{argname}, @var{validator})
## Add new mandatory argument to the object @var{parser} of inputParser class.
##
## This method belongs to the inputParser class and implements an ordered
## arguments type of API.
##
## @var{argname} must be a string with the name of the new argument. The order
## in which new arguments are added with @command{addrequired}, represents the
## expected order of arguments.
##
## @var{validator} is an optional function handle to validate the given values
## for the argument with name @var{argname}. Alternatively, a function name
## can be used.
##
## See @command{help inputParser} for examples.
##
## @emph{Note}: this can be used together with the other type of arguments but
## it must be the first (see @command{@@inputParser}).
##
## @seealso{inputParser, @@inputParser/addOptional, @@inputParser/addParamValue
## @@inputParser/addParamValue, @@inputParser/addSwitch, @@inputParser/parse}
## @end deftypefn

function inPar = addRequired (inPar, name, val)

  ## check @inputParser/subsref for the actual code
  if (nargin == 2)
    inPar = subsref (inPar, substruct(
                                      '.' , 'addRequired',
                                      '()', {name}
                                      ));
  elseif (nargin == 3)
    inPar = subsref (inPar, substruct(
                                      '.' , 'addRequired',
                                      '()', {name, val}
                                      ));
  else
    print_usage;
  endif

endfunction
