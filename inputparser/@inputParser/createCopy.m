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
## @deftypefn {Function File} {@var{new_parser} =} createCopy (@var{parser})
## Creates a copy @var{new_parser} of the object @var{parser} of the inputParser
## class.
##
## @seealso{inputParser, @@inputParser/addOptional, @@inputParser/addParamValue
## @@inputParser/addParamValue, @@inputParser/addRequired, @@inputParser/addSwitch,
## @@inputParser/parse}
## @end deftypefn

function outPar = createCopy (inPar)

  if ( nargin != 1 )
    print_usage;
  elseif ( !isa (inPar, 'inputParser') )
    error ("object must be of the inputParser class but '%s' was used", class (inPar) );
  endif

  ## yes, it's a ridiculous function but exists for MatLab compatibility. In there
  ## the inputParser class is a 'handle class' and this would just return a reference
  outPar = inPar;

endfunction
