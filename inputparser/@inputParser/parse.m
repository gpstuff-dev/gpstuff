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
## @deftypefn {Function File} {@var{parser} =} parse (@var{parser}, @var{varargin})
## @deftypefnx {Function File} {@var{parser} =} @var{parser}.parse (@var{varargin})
## Parses and validates list of arguments according to object @var{parser} of the
## class inputParser.
##
## After parsing, the results can be accessed with the @command{Results}
## accessor. See @command{help inputParser} for a more complete description.
##
## @seealso{inputParser, @@inputParser/addOptional, @@inputParser/addParamValue
## @@inputParser/addParamValue, @@inputParser/addRequired, @@inputParser/addSwitch}
## @end deftypefn

function inPar = parse (inPar, varargin)

  ## check @inputParser/subsref for the actual code
  inPar = subsref (inPar, substruct(
                                    '.' , 'parse',
                                    '()', varargin
                                    ));

endfunction
