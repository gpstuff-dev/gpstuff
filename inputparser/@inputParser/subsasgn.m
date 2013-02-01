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

function inPar = subsasgn (inPar, idx, rhs)

  if ( idx.type != '.' )
    error ("invalid index for class %s", class (inPar) );
  endif

  switch idx.subs
    case {'CaseSensitive', 'KeepUnmatched', 'StructExpand'}
      if ( !islogical (rhs) )
        error("Property '%s' of the class inputParser must be logical", idx.subs)
      endif
      inPar.(idx.subs) = rhs;
    case 'FunctionName'
      if ( !ischar (rhs) )
        error("Property 'FunctionName' of the class inputParser can only be set to a string")
      endif
      inPar.(idx.subs) = sprintf("%s : ", rhs);
    otherwise
      error ("invalid index for assignment of class %s", class (inPar) );
  endswitch

endfunction
