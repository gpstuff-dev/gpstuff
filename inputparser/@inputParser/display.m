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

function display (inPar)

  if (inPar.FunctionName)
    name = inPar.FunctionName(1:end-3);
  else
    name = "";
  endif

  required    = arg_list (inPar.Required);
  optional    = arg_list (inPar.Optional);
  paramvalue  = arg_list (inPar.ParamValue);
  switches    = arg_list (inPar.Switch);

  printf ("Input Parser object with:\n");
  printf ("CaseSensitive: %s\n", binstr (inPar.CaseSensitive));
  printf ("StructExpand : %s\n", binstr (inPar.StructExpand));
  printf ("KeepUnmatched: %s\n", binstr (inPar.KeepUnmatched));
  printf ("FunctionName : '%s'\n", name);
  printf ("\n");
  printf ("Required arguments  : %s\n", required);
  printf ("Optional arguments  : %s\n", optional);
  printf ("ParamValue arguments: %s\n", paramvalue);
  printf ("Switch arguments    : %s\n", switches);

endfunction

function [str] = binstr (bin)
  if (bin)
    str = "true";
  else
    str = "false";
  endif
endfunction

function [str] = arg_list (args)
  str = strcat ("'", fieldnames (args), {"', "});
  if (!isempty (str))
    str = cstrcat (str{:});
    str = str(1:end-2);     # remove the last comma and space
  else
    str = "";
  endif
endfunction
