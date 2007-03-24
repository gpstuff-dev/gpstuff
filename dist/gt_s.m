function s = gt_s(a);
%GT_S        Maximum log likelihood second derivatives for t- distribution.
%
%            Description
%            S = GT_S(A) takes a structure A, which contains
%            information about sigma of a parameter, and returns
%            the maximum log likelihood second derivative estimate.

% Copyright (c) 1998-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

li=cellfun('length',a.ii);
li=li./sum(li);
s=sum(((a.nu+1) ./ (a.nu .* a.s.^2)).*li);
  