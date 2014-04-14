function y = sumlogs(x,dim)
%SUMLOGS Sum of elements where numbers are represented by their logarithms.
%
%  Description
%    C=SUMLOGS(A) computes C=log(sum(exp(A))) in such a fashion
%    that it works even when elements have large magnitude.
%
%    C=SUMLOGS(A,DIM) sums along the dimension DIM. 
%
%    See also SUM
%
% Copyright (c) 2013 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

if nargin<2
  dim=find(size(x)>1,1);
end
maxx=max(x(:));
y=maxx+log(sum(exp(x-maxx),dim));
