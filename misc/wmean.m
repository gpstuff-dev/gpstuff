function y = wmean(x, w)
%WMEAN  Weighted average or mean value.
%
%  Description
%    WMEAN(X,W) is the weighted mean value of the elements in X
%    (along first dimension) given weights W.
%
%  See also wprctile
%
% Copyright (c) 2000-2013 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

y=sum(bsxfun(@times,x,w),1);
