function s = resampwor(p,m,n)
%RESAMWOR Random resampling without replacement
%
%   Description:
%   S = RESAMWOR(P) returns a new set of indices according to the
%   probabilities P without replacemnt. P is array of probabilities,
%   which are not necessarily normalized, though they must be
%   non-negative, and not all zero. The size of S is the size of P.
%
%   S = RESAMWOR(P,M,N) returns M by N matrix.
%
%   S = RESAMWOR(P,M) returns M by M matrix.
%
%   See also RESAMPSIM, RESAMPRES, RESAMPSTR, RESAMPDET
%
% Copyright (c) 2003-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin<2
    [m,n] = size(p);
elseif nargin==2
    n = m;
end
if m*n>numel(p)
  error('In resampling without replacment M*N has to be smaller than numel(P)')
end
r=rand([m,n]);
s=zeros([m,n]);
for i=1:m*n
  pc=cumsum(p(:));
  pc=pc./pc(end);
  s(i)=binsgeq(pc,r(i));
  p(s(i))=0;
end
