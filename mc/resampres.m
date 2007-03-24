function s = resampres(p);
%RESAMPRES Residual resampling
%
%   Description:
%   S = RESAMPRES(P) returns a new set of indices according to 
%   the probabilities P. P is array of probabilities, which are
%   not necessarily normalized, though they must be non-negative,
%   and not all zero. The size of S is the size of P.
%
%   Note that stratified and deterministic resampling have smaller
%   variance.
%
%   Residual resampling retains k_j=floor(n*p_j) copies of index j,
%   where n is length of P and p is normalized probabilities.
%   Additionally n_r=n-k_1-...-k_n indices are sampled with
%   probabilities proportional to n*p_j-k_j. See, e.g., Liu, J. S.,
%   Monte Carlo Strategies in Scientific Computing, Springer, 2001,
%   p. 72.
%
%   See also RESAMPSIM, RESAMPSTR, RESAMPDET

% Copyright (c) 2003-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

n=numel(p);
p=p(:);
pn=p./sum(p).*n;
fpn=floor(pn);
s=zeros(size(p));
k=0;
for i=1:n
  if pn(i)>=1
    a=fpn(i);
    pn(i)=pn(i)-a;
    s(k+[1:a])=i;
    k=k+a;
  end
end
pc=cumsum(pn./(n-k));
pc(end)=1;
s((k+1):n)=binsgeq(pc,rand(1,n-k));
