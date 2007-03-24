function s = resampsim(p);
%RESAMPSIM Simple random resampling
%
%   Description:
%   S = RESAMPSIM(P) returns a new set of indices according to 
%   the probabilities P. P is array of probabilities, which are
%   not necessarily normalized, though they must be non-negative,
%   and not all zero. The size of S is the size of P.
%
%   Note that residual, stratified and deterministic resampling all
%   have smaller variance.
%
%   Simple random resampling samples indices randomly according
%   to the probabilities P. See, e.g., Liu, J. S., Monte Carlo
%   Strategies in Scientific Computing, Springer, 2001, p. 72.
%
%   See also RESAMPRES, RESAMPSTR, RESAMPDET

% Copyright (c) 2003-2004 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

pc=cumsum(p(:));
pc=pc./pc(end);
s=binsgeq(pc,rand(size(p)));
