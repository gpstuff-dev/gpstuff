function lik = lik_gaussian(varargin)
%LIK_LOGIT  Create a Gaussian likelihood 
%
%  Description
%    LIK = LIK_GAUSSIAN returns string 'gaussian', which used
%    instead of other likelihood structures, invokes the special
%    handling of Gaussian likelihood, that is, analytic integration
%    over the noise covariance.
%       
%       See also
%       GP_SET, GPCF_NOISE, GPCF_NOISET, GPCF_*, LIK_*
%

% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  lik='gaussian';

end
