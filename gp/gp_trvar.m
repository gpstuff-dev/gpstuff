function [K, C] = gp_trvar(gp, x1, predcf)
%GP_TRVAR  Evaluate training variance vector.
%
%  Description
%    K = GP_TRVAR(GP, TX, PREDCF) takes in Gaussian process GP and
%    matrix TX that contains training input vectors to GP. Returns
%    (noiseless) variance vector K for latent values (diagonal of
%    the covariance matrix returned by gp_trcov), which is formed
%    as a sum of the variances from covariance functions in GP.cf
%    array. Every element ij of K contains covariance between
%    inputs i and j in TX. PREDCF is an array specifying the
%    indexes of covariance functions, which are used for forming
%    the matrix. If not given, the matrix is formed with all
%    functions.
%
%    [K, C] = GP_TRCOV(GP, TX, PREDCF) returns also the (noisy)
%    variance vector C, which is sum of K and the variance term
%    for example, from Gaussian noise.
%
%  See also
%    GP_SET, GPCF_*
%
% Copyright (c) 2006, 2016 Jarno Vanhatalo
% Copyright (c) 2010 Tuomas Nikoskinen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% % no covariance functions?
% if length(gp.cf)==0 || (nargin>2 && ~isempty(predcf) && predcf(1)==0) ...
%         || isfield(gp, 'lik_mono')
%     K=[];
%     C=[];
%     if nargout>1 && isfield(gp.lik.fh,'trcov')
%         C=sparse(0);
%         % Add Gaussian noise to the covariance
%         C = C + gp.lik.fh.trvar(gp.lik, x1);
%         if ~isempty(gp.jitterSigma2)
%             C=C+gp.jitterSigma2;
%         end
%     end
%     return
% end

[n,m]=size(x1);
n1 = n+1;
ncf = length(gp.cf);

%if ~(isfield(gp,'derivobs') && gp.derivobs)
% Evaluate the covariance without noise
K = 0;
if nargin < 3 || isempty(predcf)
    predcf = 1:ncf;
end
% loop through covariance functions
for i=1:length(predcf)
    gpcf = gp.cf{predcf(i)};
    if isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude && ...
            (~isfield(gp,'comp_cf') || (isfield(gp,'comp_cf') && sum(gp.comp_cf{1}==predcf)))
        gpcf.magnSigma2=1;
    else
        if isfield(gp,'derivobs') && gp.derivobs
            % derivative observations in use
            K = K + [gpcf.fh.trvar(gpcf, x1) ; gpcf.fh.ginput2(gpcf, x1, x1, 'takeOnlyDiag')];
        else
            % no observations
            K = K + gpcf.fh.trvar(gpcf, x1);
        end
    end
end

if ~isempty(gp.jitterSigma2)
    K = K + gp.jitterSigma2;
end

if nargout >1
    C=K;
    if isfield(gp.lik.fh,'trvar')
        % Add Gaussian noise to the covariance
        C = C + gp.lik.fh.trvar(gp.lik, x1);
    end
end

  