function [w, f] = gpmc_pak(gp_rec)
%GP_PRED Combine parameters of GP-IA cell array into matrix
%
%  Description
%    [W, F] = GPIA_PAK(GP_REC)
%    Combine the sampled hyperparameters in GP_REC into matrix W and in
%    case of non-Gaussian likelihood, the sampled latent values to F.
%
%  See also
%    GPMC_UNPAK, GP_PAK, GP_MC
%
% Copyright (c) 2014 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

if ~isstruct(gp_rec)
  error('Input must proper sampled structure. See for example gp_mc.m.');
end

w=gp_pak(gp_rec);
if isfield(gp_rec, 'latentValues')
  f=gp_rec.latentValues;
else
  f=[];
end

