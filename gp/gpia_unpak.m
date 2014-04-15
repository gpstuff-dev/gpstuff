function gp_array = gpia_unpak(gp_array, w, P_TH)
%GP_PRED Set parameters of GP-IA cell array
%
%  Description
%    GP_ARRAY = GPIA_UPAK(GP_ARRAY, W, P_TH)
%    Set the hyperparameters of the GP_ARRAY for corresponding
%    hyperparameters in W and weights in P_TH
%
%  See also
%    GPIA_PAK, GP_PAK, GP_UNPAK, GP_IA
%
% Copyright (c) 2014 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

if ~iscell(gp_array)
  error('Input must proper GP-IA cell array. See for example gp_ia.m.');
end

nsamples=length(gp_array);
for i1=1:nsamples
  gp_array{i1}=gp_unpak(gp_array{i1}, w(i1,:));
  gp_array{i1}.ia_weight = P_TH(i1);
end

