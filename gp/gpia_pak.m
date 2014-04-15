function [w, P_TH] = gpia_pak(gp_array)
%GP_PRED Combine parameters of GP-IA cell array into matrix
%
%  Description
%    [W, P_TH] = GPIA_PAK(GP_ARRAY)
%    Combine the hyperparameters in GP_ARRAY into matrix W and IA-weights
%    to vector P_TH.
%
%  See also
%    GPIA_UNPAK, GP_PAK, GP_IA
%
% Copyright (c) 2014 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

if ~iscell(gp_array)
  error('Input must proper GP-IA cell array. See for example gp_ia.m.');
end

nsamples=length(gp_array);
P_TH=ones(1,nsamples);
for i1=1:nsamples
  w(i1,:)=gp_pak(gp_array{i1});
  P_TH(i1)=gp_array{i1}.ia_weight;
end

