function gp_rec = gpmc_unpak(gp_rec, w, varargin)
%GPMC_UNPAK Set parameters of GP-MC structure
%
%  Description
%    GP_REC = GPMC_UNPAK(GP, W)
%    Given Gaussian process GP and hyperparameter weight matrix W,
%    create GP-MC structure with hyperparameter values filled from W.
%
%  See also
%    GPMC_PAK, GP_PAK, GP_UNPAK, GP_MC
%
% Copyright (c) 2014 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

if ~isstruct(gp_rec)
  error('Input GP_REC must be either sampled GP (see gp_mc) or normal GP structure (see gp_set).');
end

ncf=length(gp_rec.cf);
nsamples=size(w,1);

% Set covariace function paramaters
ind=0;
if isfield(gp_rec, 'etr')
  gptmp=take_nth(gp_rec,1);
else
  gptmp=gp_rec;
end
for i1=1:ncf
  gpcf=gptmp.cf{i1};
  np=size(gpcf.fh.pak(gpcf),2);
  gpcftmp=gpcf;
  reccf=gpcftmp.fh.recappend(gpcftmp, gpcftmp);
  for i2=1:nsamples
    gpcftmp=gpcftmp.fh.unpak(gpcftmp, w(i2, ind+1:ind+np));
    reccf=gpcftmp.fh.recappend(reccf, i2, gpcftmp);
  end
  ind=ind+np;
  gp_rec.cf{i1}=reccf;
end
% Set likelihood function parameters
lik=gptmp.lik;
reclik=lik.fh.recappend(lik, lik);
if ind < size(w,2)
  np=size(lik.fh.pak(lik),2);
  for i2=1:nsamples
    lik=lik.fh.unpak(lik, w(i2, ind+1:ind+np));
    reclik=lik.fh.recappend(reclik, i2, lik);
  end
  gp_rec.lik=reclik;
end


if nargin==3
  gp_rec.latentValues=varargin{1};
end