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
% Copyright (c) 2006 Jarno Vanhatalo
% Copyright (c) 2010 Tuomas Nikoskinen

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

[n,m]=size(x1);
ncf = length(gp.cf);
if nargin < 3 || isempty(predcf)
    predcf = 1:ncf;
end
K = zeros(n,1);
  
if ~(isfield(gp,'deriv') && ~gp.deriv==0)
  % Evaluate the covariance without noise
  if isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude && ...
      (~isfield(gp,'comp_cf') || (isfield(gp,'comp_cf') && sum(gp.comp_cf{1}==predcf)))
    K=ones(n,1);
  else
    for i=1:length(predcf)
      gpcf = gp.cf{predcf(i)};
      K = K + gpcf.fh.trvar(gpcf, x1);
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
else
  % Derivative observations
  
  ind_Ddim = x1(:,gp.deriv);
  ind_Ddim_derivs = ind_Ddim(ind_Ddim>0);
  uDdim = unique(ind_Ddim_derivs);
  x1 = x1(:,setdiff(1:m,gp.deriv));   % Take only the non-index columns
  if any(strcmp(gp.type,{'FIC' 'PIC' 'PIC_BLOCK' 'CS+FIC' 'VAR' 'DTC' 'SOR'}))
      error('derivative observations have not been implemented for sparse GPs')
  end
  
  
  for i=1:length(predcf)
      gpcf = gp.cf{predcf(i)};
      Ktemp = zeros(n,1);
      if (~isfield(gpcf, 'selectedVariables') || any(ismember(gpcf.selectedVariables,uDdim)))      
          if size(x1,2) <2    % One dimensional input
              Kff = gpcf.fh.trvar(gpcf, x1(ind_Ddim==0,:));
              D = gpcf.fh.ginput2(gpcf, x1(ind_Ddim==1,:), x1(ind_Ddim==1,:),1,'takeOnlyDiag');
              Ktemp = [Kff ; D{1}];
          else
              % the block of covariance matrix
              Ktemp(ind_Ddim==0) = gpcf.fh.trvar(gpcf, x1(ind_Ddim==0,:));
              for u1 = 1:length(uDdim)
                  D = gpcf.fh.ginput2(gpcf, x1(ind_Ddim==uDdim(u1),:), x1(ind_Ddim==uDdim(u1),:), uDdim(u1),'takeOnlyDiag');
                  Ktemp(ind_Ddim==uDdim(u1)) = D{1};                  
              end
          end
      else
          Ktemp(ind_Ddim==0) = gpcf.fh.trvar(gpcf, x1(ind_Ddim==0,:));
      end
      K= K + Ktemp;
      
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
    C(C<eps)=0;
  end
  K(K<eps)=0;
  
end