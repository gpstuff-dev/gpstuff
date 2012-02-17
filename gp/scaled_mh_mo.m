function [f, energ, diagn] = scaled_mh_mo(f, opt, gp, x, y, z)
%SCALED_MH  A scaled Metropolis-Hastings sampling for latent values
%
%  Description
%    [F, ENERG, DIAG] = SCALED_MH_MO(F, OPT, GP, X, Y) takes the
%    current latent values F, options structure OPT, Gaussian
%    process structure GP, inputs X and outputs Y. Samples new
%    latent values and returns also energies ENERG and diagnostics
%    DIAG. The latent values are sampled from their conditional
%    posterior p(f|y,th).
%
%    The latent values are whitened with the prior covariance
%    before the sampling. This reduces the autocorrelation and
%    speeds up the mixing of the sampler. See (Neal, 1993) for
%    details on implementation.
%
%    The options structure should include the following fields:
%      repeat              - the number MH-steps before 
%                            returning single sample
%      sample_latent_scale - the scale for the MH-step
%
%  See also
%    GP_MC
  
% Copyright (c) 1999 Aki Vehtari
% Copyright (c) 2006-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

  maxcut = -log(eps);
  mincut = -log(1/realmin - 1);
  lvs=opt.sample_latent_scale;
  a = max(min(f, maxcut),mincut);
  [n,nout] = size(y);
  f = reshape(f,n,nout);
  
        
  if isfield(gp, 'comp_cf')  % own covariance for each ouput component
      multicf = true;
      if length(gp.comp_cf) ~= nout
          error('GPLA_MO_E: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
      end
  else
      multicf = false;
  end
  
  L = zeros(n,n,nout);
  if multicf
      for i1=1:nout
          [tmp, C] = gp_trcov(gp, x, gp.comp_cf{i1});
          L(:,:,i1)=chol(C, 'lower');
      end
  else
      for i1=1:nout
          [tmp, C] = gp_trcov(gp, x);
          L(:,:,i1)=chol(C, 'lower');
      end
  end
  
  e = -gp.lik.fh.ll(gp.lik, y, f, z);
  ft = zeros(size(y));
  
  % Adaptive control algorithm to find such a value for lvs
  % that the rejection rate of Metropolis is optimal.
  slrej = 0;
  for li=1:100
      for i1 =1:nout
          ft(:,i1)=sqrt(1-lvs.^2).*f(:,i1)+lvs.*L(:,:,i1)*randn(n,1);
      end
      ed = -gp.lik.fh.ll(gp.lik, y, ft, z);
      a=e-ed;
      if exp(a) > rand(1)
          f=ft;
          e=ed;
          lvs=min(1,lvs*1.1);
      else
          lvs=max(1e-8,lvs/1.05);
      end
  end
  opt.sample_latent_scale=lvs;
  % Do the actual sampling
  for li=1:(opt.repeat)
      for i1 =1:nout
          ft(:,i1)=sqrt(1-lvs.^2).*f(:,i1)+lvs.*L(:,:,i1)*randn(n,1);
      end
      ed = -gp.lik.fh.ll(gp.lik, y, ft, z);
      a=e-ed;
      if exp(a) > rand(1)
          f=ft;
          e=ed;
      else
          slrej=slrej+1;
      end
  end
  diagn.rej = slrej/opt.repeat;
  diagn.lvs = lvs;
  diagn.opt=opt;
  energ=[];
  f = f(:)';
    
end
