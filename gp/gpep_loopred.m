function [Eft, Varft, Eyt, Varyt, pyt] = gpep_loopred(gp, x, y, varargin)
%GPEP_LOOPRED  Leave-one-out predictions with EP approximation
%
%  Description
%    [EFT, VARFT, EYT, VARYT, PYT] = GPEP_LOOPRED(GP, X, Y,
%    OPTIONS) takes a Gaussian process structure GP together with a
%    matrix XT of input vectors, matrix X of training inputs and
%    vector Y of training targets, and evaluates the leave-one-out
%    predictive distribution at inputs X. Returns a posterior mean
%    EFT and variance VARFT of latent variables, the posterior
%    predictive mean EYT and variance VARYT of observations, and
%    posterior predictive density PYT at input locations X.
%
%    EP leave-one-out is approximated by leaving-out site-term and
%    using cavity distribution as leave-one-out posterior for the
%    ith latent value. Since the ith likelihood has influenced
%    other site terms through the prior, this estimate can be
%    over-optimistic.
%
%    OPTIONS is optional parameter-value pair
%      z  - optional observed quantity in triplet (x_i,y_i,z_i)
%           Some likelihoods may use this. For example, in case of
%           Poisson likelihood we have z_i=E_i, that is, expected
%           value for ith case.
%
%  See also
%    GPEP_E, GPEP_G, GP_PRED, DEMO_SPATIAL, DEMO_CLASSIFIC
  
% Copyright (c) 2010  Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPEP_LOOPRED';
  ip.addRequired('gp', @(x) isstruct(x) || iscell(x));
  ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.parse(gp, x, y, varargin{:});
  z=ip.Results.z;

  if ~iscell(gp)
    % Single GP
    [~,~,~,~,~,~,~,~,muvec_i,sigm2vec_i,Z_i] = gpep_e(gp_pak(gp), ...
                                                      gp, x, y, 'z', z);

    Eft=muvec_i;
    Varft=sigm2vec_i;
    pyt=Z_i;
    n=length(y);
    if nargout > 2
      for cvi=1:n
        [Eyt(cvi,1), Varyt(cvi,1)] = feval(gp.lik.fh.predy, gp.lik, ...
                                           muvec_i(cvi), sigm2vec_i(cvi));
      end
    end
    
  else
    % Cell array of GPs
    nGP = numel(gp);
    for j = 1:nGP
      [~,~,~,~,~,~,~,~,muvec_i,sigm2vec_i,Z_i] = gpep_e(gp_pak(gp{j}), ...
                                                        gp{j}, x, y, 'z', z);
      
      P_TH(j,:) = gp{j}.ia_weight;
      Eft_grid(j,:)=muvec_i;
      Varft_grid(j,:)=sigm2vec_i;
      pyt_grid(j,:)=Z_i;
      n=length(y);
      if nargout > 2
        for cvi=1:n
          [Eyt_grid(j,cvi), Varyt_grid(j,cvi)] = ...
              feval(gp{j}.lik.fh.predy, gp{j}.lik, muvec_i(cvi), ...
                    sigm2vec_i(cvi), [], z(cvi));
        end
      end
    end
    
    ft = zeros(size(Eft_grid,2),501);
    for j = 1 : size(Eft_grid,2);
        ft(j,:) = Eft_grid(1,j)-10*sqrt(Varft_grid(1,j)) : 20*sqrt(Varft_grid(1,j))/500 : Eft_grid(1,j)+10*sqrt(Varft_grid(1,j));  
    end
    
    % Calculate the density in each grid point by integrating over
    % different models
    pft = zeros(size(Eft_grid,2),501);
    for j = 1 : size(Eft_grid,2)
        pft(j,:) = sum(normpdf(repmat(ft(j,:),size(Eft_grid,1),1), repmat(Eft_grid(:,j),1,size(ft,2)), repmat(sqrt(Varft_grid(:,j)),1,size(ft,2))).*repmat(P_TH,1,size(ft,2)),1); 
    end

    % Normalize distributions
    pft = bsxfun(@rdivide,pft,sum(pft,2));

    % Widths of each grid point
    dft = diff(ft,1,2);
    dft(:,end+1)=dft(:,end);

    % Calculate mean and variance of the distributions
    Eft = sum(ft.*pft,2)./sum(pft,2);
    Varft = sum(pft.*(repmat(Eft,1,size(ft,2))-ft).^2,2)./sum(pft,2);
    
    if nargout > 2
      Eyt = sum(Eyt_grid.*repmat(P_TH,1,size(Eyt_grid,2)),1);
      Varyt = sum(Varyt_grid.*repmat(P_TH,1,size(Eyt_grid,2)),1) + sum((Eyt_grid - repmat(Eyt,nGP,1)).^2, 1);
      Eyt=Eyt';
      Varyt=Varyt';
    end
    pyt = sum(bsxfun(@times,pyt_grid,P_TH),1)';

  end
end
