function [g, gdata, gprior] = gp2_g(w, gp, x, y, varargin)
%GP_G  Evaluate the gradient of energy (GP_E) for Gaussian Process
%
%  Description
%    G = GP_G(W, GP, X, Y, OPTIONS) takes a full GP parameter
%    vector W, GP structure GP, a matrix X of input vectors and a
%    matrix Y of target vectors, and evaluates the gradient G of
%    the energy function (gp_e). Each row of X corresponds to one
%    input vector and each row of Y corresponds to one target
%    vector.
%
%    [G, GDATA, GPRIOR] = GP_G(W, GP, X, Y, OPTIONS) also returns
%    separately the data and prior contributions to the gradient.
%
%    OPTIONS is optional parameter-value pair
%      z - optional observed quantity in triplet (x_i,y_i,z_i)
%          Some likelihoods may use this. For example, in case of
%          Poisson likelihood we have z_i=E_i, that is, expected
%          value for ith case.
%
%  See also
%    GP_E, GP_PAK, GP_UNPAK, GPCF_*
%

% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2010 Heikki Peura

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

if isfield(gp,'latent_method') && ~strcmp(gp.latent_method,'MCMC')
  % use an inference specific method
  fh_g=gp.fh.g;
  switch nargout 
    case 1
      [g] = fh_g(w, gp, x, y, varargin{:});
    case 2
      [g, gdata] = fh_g(w, gp, x, y, varargin{:});
    case 3
      [g, gdata, gprior] = fh_g(w, gp, x, y, varargin{:});
  end
  return
end

ip=inputParser;
ip.FunctionName = 'GP_G';
ip.addRequired('w', @(x) isvector(x) && isreal(x) && all(isfinite(x)));
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.parse(w, gp, x, y, varargin{:});
z=ip.Results.z;

% unpak the parameters
gp=gp_unpak(gp, w);
ncf = length(gp.cf);
switch gp.lik.type
  case {'LGP', 'LGPC', 'Coxph'}
    error('GP2_G not implemented for this type of likelihood');
  case {'Softmax', 'Multinom'}    
    [n,nout]=size(y);
  otherwise
    n=size(y,1);
    nout=length(gp.comp_cf);
end

g = [];
gdata = [];
gprior = [];

switch gp.type
  case 'FULL'
    % ============================================================
    % FULL
    % ============================================================
    % Evaluate covariance
    
    if isfield(gp, 'comp_cf')  % own covariance for each ouput component
        multicf = true;
        if length(gp.comp_cf) ~= nout
            error('GP2_G: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
        end
    else
        multicf = false;
    end
    
    b = zeros(n,nout);
    invC = zeros(n,n,nout);
    if multicf
        for i1=1:nout
            [tmp, C] = gp_trcov(gp, x, gp.comp_cf{i1});
            invC(:,:,i1) = inv(C);
            b(:,i1) = C\y(:,i1);
        end
    else
        [tmp, C] = gp_trcov(gp, x);
        invCtmp = inv(C);
        for i1=1:nout
            invC(:,:,i1) = invCtmp;
            b(:,i1) = C\y(:,i1);
        end
    end
        
    % =================================================================
    % Gradient with respect to covariance function parameters
    if ~isempty(strfind(gp.infer_params, 'covariance'))
      for i=1:ncf
        i1=0;
        if ~isempty(gprior)
          i1 = length(gprior);
        end
        
        % check in which components the covariance function is present
        do = false(nout,1);
        if multicf
            for z1=1:nout
                if any(gp.comp_cf{z1}==i)
                    do(z1) = true;
                end
            end
        else
            do = true(nout,1);
        end
        
        gpcf = gp.cf{i};
        DKff = gpcf.fh.cfg(gpcf, x);
        gprior_cf = -gpcf.fh.lpg(gpcf);
        
        % Evaluate the gradient with respect to covariance function
        % parameters
        for i2 = 1:length(DKff)
            i1 = i1+1;
            Bdl=0; Cdl=0;
            for z1=1:nout
                if do(z1)
                    Bdl = Bdl + b(:,z1)'*(DKff{i2}*b(:,z1));
                    Cdl = Cdl + sum(sum(invC(:,:,z1).*DKff{i2})); % help arguments
                end
            end
            gdata(i1)=0.5.*(Cdl - Bdl);
            gprior(i1) = gprior_cf(i2);
        end
                
        % Set the gradients of hyperparameter
        if length(gprior_cf) > length(DKff)
          for i2=length(DKff)+1:length(gprior_cf)
            i1 = i1+1;
            gdata(i1) = 0;
            gprior(i1) = gprior_cf(i2);
          end
        end    
      end
    end
    
    % =================================================================
    % Gradient with respect to Gaussian likelihood function parameters
    if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh,'trcov')
      % Evaluate the gradient from Gaussian likelihood
      DCff = gp.lik.fh.cfg(gp.lik, x);
      gprior_lik = -gp.lik.fh.lpg(gp.lik);
            
      for i2 = 1:length(DCff)
        i1 = i1+1;
          if size(DCff{i2}) > 1
            yKy = b'*(DCff{i2}*b);
            trK = sum(sum(invC.*DCff{i2})); % help arguments
            gdata_zeromean(i1)=0.5.*(trK - yKy);
          else 
            yKy=DCff{i2}.*(b'*b);
            trK = DCff{i2}.*(trace(invC));
            gdata_zeromean(i1)=0.5.*(trK - yKy);
          end
          gdata(i1)=gdata_zeromean(i1);
      
        gprior(i1) = gprior_lik(i2);
      end
      
      % Set the gradients of hyperparameter
      if length(gprior_lik) > length(DCff)
        for i2=length(DCff)+1:length(gprior_lik)
          i1 = i1+1;
          gdata(i1) = 0;
          gprior(i1) = gprior_lik(i2);
        end
      end
    end
    
    g = gdata + gprior;
  otherwise
        error('unknown type of covariance function')
end
