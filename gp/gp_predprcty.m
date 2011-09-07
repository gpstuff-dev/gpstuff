function prctys = gp_predprcty(gp, x, y, xt, varargin) 
%GP_PREPRCTY  Percentiles of the predictive distribution at test points
%
%  Description
%    PRCTY = GP_PREDPRCTY(GP, X, Y, XT, OPTIONS)
%    takes a GP structure together with matrix X of training
%    inputs and vector Y of training targets, and evaluates the
%    percentiled of the predictive distribution at test inputs XT. 
%
%    OPTIONS is optional parameter-value pair
%      prct   - percentiles to be computed (default = [5 50 95])
%      nsamp  - determines the number of samples used by GP_RND in case of 
%               MCMC or IA (default = 5000).
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case.
%      zt     - optional observed quantity in triplet (xt_i,yt_i,zt_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, the expected 
%               value for the ith case. 
%
%  See also
%    GP_PRED, GP_PAK, GP_UNPAK
%

% Copyright (c) 2011 Ville Tolvanen, Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GP_PREDPRCTY';
  ip.addRequired('gp',@(x) isstruct(x) || iscell(x));
  ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('xt', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('prct', [5 50 95], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('nsamp', 5000, @(x) isreal(x) && all(isfinite(x(:))))
  ip.parse(gp, x, y, xt, varargin{:});
  % pass these forward
  options=struct();
  z = ip.Results.z;
  zt = ip.Results.zt;
  options.z=z;
  options.zt=zt;
  prct = ip.Results.prct;
  nsamp = ip.Results.nsamp;
  
  [tn, nin] = size(x);
  
   % ====================================================
  if isstruct(gp)     % Single GP or MCMC solution
    switch gp.type
      case {'FULL' 'VAR' 'DTC' 'SOR'}
        tstind = [];
      case {'FIC' 'CS+FIC'}
        tstind = 1:tn;
      case 'PIC'
        tstind = gp.tr_index;
    end

    if isfield(gp, 'etr')
      % MCMC solution
      type = 'MCMC';
    else
      % A single GP
      type = 'Single';
    end
    
    switch type
      
      case 'MCMC'
        % MCMC solution
        if isfield(gp.lik.fh, 'trcov')
          % Gaussian likelihood
          [~, sampyt] = gp_rnd(gp,x,y,xt, 'nsamp', nsamp, options);
          prctys = prctile(sampyt, prct, 2);
        else
          % Non-Gaussian likelihood
          [Eft, Varft] = gp_pred(gp,x,y,xt, 'tstind', tstind, options);
          prctys=gp.lik.fh.predprcty(gp.lik, Eft, Varft, zt, prct);
%         if strcmp(gp.type, 'PIC')
%           tr_index = gp.tr_index;
%           gp = rmfield(gp, 'tr_index');
%         else
%           tr_index = [];
%         end
%           for i=1:nsamples
%             Gp = take_nth(gp,i);
%             if  strcmp(gp.type, 'FIC') | strcmp(gp.type, 'PIC')  || strcmp(gp.type, 'CS+FIC') || strcmp(gp.type, 'VAR') || strcmp(gp.type, 'DTC') || strcmp(gp.type, 'SOR')
%               Gp.X_u = reshape(Gp.X_u,length(Gp.X_u)/nin,nin);
%             end
%             Gp.tr_index = tr_index;
%             [Ef, Varf] = gp_pred(Gp, x, y, x, 'yt', y, 'tstind', tstind, options);
%             prctys(:,:,i) = gp.lik.fh.predprcty(gp.lik, Ef, Varf, zt, prct);
%           end
%           prctys = mean(prctys,3);
        end
        
      case 'Single'
        % Single GP 
        
          if isfield(gp.lik.fh,'trcov')
            % Gaussian likelihood
            [~, ~, ~, Eyt, Varyt] = gp_pred(gp,x,y,xt, 'tstind', ...
                                           tstind);
            prct = prct./100;
            prct = norminv(prct, 0, 1);
            prctys = bsxfun(@plus, Eyt, bsxfun(@times, sqrt(Varyt), prct));
          else
            % Non-Gaussian likelihood
            [Eft, Varft] = gp_pred(gp,x,y,xt, 'tstind', tstind, options);
            prctys=gp.lik.fh.predprcty(gp.lik, Eft, Varft, zt, prct);
          end
    end
  
  elseif iscell(gp)
      
    % gp_ia solution
    
    switch gp{1}.type
      case {'FULL' 'VAR' 'DTC' 'SOR'}
        tstind = [];
      case {'FIC' 'CS+FIC'}
        tstind = 1:tn;
      case 'PIC'
        tstind = gp{1}.tr_index;
    end
    
    if isfield(gp{1}.lik.fh, 'trcov')
      % Gaussian likelihood
      [~, sampyt] = gp_rnd(gp,x,y,xt, 'nsamp', nsamp, options);
      prctys = prctile(sampyt, prct, 2);
    else
      % Non-Gaussian likelihood
      [Eft, Varft] = gp_pred(gp,x,y,xt, 'tstind', tstind, options);
      prctys=gp{1}.lik.fh.predprcty(gp{1}.lik, Eft, Varft, zt, prct);
%       sampft = gp_rnd(gp,x,y,xt,'tstind',tstind, options, 'nsamp', nsamp);
%       Eft = mean(sampft,2);
%       Varft = var(sampft,[],2);
%       prctys=gp{1}.lik.fh.predprcty(gp{1}.lik, Eft, Varft, zt, prct);
    end

  end

end
