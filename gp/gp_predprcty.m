function prctys = gp_predprcty(gp, x, y, xt, varargin) 
%GP_PREPRCTY  Percentiles for test inputs
%
%  Description
%    PRCTY = GP_PREDPRCTY(GP, X, Y, XT, OPTIONS)
%    Calculates percentiles for test inputs XT in posterior predictive
%    distribution.
%
%    OPTIONS is optional parameter-value pair
%      prcty  - percentiles to be calculated by gp_predprcty (default = 
%               [5 50 95])
%      nsamp  - determines the number of samples used by gp_rnd in case of 
%               MCMC or IA (default = 5000).
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case.
%
%  See also
%    GP_PRED, GP_PAK, GP_UNPAK
%


% Copyright (c) 2011 Ville Tolvanen



  ip=inputParser;
  ip.FunctionName = 'GP_PREDPRCTY';
  ip.addRequired('gp',@(x) isstruct(x) || iscell(x));
  ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('xt', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('prcty', [5 50 95], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('nsamp', 5000, @(x) isreal(x) && all(isfinite(x(:))))
  ip.parse(gp, x, y, xt, varargin{:});
  % pass these forward
  options=struct();
  z = ip.Results.z;
  prcty = ip.Results.prcty;
  nsamp = ip.Results.nsamp;
  if ~isempty(ip.Results.z)
    options.zt=ip.Results.z;
    options.z=ip.Results.z;
  end
  
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
        
        [sampft, sampyt] = gp_rnd(gp,x,y,xt, 'nsamp', nsamp, options);
        prctys = prctile(sampyt, prcty, 2);
        
      case 'Single'
        % Single GP 
        
        [Ef, Varf, ~, Ey, Vary] = gp_pred(gp,x,y,xt, 'yt', ones(size(xt,1),1), 'tstind', tstind, options);
          
          if isfield(gp.lik.fh,'trcov')
            % Gaussian likelihood
            
            prcty = prcty./100;
            prcty = norminv(prcty, 0, 1);
            prctys = bsxfun(@plus, Ey, bsxfun(@times, sqrt(Vary), prcty));
          else
            % Non-Gaussian likelihood
            
            prcty = prcty./100;
            prcty = norminv(prcty, 0, 1);
            n = length(prcty);
            prctys = zeros(size(Ef,1),n);
            for i=1:n
              prctys(:,i) = gp.lik.fh.invlink(gp.lik, Ef+prcty(i).*sqrt(Varf), z);
            end
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
    
    [~, sampyt] = gp_rnd(gp,x,y,xt, 'nsamp', nsamp, 'tstind', tstind, options);
    prctys = prctile(sampyt, prcty, 2);

  end

end
