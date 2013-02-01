function [cdf] = gp_kfcv_cdf(gp,x,y,varargin)
%GP_KFCV_CDF  K-fold cross validation to predict CDF for GP model
%
%  Description
%    [cdf] = GP_KFCV_CDF(GP, X, Y, OPTIONS)
%    Performs K-fold cross-validation for a GP model given input matrix X
%    and target vector Y. 
%
%    OPTIONS is optional parameter-value pair
%      z          - optional observed quantity in triplet (x_i,y_i,z_i)
%                   Some likelihoods may use this. For example, in
%                   case of Poisson likelihood we have z_i=E_i,
%                   that is, expected value for ith case.
%      yt         - optional observed yt in test points. 
%                   Default option for yt is yt=y.
%      inf_method - inference method. Possible methods are
%                    'MAP'      parameters optimized to MAP (default)
%                    'MCMC'     MCMC sampling using GP_MC
%                    'IA'       integration approximation using GP_IA
%                    'fixed'    parameters are fixed, it either use MAP 
%                               or integration approximation, depending if 
%                               GP is a single GP structure or a GP array
%                               (for example from GP_IA)
%      optimf     - function handle for an optimization function, which is
%                   assumed to have similar input and output arguments
%                   as usual fmin*-functions. Default is @fminscg.
%      opt        - options for the inference method. If 'MAP' is used
%                   use optimset to set options for optimization.
%                   Default options for optimization are 'GradObj'
%                   is 'on', 'LargeScale' is 'off', 'Display' is 'off'
%      k          - number of folds in CV, default k=10
%      rstream    - number of a random stream to be used for
%                   permuting the data befor division. This way
%                   same permutation can be obtained for different
%                   models. Default is 1. See doc RandStream for
%                   more information.
%      trindex    - k-fold CV training indices. A cell array with k
%                   fields each containing index vector for respective
%                   training set.
%      tstindex   - k-fold CV test indices. A cell array with k
%                   fields each containing index vector for
%                   respective test set.
%      display    - defines if messages are displayed. 
%                   - 'iter' displays output at each iteration
%
%    The output argument is 
%
%           cdf- Predictive Cumulative Distribiton function evaluated in y
%           (default) or yt.
%
%
%  See also
%    DEMO_MODELCOMPARISON1, DEMO_MODELCOMPARISON2
%
% Copyright (c) 2009-2010 Jarno Vanhatalo
% Copyright (c) 2010-2011 Aki Vehtari
% Copyright (c) 2012 Ernesto Ulloa


% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GP_KFCV';
  ip=iparser(ip,'addRequired','gp',@(x) isstruct(x) || iscell(x));
  ip=iparser(ip,'addRequired','x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'addRequired','y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'addParamValue','z', [], @(x) isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'addParamValue','yt', [], @(x) isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'addParamValue','inf_method', 'MAP', @(x) ...
    ismember(x,{'MAP' 'LOO' 'KFCV' 'WAIC' 'WAICV' 'WAICG' 'MCMC' 'IA' 'fixed'}));
  ip=iparser(ip,'addParamValue','optimf', @fminscg, @(x) isa(x,'function_handle'));
  ip=iparser(ip,'addParamValue','opt', struct(), @isstruct);
  ip=iparser(ip,'addParamValue','k', 10, @(x) isreal(x) && isscalar(x) && isfinite(x) && x>0);
  ip=iparser(ip,'addParamValue','rstream', 1, @(x) isreal(x) && isscalar(x) && isfinite(x) && x>0);
  ip=iparser(ip,'addParamValue','trindex', [], @(x) isempty(x) || iscell(x));
  ip=iparser(ip,'addParamValue','tstindex', [], @(x) isempty(x) || iscell(x));
  ip=iparser(ip,'addParamValue','display', 'on', @(x) islogical(x) || ...
                   ismember(x,{'iter' 'fold'}));
  ip=iparser(ip,'parse',gp, x, y, varargin{:});
  z=ip.Results.z;
  yt=ip.Results.yt;
  inf_method=ip.Results.inf_method;
  optimf=ip.Results.optimf;
  opt=ip.Results.opt;
  k=ip.Results.k;
  rstream=ip.Results.rstream;
  trindex=ip.Results.trindex;
  tstindex=ip.Results.tstindex;
  display = ip.Results.display;
  if isequal(display,'fold');display='iter';end

  [n,nin] = size(x);

  gp_orig = gp;

  if ismember(inf_method,{'MAP' 'LOO' 'KFCV' 'WAIC' 'WAICV' 'WAICG'})
    optdefault=struct('Display','off');
    opt=optimset(optdefault,opt);
  end

  if (isempty(trindex) && ~isempty(tstindex)) || (~isempty(trindex) && isempty(tstindex))
    error('gp_kfcv: If you give cross-validation indexes, you need to provide both trindex and tstindex.')
  end

  if isempty(trindex) || isempty(tstindex)
    [trindex, tstindex] = cvit(n, k, rstream);
  end
  
   % *** note: yt must can be: a scalar or a vector of size(y)x1 or even a matrix of
   % size(y)xN
 
    if ~isempty(yt)
       if size(yt,1)==size(y,1)
       yt=yt;
       elseif size(yt,1)==1
       yt=bsxfun(@times,ones(size(y)),yt);
       else
       error('size of yt does not match y nor is it a scalar');   
       end
       ytflag=1;
    else
    ytflag=0;
    yt=y;
    end

  cvws=[];
  trw=[];
  % loop over the crossvalidation sets
  
  nargout2 = nargout;
  
  % parfor enables parallel loop
  %parfor i=1:length(trindex)
  
  for i=1:length(trindex)
    if isempty(tstindex{i})
      continue
    end

    if isequal(display,'iter')
      fprintf('The CV-iteration number: %d \n', i)
    end
    
    % Set the training and test sets for i'th cross-validation set
    xtr = x(trindex{i},:);
    ytr = y(trindex{i},:);
    yttr= yt(trindex{i},:);
    
    xtst = x(tstindex{i},:);
    ytst = y(tstindex{i},:);
    yttst = yt(tstindex{i},:);
    
      if ~isempty(z)
          ztr = z(trindex{i},:);
          zt = z;
          flagz=1;
         %* yt = y;
    %       opt_tst.zt = z;
    %       opt_tst.yt = y;
      else
          flagz=0;
          ztr = [];
          %*yt = y;
          zt = [];
    %       opt_tr = struct();
    %       opt_tst.yt = y;
     end

    gp = gp_orig;

    if iscell(gp)
      gptype=gp{1}.type;
    else
      gptype=gp.type;
    end
    tstind2 = [];
    
    switch gptype
      case {'FIC' 'CS+FIC'}
        tstind2 = trindex{i};
      case 'PIC'
        % Set the block indices for the cv set of data points. Variable
        % naming(e.g tstind2) because parfor loop.
        ntr = size(xtr,1);
        ntst = size(xtst,1);
        trind2 = [];
        for i1=1:length(gp.tr_index)
          tstind2{i1} = [];
          trind2{i1} = [];
          for j1 = 1:length(gp.tr_index{i1})
            indtmp = find( sum((xtr - repmat(x(gp.tr_index{i1}(j1),:),ntr,1)).^2,2) == 0 );
            if isempty( indtmp )
              indtmp = find( sum((xtst - repmat(x(gp.tr_index{i1}(j1),:),ntst,1)).^2,2) == 0 );
              tstind2{i1} = [tstind2{i1} indtmp];
            else
              trind2{i1} = [trind2{i1} indtmp];
            end
          end
        end
        if iscell(gp)
          for j=1:numel(gp)
            gp{j}.tr_index=trind2;
          end
        else
          gp.tr_index = trind2;
        end
    end
    
    % Conduct inference
    switch inf_method
      case 'MAP'
            if flagz
                gp=gp_optim(gp,xtr,ytr,'z',ztr(:,size(z,2)),'opt',opt, 'optimf', optimf);
                w=gp_pak(gp);
                cvws(i,:)=w;
            else
                gp=gp_optim(gp,xtr,ytr,'z',ztr,'opt',opt, 'optimf', optimf);
                w=gp_pak(gp);
                cvws(i,:)=w;
            end
      case {'LOO' 'KFCV' 'WAIC' 'WAICV' 'WAICG'}
        if ismember('optimf',ip.UsingDefaults)

              if flagz
              gp=gp_optim(gp,xtr,ytr,'z',ztr(:,size(z,2)),'opt',opt,'loss',inf_method);
              else
              gp=gp_optim(gp,xtr,ytr,'z',ztr,'opt',opt,'loss',inf_method);  
              end

        else
              if flagz
              gp=gp_optim(gp,xtr,ytr,'z',ztr(:,size(z,2)),'opt',opt,'loss',inf_method, 'optimf', optimf);
              else
              gp=gp_optim(gp,xtr,ytr,'z',ztr,'opt',opt,'loss',inf_method, 'optimf', optimf);
              end
        end
        w=gp_pak(gp);
        cvws(i,:)=w;
      case 'MCMC'
        if numel(gp.jitterSigma2)>1
          gp=thin(gp,numel(gp.jitterSigma2)-1);
        end
        % Scaled mixture noise model is a special case
        % where we need to modify the noiseSigmas2 vector
        % to a right length
        if isequal(gp.lik.type, 'lik_smt')
            gp.lik.noiseSigmas2 = gp_orig.lik.noiseSigmas2(trindex{i});
            gp.lik.r = gp_orig.lik.r(trindex{i});
            gp.lik.U = gp_orig.lik.U(trindex{i});
            gp.lik.ndata = length(trindex{i});
        end
        % Pick latent values for the training set in this fold
        if isfield(gp,'latentValues')
          gp.latentValues=gp_orig.latentValues(trindex{i});
        end
        gp = gp_mc(gp, xtr, ytr, 'z', ztr, opt);
        nburnin = floor(length(gp.etr)/3);
        gp = thin(gp,nburnin);
      case 'IA'
        if flagz
        gp = gp_ia(gp, xtr, ytr, 'z', ztr(:,size(z,2)), opt);
        else
        gp = gp_ia(gp, xtr, ytr, 'z', ztr, opt);
        end
      case 'fixed'
        % nothing to do here
    end

    if iscell(gp)
      gplik=gp{1}.lik;
    else
      gplik=gp.lik;
    end
    
    if ~isfield(gplik.fh,'trcov') && isfield(gp.lik.fh,'predcdf')
        for it=1:size(yt,2)
        [Eft, Varft] = gp_pred(gp, xtr, ytr, x, 'tstind', tstind2, 'z', ztr(:,it), 'yt', yt(:,it), 'zt', zt(:,it));
        cdftemp{it}= gp.lik.fh.predcdf(gplik, Eft, Varft,yt(:,it));
        end
    elseif isfield(gplik.fh,'trcov') && isfield(gp.lik.fh,'predcdf')
       for it=1:size(yt,2)
       [Eft, Varft] = gp_pred(gp, xtr, ytr, x, 'tstind', tstind2,'yt', yt(:,it));
       cdftemp{it}= gp.lik.fh.predcdf(gplik, Eft, Varft,yt(:,it));
       end
    else
        error('This likelihood has not been implemented for this function')
    end
    
    % Because parfor loop, must use temporary cells *_cvt/cv, and save
    % results later in cvtpreds and cvpreds structures.
    
    for it=1:size(yt,2)
    cdf_cv(tstindex{i},it)=cdftemp{it}(tstindex{i},:);
    end
    
  end
  
  cdf=cdf_cv;
  
  % Save values from parfor loop to right indices.
   
    %  
    %   for i=1:length(trindex)
    %     if isempty(tstindex{i})
    %       continue
    %     end
    %     for it=1:size(yt,2)
    %     cdf(tstindex{i},it)=cdf_cv(i,it);
    %     end
    %   end
  
end

