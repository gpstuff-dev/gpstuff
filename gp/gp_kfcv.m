function [criteria, cvpreds, cvws, trpreds, trw, cvtrpreds] = gp_kfcv(gp, x, y, varargin)
%GP_KFCV  K-fold cross validation for GP model
%
%  Description
%    [CRITERIA, CVPREDS, CVWS, TRPREDS, TRW] = GP_KFCV(GP, X, Y, OPTIONS)
%    Performs K-fold cross-validation for a GP model given input matrix X
%    and target vector Y.
%
%    OPTIONS is optional parameter-value pair
%      z          - optional observed quantity in triplet (x_i,y_i,z_i)
%                   Some likelihoods may use this. For example, in
%                   case of Poisson likelihood we have z_i=E_i,
%                   that is, expected value for ith case.
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
%      k          - number of folds in CV
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
%                   - 'off' displays no output
%                   - 'on' (default) gives some output  
%                   - 'iter' displays output at each iteration
%      save_results
%                 - defines if detailed results are stored 'false'
%                   (default) or 'true'. If 'true' gp_kfcv stores the
%                   results in the current working directory into a
%                   cv_resultsX folder (or in 'folder', see next
%                   option), where X is a number. If there are
%                   cv_results* folders already, X is the smallest
%                   number not in use yet.
%       folder    - string defining the folder where to save the
%                   results. That is, the results will be stored in
%                   'current working directory'/folder. See previous
%                   option for default.
%
%    The output arguments are the following
%       criteria  - structure including the following fields
%                     mlpd_cv     - mean log predictive density
%                     Var_lpd_cv  - variance estimate for mlpd
%                     rmse_cv     - root mean squared error
%                     Var_rmse_cv - variance estimate for mrmse
%                     mabs_cv     - mean absolute error
%                     Var_abs_cv  - variance estimate for mabs
%       cvpreds   - CV predictions structure including the same fields
%                   as trpreds
%       trpreds   - training predictions structure including
%                   the following fields:
%                     Eft
%                     Varft
%                     Eyt
%                     Varyt
%                     pyt
%       cvws      - parameter weight vectors for each CV fold
%       trw       - parameter weight vector for training data
%
%     The K-fold cross validation is performed as follows: The data
%     are divided into k groups D_k. For each group, we evaluate
%     the test statistics
%
%            u(D_k | D_{k-1})
%
%     where u is the utility function and D_{k-1} is the data in
%     the k-1 groups other than k. The utility functions provided
%     by gp_kfcv are
%
%       log predictive density
%         lpd(D_k | D_{k-1})  = mean( log( p( y_k|D_{k-1} ) ) )
%       squared error
%         rmse(D_k | D_{k-1}) = mean( ( E[y_k|D_{k-1}] - y_k ).^2 )
%       absolute error
%         abs(D_k | D_{k-1})  = mean( abs( E[y_k|D_{k-1}] - y_k ) )
%
%     After the utility is evaluated for each group, we can
%     evaluate the output arguments, which are obtained as follows
%
%       mean log predictive density
%         mlpd_cv  = mean( lpd(D_k | D_{k-1}) )          ,k=1...K
%       root mean squared error
%         mrmse_cv = sqrt( mean( rmse(D_k | D_{k-1}) ) ) ,k=1...K
%       mean absolute error
%         mabs_cv  = mean( abs(D_k | D_{k-1}) )          ,k=1...K
%
%     The variance estimates for the above statistics are evaluated
%     across the groups K. For mean log predictive density and mean
%     absolute error this reduces to evaluate, for example,
%
%         Var_lpd_cv = var( lpd(D_k | D_{k-1}) ) / K,    k=1...K.
%
%     For root mean squared error, we need to take the square root
%     of each group statistics first to obtain
%
%         Var_rmse_cv = var( sqrt( rmse(D_k | D_{k-1}) ) ) / K,    k=1...K.
%
%     The above statistics are returned by the function. However,
%     if we use the save_results option we obtain some additional
%     test statistics, which are only saved in the result file.
%     These extra statistics include, for example, bias corrected
%     expected utilities (Vehtari and Lampinen, 2002) and the
%     training utility for the whole data and each cross-validation
%     training set. The detailed list of variables saved in the
%     result file is:
%
%
%     For more information see the file cv_results.mat,
%     which contains the following variables
%       lpd_cv      - log predictive density (nx1 vector)
%       rmse_cv     - squared error (nx1 vector)
%       abs_cv      - absolute error (nx1 vector)
%       mlpd_cv     - mean log predictive density (a scalar summary)
%       mrmse_cv    - root mean squared error (a scalar summary)
%       mabs_cv     - mean absolute error (a scalar summary)
%       Var_lpd_cv  - variance of mean log predictive density
%                     (a scalar summary)
%       Var_rmse_cv - variance of the root mean squared error
%                     (a scalar summary)
%       Var_abs_cv  - variance of the mean absolute error
%                     (a scalar summary)
%       trindex     - training indices
%       tstindex    - test indices
%       lpd_cvtr    - mean log predictive density for each of
%                     k-CV training sets (kx1 vector)
%       rmse_cvtr   - root mean squared error for each of
%                     k-CV training sets (kx1 vector)
%       abs_cvtr    - absolute error for each of
%                     k-CV training sets (kx1 vector)
%       lpd_tr      - log predictive density for the
%                     full training set
%       rmse_tr     - root mean squared error for the
%                     full training set
%       abs_tr      - absolute error for the full trainng set
%       lpd_ccv     - log predictive density with corrected k-CV
%       rmse_ccv    - root mean squared error with corrected k-CV
%       abs_ccv     - absolute error with corrected k-CV
%       cpu_time    - the cpu time used for inferring the full
%                     data set
%
%  See also
%    DEMO_MODELASSESMENT1, GP_PEFF, GP_DIC
%
%  References:
%    Spiegelhalter, Best, Carlin and van der Linde (2002). Bayesian
%    measures of model complexity and fit. J. R. Statist. Soc. B,
%    64, 583-639.
%
%    Gelman, Carlin, Stern and Rubin (2004) Bayesian Data Analysis,
%    second edition. Chapman & Hall / CRC.
%
%    Aki Vehtari and Jouko Lampinen. Bayesian model assessment and
%    comparison using cross-validation predictive densities. Neural
%    Computation, 14(10):2439-2468, 2002.
%

%  Experimental features
%      inf_method - inference method. Possible methods are
%                    'LOO'      parameters optimized using leave-one-out
%                    'KFCV'     parameters optimized using k-fold-CV
%                    'WAIC'     parameters optimized using WAIC
  
% Copyright (c) 2009-2010 Jarno Vanhatalo
% Copyright (c) 2010-2011 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GP_KFCV';
  ip.addRequired('gp',@(x) isstruct(x) || iscell(x));
  ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
  ip.addParamValue('inf_method', 'MAP', @(x) ...
    ismember(x,{'MAP' 'LOO' 'KFCV' 'WAIC' 'WAICV' 'WAICG' 'MCMC' 'IA' 'fixed'}))
  ip.addParamValue('optimf', @fminscg, @(x) isa(x,'function_handle'))
  ip.addParamValue('opt', struct(), @isstruct)
  ip.addParamValue('k', 10, @(x) isreal(x) && isscalar(x) && isfinite(x) && x>0)
  ip.addParamValue('rstream', 1, @(x) isreal(x) && isscalar(x) && isfinite(x) && x>0)
  ip.addParamValue('trindex', [], @(x) isempty(x) || iscell(x))
  ip.addParamValue('tstindex', [], @(x) isempty(x) || iscell(x))
  ip.addParamValue('display', 'on', @(x) islogical(x) || ...
                   ismember(x,{'on' 'off' 'iter' 'fold'}))
  ip.addParamValue('save_results', false, @(x) islogical(x))
  ip.addParamValue('folder', [], @(x) ischar(x) )
  ip.parse(gp, x, y, varargin{:});
  z=ip.Results.z;
  inf_method=ip.Results.inf_method;
  optimf=ip.Results.optimf;
  opt=ip.Results.opt;
  k=ip.Results.k;
  rstream=ip.Results.rstream;
  trindex=ip.Results.trindex;
  tstindex=ip.Results.tstindex;
  display = ip.Results.display;
  if isequal(display,'fold');display='iter';end
  save_results=ip.Results.save_results;
  folder = ip.Results.folder;

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
  parent_folder = pwd;

  cvws=[];
  trw=[];
  % loop over the crossvalidation sets
  if ismember(display,{'on','iter'})
    fprintf('\n Evaluating the CV utility\n')
  end
  nargout2 = nargout;
  % parfor enables parallel loop
  parfor i=1:length(trindex)
  

    if isequal(display,'iter')
      fprintf('The CV-iteration number: %d \n', i)
    end

    % Set the training and test sets for i'th cross-validation set
    xtr = x(trindex{i},:);
    ytr = y(trindex{i},:);
    xtst = x(tstindex{i},:);
    ytst = y(tstindex{i},:);

    if ~isempty(z)
      ztr = z(trindex{i},:);
      zt = z;
      yt = y;
%       opt_tst.zt = z;
%       opt_tst.yt = y;
    else
      ztr = [];
      yt = y;
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
        gp=gp_optim(gp,xtr,ytr,'z',ztr,'opt',opt, 'optimf', optimf);
        w=gp_pak(gp);
        cvws(i,:)=w;
      case {'LOO' 'KFCV' 'WAIC' 'WAICV' 'WAICG'}
        gp=gp_optim(gp,xtr,ytr,'z',ztr,'opt',opt,'loss',inf_method, 'optimf', optimf);
        w=gp_pak(gp);
        cvws(i,:)=w;
      case 'MCMC'
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
        gp = gp_ia(gp, xtr, ytr, 'z', ztr, opt);
      case 'fixed'
        % nothing to do here
    end

    if iscell(gp)
      gplik=gp{1}.lik;
    else
      gplik=gp.lik;
    end
    if ~isfield(gplik.fh,'trcov')
      [Eft, Varft, lpyt, Eyt, Varyt] = gp_pred(gp, xtr, ytr, x, 'tstind', tstind2, 'z', ztr, 'yt', yt, 'zt', zt);
    else
      [Eft, Varft, lpyt, Eyt, Varyt] = gp_pred(gp, xtr, ytr, x, 'tstind', tstind2, 'yt', yt);
    end
    % Because parfor loop, must use temporary cells *_cvt/cv, and save
    % results later in cvtpreds and cvpreds structures.
    if nargout2>=6
      Eft_cvt{i}=Eft([trindex{i}(:) ; tstindex{i}(:)],:);
      Varft_cvt{i}=Varft([trindex{i}(:) ; tstindex{i}(:)],:);
      lpyt_cvt{i}=lpyt([trindex{i}(:) ; tstindex{i}(:)],:);      
      Eyt_cvt{i}=Eyt([trindex{i}(:) ; tstindex{i}(:)],:);
      Varyt_cvt{i}=Varyt([trindex{i}(:) ; tstindex{i}(:)],:);
    end
    if nargout2>=2
      Eft_cv{i}=Eft(tstindex{i},:);
      Varft_cv{i}=Varft(tstindex{i},:);
      lpyt_cv{i}=lpyt(tstindex{i},:);      
      Eyt_cv{i}=Eyt(tstindex{i},:);
      Varyt_cv{i}=Varyt(tstindex{i},:);
    end

    % Evaluate statistics
    % Use temporary cells (lpd_cv2, rmse_cv2, abs_cv2) here also.
    lpd_cv2{i} = log(mean(exp(lpyt(tstindex{i},:)),2));
    lpd_cvtr(i) = mean(log(mean(exp(lpyt(trindex{i})),2)));

    rmse_cv2{i} = (mean(Eyt(tstindex{i},:),2) - ytst).^2;
    rmse_cvtr(i) = sqrt(mean((mean(Eyt(trindex{i},:),2) - ytr).^2));

    abs_cv2{i} = abs(mean(Eyt(tstindex{i},:),2) - ytst);
    abs_cvtr(i) = mean(abs(mean(Eyt(trindex{i},:),2) - ytr));

    lpd_cvm(i) = mean(log(mean(exp(lpyt(tstindex{i},:)),2)));
    rmse_cvm(i) = sqrt(mean((mean(Eyt(tstindex{i},:),2) - ytst).^2));
    abs_cvm(i) = mean(abs(mean(Eyt(tstindex{i},:),2) - ytst));
  end

  % Save values from parfor loop to right indices.
  for i=1:length(trindex)
    lpd_cv(tstindex{i}) = lpd_cv2{i};
    rmse_cv(tstindex{i}) = rmse_cv2{i};
    abs_cv(tstindex{i}) = abs_cv2{i};
    if nargout>=6
      cvtrpreds.Eft([trindex{i}(:) ; tstindex{i}(:)],i)=Eft_cvt{i};
      cvtrpreds.Varft([trindex{i}(:) ; tstindex{i}(:)],i)=Varft_cvt{i};
      cvtrpreds.lpyt([trindex{i}(:) ; tstindex{i}(:)],i)=lpyt_cvt{i};
      cvtrpreds.Eyt([trindex{i}(:) ; tstindex{i}(:)],i)=Eyt_cvt{i};
      cvtrpreds.Varyt([trindex{i}(:) ; tstindex{i}(:)],i)=Varyt_cvt{i};
    end
    if nargout>=2
      cvpreds.Eft(tstindex{i},:)=Eft_cv{i};
      cvpreds.Varft(tstindex{i},:)=Varft_cv{i};
      cvpreds.lpyt(tstindex{i},:)=lpyt_cv{i};
      cvpreds.Eyt(tstindex{i},:)=Eyt_cv{i};
      cvpreds.Varyt(tstindex{i},:)=Varyt_cv{i};
    end
  end
  mlpd_cv = mean(lpd_cv);
  mrmse_cv = sqrt(mean(rmse_cv));
  mabs_cv = mean(abs_cv);

  Var_lpd_cv = var(lpd_cvm)./length(trindex);
  Var_rmse_cv = var(rmse_cvm)./length(trindex);
  Var_abs_cv = var(abs_cvm)./length(trindex);

  criteria.mlpd_cv=mlpd_cv;
  criteria.Var_lpd_cv=Var_lpd_cv;
  criteria.mrmse_cv=mrmse_cv;
  criteria.Var_rmse_cv=Var_rmse_cv;
  criteria.mabs_cv=mabs_cv;
  criteria.Var_abs_cv=Var_abs_cv;

  if save_results || nargout >=4
    % compute full training result

    gp = gp_orig;
    if iscell(gp)
      gptype=gp{1}.type;
    else
      gptype=gp.type;
    end
    switch gptype
      case {'FULL' 'VAR' 'DTC' 'SOR'}
        tstind = [];
      case 'FIC'
        tstind = 1:n;
      case 'PIC'
        if iscell(gp)
          tstind = gp{1}.tr_index;
        else
          tstind = gp.tr_index;
        end
    end

    if ~isempty(z)
      opt_tr.z = z;
      opt_tst.zt = z;
      opt_tst.yt = y;
    else
      opt_tr = struct();
      opt_tst.yt = y;
    end

    % Evaluate the training utility
    if ismember(display,{'on','iter'})
      fprintf('\n Evaluating the training utility \n')
    end

    % Conduct inference
    cpu_time = cputime;
    switch inf_method
      case 'MAP'
        gp=gp_optim(gp,x,y,'z',z,'opt',opt_tr, 'optimf', optimf);
        w=gp_pak(gp);
        trw=w;
      case {'LOO' 'KFCV' 'WAIC' 'WAICV' 'WAICG'}
        gp=gp_optim(gp,x,y,'z',z,'opt',opt_tr,'loss',inf_method, 'optimf', optimf);
        w=gp_pak(gp);
        trw=w;
      case 'MCMC'
        gp = gp_mc(gp, x, y, opt_tr, opt);
        nburnin = floor(length(gp.etr)/3);
        gp = thin(gp,nburnin);
      case 'IA'
        gp = gp_ia(gp, x, y, opt_tr, opt);
      case 'fixed'
        % nothing to do here
    end
    cpu_time = cputime - cpu_time;

    % make the prediction
    [Eft, Varft, lpyt, Eyt, Varyt] = gp_pred(gp, x, y, x, 'tstind', tstind, opt_tr, opt_tst);
    if nargout>=4
      trpreds.Eft=Eft;
      trpreds.Varft=Varft;
      trpreds.lpyt=lpyt;      
      trpreds.Eyt=Eyt;
      trpreds.Varyt=Varyt;
    end

    lpd_tr = mean(log(mean(exp(lpyt),2)));
    rmse_tr = sqrt(mean((mean(Eyt,2) - y).^2));
    abs_tr = mean(abs(mean(Eyt,2) - y));

    % compute bias corrected results
    mlpd_ccv =  mlpd_cv +  mean(lpd_tr) -  mean(lpd_cvtr);
    mrmse_ccv =  mrmse_cv +  mean(rmse_tr) -  mean(rmse_cvtr);
    mabs_ccv =  mabs_cv +  mean(abs_tr) -  mean(abs_cvtr);

    criteria.mlpd_ccv=mlpd_ccv;
    criteria.rmse_ccv=mrmse_ccv;
    criteria.mabs_ccv=mabs_ccv;
  end

  if save_results
    % Save the results
    if isempty(folder)
      succes = 1;
      result = 1;
      while succes == 1
        folder = sprintf('cv_results%d', result);
        if exist(['./' folder])
          result = result + 1;
        else
          succes = 0;
        end
      end
    else
      if exist(folder)
        folder_alk = folder;
        succes = 1;
        result = 1;
        while succes == 1
          folder = sprintf([folder '%d'], result);
          if exist(['./' folder])
            result = result + 1;
          else
            succes = 0;
          end
        end
        warning('The given folder: %s exists already. gp_kfcv saves the results in: %s instead.', folder_alk, folder)
      end
    end
    mkdir(folder);

    save([folder '/cv_results.mat'], 'lpd_cv', 'rmse_cv', 'abs_cv', 'mlpd_cv', 'mrmse_cv',...
      'mabs_cv','Var_lpd_cv', 'Var_rmse_cv', 'Var_abs_cv', 'trindex', 'tstindex', 'lpd_cvtr', 'rmse_cvtr',...
      'abs_cvtr', 'lpd_tr', 'rmse_tr', 'abs_tr', 'mlpd_ccv', 'mrmse_ccv', 'mabs_ccv', 'cpu_time', 'cvpreds');

    if ismember(display,{'on','iter'})
      fprintf('The results have been saved in the folder:\n %s/%s \n', parent_folder, folder);
    end

    f = fopen([folder '/description.txt'],'w');
    fprintf(f,'The cv results were the following: \n\n');
    fprintf(f,'mlpd_cv = %.4f (+/- %.4f) \n', mlpd_cv, Var_lpd_cv);
    fprintf(f,'mrmse_cv = %.4f (+/- %.4f)  \n', mrmse_cv, Var_rmse_cv);
    fprintf(f,'mabs_cv = %.4f (+/- %.4f) \n', mabs_cv, Var_abs_cv);
    fprintf(f,'cpu time = %.4f \n', cpu_time);
    fprintf(f,'\n\n');

    fprintf(f,'For more information see the file cv_results.mat, which contains ');
    fprintf(f,'the following variables\n\n');
    fprintf(f,'lpd_cv      = log predictive density (nx1 vector) \n');
    fprintf(f,'rmse_cv     = squared error (nx1 vector) \n');
    fprintf(f,'abs_cv      = absolute error (nx1 vector) \n');
    fprintf(f,'mlpd_cv     = mean log predictive density (a scalar summary) \n');
    fprintf(f,'mrmse_cv    = root mean squared error (a scalar summary) \n');
    fprintf(f,'mabs_cv     = mean absolute error (a scalar summary) \n');
    fprintf(f,'Var_lpd_cv  = variance of mean log predictive density (a scalar summary) \n');
    fprintf(f,'Var_rmse_cv = variance of the root mean squared error (a scalar summary) \n');
    fprintf(f,'Var_abs_cv  = variance of the mean absolute error (a scalar summary) \n');
    fprintf(f,'trindex     = training indices \n');
    fprintf(f,'tstindex    = test indices \n');
    fprintf(f,'lpd_cvtr    = mean log predictive density for each of k-CV trainng sets (kx1 vector) \n');
    fprintf(f,'rmse_cvtr   = root mean squared error for each of k-CV trainng sets (kx1 vector) \n');
    fprintf(f,'abs_cvtr    = absolute error for each of k-CV training sets (kx1 vector) \n');
    fprintf(f,'lpd_tr      = log predictive density for the full training set  \n');
    fprintf(f,'rmse_tr     = root mean squared error for the full training set  \n');
    fprintf(f,'abs_tr      = absolute error for the full training set  \n');
    fprintf(f,'lpd_ccv     = log predictive density with corrected cross validation   \n');
    fprintf(f,'rmse_ccv    = root mean squared error with corrected cross validation \n');
    fprintf(f,'abs_ccv     = absolute error with corrected cross validation \n');
    fprintf(f,'cpu_time    = The cpu time used for inferring the full data set \n');
    fprintf(f,'cvpreds     = CV predictions structure  \n');
    fclose(f);
  end
end
