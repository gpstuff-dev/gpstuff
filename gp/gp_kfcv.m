function [criteria, cvpreds, cvws, trpreds, trw, cvtrpreds] = gp_kfcv(gp, x, y, varargin)
%GP_KFCV        K-fold cross validation for GP model
%
%	Description
%	[criteria, cvpreds, cvws, trpreds, trw] = gp_kfcv(gp, x, y, OPTIONS)
%
%       Perform K-fold cross validation for GP model.
%
%       The mandatory input arguments are the following:
%         gp           - GP data structure containing the model
%         x            - inputs
%         y            - outputs
%
%       OPTIONS is optional parameter-value pair
%         'z'          - optional observed quantity in triplet (x_i,y_i,z_i)
%                         Some likelihoods may use this. For example, Poisson
%         'inf_method' - inference method. Possible methods are
%                         'MAP_scg2'     hyperparameter optimization with SCG
%                         'MAP_fminunc'  hyperparameter optimization with fminunc
%                         'MCMC'         MCMC sampling using gp_mc
%                         'IA'           integration approximation using gp_ia
%                         The default is 'MAP_scg2'
%         'opt'        - options for the inference method
%         'k'          - number of folds in CV  
%         'rstream'    - number of a random stream to be used for
%                        permuting the data befor division. This way same 
%                        permutation can be obtained for different models.
%         'trindex'    - k-fold CV training indices. A cell array with k 
%                        fields each containing index vector for respective 
%                        training set. 
%         'tstindex'   - k-fold CV test indices. A cell array with k fields 
%                        each containing index vector for
%                        respective test set. 
%         'save_results'- defines if results are stored 'false' or 'true'. 
%                        By default false. If 'true' gp_kfcv stores
%                        the results in the current working
%                        directory into a cv_resultsX folder (or in 
%                        'folder', see next option), where
%                        X is a number. If there are cv_results*
%                        folders already, X is the smallest number
%                        not in use yet.
%         'folder'     - string defining the folder where to save the 
%                        results. That is, the results will be stored in
%                        'current working directory'/folder. See previous 
%                        option for default.
%                        
%
%       The output arguments are the following
%         criteria     - structure including the following fields
%                         mlpd_cv     - mean log predictive density
%                         Var_lpd_cv  - variance estimate for mlpd
%                         rmse_cv     - root mean squared error
%                         Var_rmse_cv - variance estimate for mrmse
%                         mabs_cv     - mean absolute error
%                         Var_abs_cv  - variance estimate for mabs
%         cvpreds       - CV predictions structure including the same fields 
%                         as trpreds
%         trpreds       - training predictions structure including 
%                         the following fields
%                         Ef
%                         Varf
%                         Ey
%                         Vary
%                         py
%         cvws          - hyperparameter weight vectors for each CV fold
%         trw           - hyperparameter weight vector for training data
%
%       The K-fold cross validation is performed as follows: The
%       data are divided into k groups D_k. For each group, we
%       evaluate the test statistics
%
%            u(D_k | D_{k-1})
% 
%       where u is the utility function and D_{k-1} is the data in
%       the k-1 groups other than k. The utility functions provided
%       by gp_kfcv are
%
%        log predictive density  
%            lpd(D_k | D_{k-1})  = mean( log( p( y_k|D_{k-1} ) ) )
%        squared error  
%            rmse(D_k | D_{k-1}) = mean( ( E[y_k|D_{k-1}] - y_k ).^2 )
%        absolute error  
%            abs(D_k | D_{k-1})  = mean( abs( E[y_k|D_{k-1}] - y_k ) )
%
%       After the utility is evaluated for each group, we can
%       evaluate the output arguments, which are obtained as follows
%    
%        mean log predictive density  
%            mlpd_cv  = mean( lpd(D_k | D_{k-1}) )          ,k=1...K
%        root mean squared error  
%            mrmse_cv = sqrt( mean( rmse(D_k | D_{k-1}) ) ) ,k=1...K
%        mean absolute error
%            mabs_cv  = mean( abs(D_k | D_{k-1}) )          ,k=1...K
%    
%       The variance estimates for the above statistics are
%       evaluated across the groups K. For mean log predictive
%       density and mean absolute error this reduces to evaluate,
%       for example,
%
%            Var_lpd_cv = var( lpd(D_k | D_{k-1}) ) / K,    k=1...K.
%
%       For root mean squared error, we need to take the
%       square root of each group statistics first to obtain
%
%            Var_rmse_cv = var( sqrt( rmse(D_k | D_{k-1}) ) ) / K,    k=1...K.
%   
%       The above statistics are returned by the function. However,
%       if we use the save_results option we obtain some additional
%       test statistics, which are only saved in the result file. These
%       extra statistics include, for example, bias corrected
%       expected utilities (Vehtari and Lampinen, 2002) and the
%       training utility for the whole data and each cross-validation
%       training set. The detailed list of variables saved in the
%       result file is:
%
%
%                For more information see the file cv_results.mat,
%                which contains the following variables
%                lpd_cv      = log predictive density (nx1 vector)
%                rmse_cv     = squared error (nx1 vector)
%                abs_cv      = absolute error (nx1 vector)
%                mlpd_cv     = mean log predictive density (a scalar summary)
%                mrmse_cv    = root mean squared error (a scalar summary)
%                mabs_cv     = mean absolute error (a scalar summary)
%                Var_lpd_cv  = variance of mean log predictive density
%                              (a scalar summary)
%                Var_rmse_cv = variance of the root mean squared error 
%                              (a scalar summary)
%                Var_abs_cv  = variance of the mean absolute error 
%                              (a scalar summary)
%                trindex     = training indices
%                tstindex    = test indices
%                lpd_cvtr    = mean log predictive density for each of 
%                              k-CV training sets (kx1 vector)
%                rmse_cvtr   = root mean squared error for each of 
%                              k-CV training sets (kx1 vector)
%                abs_cvtr    = absolute error for each of 
%                              k-CV training sets (kx1 vector)
%                lpd_tr      = log predictive density for the 
%                              full training set
%                rmse_tr     = root mean squared error for the 
%                              full training set
%                abs_tr      = absolute error for the full trainng set
%                lpd_ccv     = log predictive density with corrected k-CV
%                rmse_ccv    = root mean squared error with corrected k-CV
%                abs_ccv     = absolute error with corrected k-CV
%                cpu_time    = The cpu time used for inferring the full 
%                              data set
%    
%  
%	See also
%	     demo_modelassesment1, gp_peff, gp_dic
%   
%       References: 
%         Spiegelhalter, Best, Carlin and van der Linde (2002). 
%         Bayesian measures of model complexity and fit. J. R. 
%         Statist. Soc. B, 64, 583-639.
%         
%         Gelman, Carlin, Stern and Rubin (2004) Bayesian Data
%         Analysis, second edition. Chapman & Hall / CRC.
%
%         Aki Vehtari and Jouko Lampinen. Bayesian model assessment
%         and comparison using cross-validation predictive
%         densities. Neural Computation, 14(10):2439-2468, 2002.

% Copyright (c) 2009-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    
    ip=inputParser;
    ip.FunctionName = 'GP_KFCV';
    ip.addRequired('gp',@isstruct);
    ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('inf_method', 'MAP_scg2', @(x) ...
                     ismember(x,{'MAP_scg2' 'MAP_fminunc' 'MCMC' 'IA'}))
    ip.addParamValue('opt', struct)
    ip.addParamValue('k', 10, @(x) isreal(x) && isscalar(x) && isfinite(x) && x>0)
    ip.addParamValue('rstream', round(rem(now,1e-3)*1e9), @(x) isreal(x) && isscalar(x) && isfinite(x) && x>0)
    ip.addParamValue('trindex', [], @(x) ~isempty(x) || iscell(x))    
    ip.addParamValue('tstindex', [], @(x) ~isempty(x) || iscell(x))
    ip.addParamValue('save_results', false, @(x) islogical(x))
    ip.addParamValue('folder', [], @(x) ischar(x) )
    ip.parse(gp, x, y, varargin{:});
    z=ip.Results.z;
    inf_method=ip.Results.inf_method;
    opt=ip.Results.opt;
    k=ip.Results.k;
    rstream=ip.Results.rstream;
    trindex=ip.Results.trindex;
    tstindex=ip.Results.tstindex;
    save_results=ip.Results.save_results;
    folder = ip.Results.folder;
    
    
    [n,nin] = size(x);
    
    gp_orig = gp;

    if isempty(opt) && strcmp(inf_method,'MAP_scg2')
       opt = scg2_opt; 
    end
    
    if (isempty(trindex) && ~isempty(tstindex)) || (~isempty(trindex) && isempty(tstindex))
        error('gp_kfcv: If you give cross validation indices, you need to provide both trindex and tstindex.')
    end
    
    if isempty(trindex) || isempty(tstindex)
        [trindex, tstindex] = cvit(n, k, rstream);
    end
    
    parent_folder = pwd;
        
    % Check which energy and gradient function
    if ~isstruct(gp.likelih)   % a Gaussian regression model
        fe=str2fun('gp_e');
        fg=str2fun('gp_g');
        switch inf_method
          case {'MAP_scg2' 'MAP_fminunc'}
            fp=str2fun('gp_pred');
          case 'MCMC'
            fp=str2fun('mc_pred');
          case 'IA'
            fp=str2fun('ia_pred');
        end
    else
        switch inf_method
          case {'MAP_scg2' 'MAP_fminunc'}
            switch gp.latent_method
              case 'Laplace'
                fe=str2fun('gpla_e');
                fg=str2fun('gpla_g');
                fp=str2fun('la_pred');
              case 'EP'
                fe=str2fun('gpep_e');
                fg=str2fun('gpep_g');
                fp=str2fun('ep_pred');
            end
          case 'MCMC'
            fp=str2fun('mc_pred');
          case 'IA'
            fp=str2fun('ia_pred');
        end
    end
    
    cvws=[];
    trw=[];
    
    % loop over the crossvalidation sets
    for i=1:length(trindex)
        
        fprintf('The CV-iteration number: %d \n', i)
        
        % Set the training and test sets for i'th cross-validation set
        xtr = x(trindex{i},:);
        ytr = y(trindex{i},:);
        xtst = x(tstindex{i},:);
        ytst = y(tstindex{i},:);
        
        if ~isempty(z)
            options_tr.z = z(trindex{i},:);
            options_tst.zt = z;
            options_tst.yt = y;
        else
            options_tr = struct();
            options_tst.yt = y;
        end
        
        gp = gp_orig;
        
        switch gp.type
          case {'FULL' 'VAR' 'DTC' 'SOR'}
            tstind = [];
          case {'FIC' 'CS+FIC'}
            tstind = trindex{i};
          case 'PIC'
            % Set the block indices for the cv set of data points
            ntr = size(xtr,1);
            ntst = size(xtst,1);
            size(trindex{i});
            for i1=1:length(gp.tr_index)
                tstind{i1} = [];
                trind{i1} = [];
                for j1 = 1:length(gp.tr_index{i1})
                    indtmp = find( sum((xtr - repmat(x(gp.tr_index{i1}(j1),:),ntr,1)).^2,2) == 0 );
                    if isempty( indtmp )
                        indtmp = find( sum((xtst - repmat(x(gp.tr_index{i1}(j1),:),ntst,1)).^2,2) == 0 );
                        tstind{i1} = [tstind{i1} indtmp];
                    else
                        trind{i1} = [trind{i1} indtmp];
                    end
                end
            end
            gp.tr_index = trind;
        end

        if isstruct(gp.likelih)
            switch gp.latent_method
              case 'Laplace'
                gp = gp_init('set', gp, 'latent_method', {'Laplace', xtr, ytr, options_tr});
              case 'EP'
                gp = gp_init('set', gp, 'latent_method', {'EP', xtr, ytr, options_tr});
              case 'MCMC'
                gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(ytr))', gp_orig.fh_mc});
            end
        end
                        
        % Conduct inference
        switch inf_method
          case 'MAP_scg2'
            w=gp_pak(gp);
            w = scg2(fe, w, opt, fg, gp, xtr, ytr, options_tr);
            gp=gp_unpak(gp,w);
            cvws(i,:)=w;
          case 'MAP_fminunc'
            w=gp_pak(gp);
            mydeal = @(varargin)varargin{1:nargout};
            w = fminunc(@(ww) mydeal(fe(ww, gp, xtr, ytr, options_tr), fg(ww, gp, xtr, ytr, options_tr)), w, opt);
            gp=gp_unpak(gp,w);
            cvws(i,:)=w;
          case 'MCMC'
            gp = gp_mc(gp, xtr, ytr, options_tr, opt);
          case 'IA'
            gp = gp_ia(gp, xtr, ytr, [], options_tr, opt);
        end
            
        % make the prediction
        [Ef, Varf, Ey, Vary, py] = feval(fp, gp, xtr, ytr, x, 'tstind', tstind, options_tr, options_tst);
        if nargout>=6
          cvtrpreds.Ef([trindex{i} tstindex{i}],i)=Ef([trindex{i} tstindex{i}],:);
          cvtrpreds.Varf([trindex{i} tstindex{i}],i)=Varf([trindex{i} tstindex{i}],:);
          cvtrpreds.Ey([trindex{i} tstindex{i}],i)=Ey([trindex{i} tstindex{i}],:);
          cvtrpreds.Vary([trindex{i} tstindex{i}],i)=Vary([trindex{i} tstindex{i}],:);
          cvtrpreds.py([trindex{i} tstindex{i}],i)=py([trindex{i} tstindex{i}],:);
        end
        if nargout>=2
          cvpreds.Ef(tstindex{i},:)=Ef(tstindex{i},:);
          cvpreds.Varf(tstindex{i},:)=Varf(tstindex{i},:);
          cvpreds.Ey(tstindex{i},:)=Ey(tstindex{i},:);
          cvpreds.Vary(tstindex{i},:)=Vary(tstindex{i},:);
          cvpreds.py(tstindex{i},:)=py(tstindex{i},:);
        end
                
        % Evaluate statistics
        lpd_cv(tstindex{i}) = log(mean(py(tstindex{i},:),2));
        lpd_cvtr(i) = mean(log(mean(py(trindex{i}),2)));
            
        rmse_cv(tstindex{i}) = (mean(Ey(tstindex{i},:),2) - ytst).^2;
        rmse_cvtr(i) = sqrt(mean((mean(Ey(trindex{i},:),2) - ytr).^2));
        
        abs_cv(tstindex{i}) = abs(mean(Ey(tstindex{i},:),2) - ytst);
        abs_cvtr(i) = mean(abs(mean(Ey(trindex{i},:),2) - ytr));
        
        lpd_cvm(i) = mean(log(mean(py(tstindex{i},:),2)));
        rmse_cvm(i) = sqrt(mean((mean(Ey(tstindex{i},:),2) - ytst).^2));
        abs_cvm(i) = mean(abs(mean(Ey(tstindex{i},:),2) - ytst));
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
        switch gp.type
          case {'FULL' 'VAR' 'DTC' 'SOR'}
            tstind = [];
          case 'FIC'
            tstind = 1:n;
          case 'PIC'
            tstind = gp.tr_index;
        end
        
        if ~isempty(z)
            options_tr.z = z;
            options_tst.zt = z;
            options_tst.yt = y;
        else
            options_tr = struct();
            options_tst.yt = y;
        end
        
        % Evaluate the training utility
        fprintf('\n Evaluating the training utility \n')
       
        
        % Conduct inference
        cpu_time = cputime;
        switch inf_method
          case 'MAP_scg2'
            w=gp_pak(gp);
            w = scg2(fe, w, opt, fg, gp, x, y, options_tr);
            gp=gp_unpak(gp,w);
            trw=w;
          case 'MAP_fminunc'
            w=gp_pak(gp);
            mydeal = @(varargin)varargin{1:nargout};
            w = fminunc(@(ww) mydeal(fe(ww, gp, x, y, options_tr), fg(ww, gp, x, y, options_tr)), w, opt);
            gp=gp_unpak(gp,w);
            trw=w;
          case 'MCMC'
            gp = gp_mc(gp, x, y, options_tr, opt);
          case 'IA'
            gp = gp_ia(gp, x, y, [], options_tr, opt);
        end
        cpu_time = cputime - cpu_time;
        
        % make the prediction
        [Ef, Varf, Ey, Vary, py] = feval(fp, gp, x, y, x, 'tstind', tstind, options_tr, options_tst);
        if nargout>=4
          trpreds.Ef=Ef;
          trpreds.Varf=Varf;
          trpreds.Ey=Ey;
          trpreds.Vary=Vary;
          trpreds.py=py;
        end
        
        lpd_tr = mean(log(mean(py,2)));
        rmse_tr = sqrt(mean((mean(Ey,2) - y).^2));
        abs_tr = mean(abs(mean(Ey,2) - y));
        
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
             'abs_cvtr', 'lpd_tr', 'rmse_tr', 'abs_tr', 'mlpd_ccv', 'mrmse_ccv', 'mabs_ccv', 'cpu_time');
        
        fprintf('The results have been saved in the folder:\n %s/%s \n', parent_folder, folder);
        
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
        fclose(f);
    end          
end
