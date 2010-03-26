function [mlpd_cv, Var_lpd_cv, mrmse_cv, Var_rmse_cv, mabs_cv, Var_abs_cv] = gp_kfcv(gp, x, y, inf_method, opt, param, SAVE, trindex, tstindex, folder)
%GP_KFCV        K-fold cross validation for GP model
%
%	Description
%	[mlpd_cv, Var_lpd_cv, mrmse_cv, Var_rmse_cv, mabs_cv, Var_abs_cv] = 
%                gp_kfcv(gp, x, y, inf_method, opt, trindex, tstindex, param, SAVE, folder)
%
%       Perform K-fold cross validation for GP model.
%
%       The input arguments are the following:
%         gp           - GP data structure containing the model
%         x            - inputs
%         y            - outputs
%         inf_method   - inference method. Possible methods are
%                         'MAP_scg2'     hyperparameter optimization with SCG
%                         'MAP_fminunc'  hyperparameter optimization with fminunc
%                         'MCMC'         MCMC sampling using gp_mc
%                         'IA'           integration approximation using gp_ia
%         opt           - options for the inference method
%         trindex       - k-fold CV training indices. A cell array with k fields each 
%                         containing index vector for respective training set. 
%                         Optional, if not given gp_kfcv constructs 10-fold CV indices.
%         tstindex      - k-fold CV test indices. A cell array with k fields each 
%                         containing index vector for respective test set. 
%                         Optional, if not given gp_kfcv constructs 10-fold CV indices.
%         param         - String defining the inferred parameters (see e.g. gp_pak). 
%                         Optional. Default value is 'covariance'.
%         SAVE          - defines if results are stored (1) or not (0). By default 1.
%                         If SAVE==1 gp_kfcv stores the results in the current working 
%                         directory (or in 'folder', see next option) into a cv_resultsX
%                         folder, where X is a number. If there is already cv_results* 
%                         folders X is the smallest number not in use yet.
%         folder        - defines the folder where to save the results. By defauls the 
%                         current working directory.
%
%       The output arguments are the following
%         mlpd_cv          - mean log predictive density
%         Var_lpd_cv       - variance estimate for mlpd
%         rmse_cv          - root mean squared error
%         Var_rmse_cv      - variance estimate for mrmse
%         mabs_cv          - mean absolute error
%         Var_abs_cv       - variance estimate for mabs
%
%
%       The K-fold cross validation is performed as follows. The data is divided into 
%       k groups D_k. For each group we evaluate the test statistics 
%
%            u(D_k | D_{k-1})
% 
%       where u is the utility function and D_{k-1} is the data in the k-1 groups other 
%       than k. The utility functions provided by gp_kfcv are
%
%            lpd(D_k | D_{k-1})  = mean( log( p( y_k|D_{k-1} ) ) )       (log predictive density)
%            rmse(D_k | D_{k-1}) = mean( ( E[y_k|D_{k-1}] - y_k ).^2 )   (squared error)
%            abs(D_k | D_{k-1})  = mean( abs( E[y_k|D_{k-1}] - y_k ) )   (absolute error)
%
%       After the utility is evaluated for each group we can evaluate the output arguments 
%       which are obtained as follows
%    
%            mlpd_cv  = mean( lpd(D_k | D_{k-1}) )          ,k=1...K  (mean log predictive density)
%            mrmse_cv = sqrt( mean( rmse(D_k | D_{k-1}) ) ) ,k=1...K  (root mean squared error)
%            mabs_cv  = mean( abs(D_k | D_{k-1}) )          ,k=1...K  (mean absolute error)
%    
%       The variance estimates for the above statistics are evaluated across the 
%       groups K. For mean log predictive density and mean absolute error this reduces 
%       to evaluate, for example, 
%
%            Var_lpd_cv = var( lpd(D_k | D_{k-1}) ) / K,    k=1...K.
%
%       For root mean squared error we need to take first the square root of 
%       each group statistics to obtain
%
%            Var_rmse_cv = var( sqrt( rmse(D_k | D_{k-1}) ) ) / K,    k=1...K.
%   
%       The above statistics are returned by the funtion. However, if we use the save option 
%       we obtain little more test statistics which are only saved in the result file. These 
%       extra statistics include, for example, bias corrected expected utilities
%       (Vehtari and Lampinen. 2002), and the training utility for whole data and each cross
%       validation training set. The detailed list of variables saved in the result file is:
%
%
%                For more information see the file cv_results.mat, which contains
%                the following variables
%                lpd_cv      = log predictive density (nx1 vector)
%                rmse_cv     = squared error (nx1 vector)
%                abs_cv      = absolute error (nx1 vector)
%                mlpd_cv     = mean log predictive density (a scalar summary)
%                mrmse_cv    = root mean squared error (a scalar summary)
%                mabs_cv     = mean absolute error (a scalar summary)
%                Var_lpd_cv  = variance of mean log predictive density (a scalar summary)
%                Var_rmse_cv = variance of the root mean squared error (a scalar summary)
%                Var_abs_cv  = variance of the mean absolute error (a scalar summary)
%                trindex     = training indeces
%                tstindex    = test indeces
%                lpd_cvtr    = mean log predictive density for each of k cv trainng sets (kx1 vector)
%                rmse_cvtr   = root mean squared error for each of k cv trainng sets (kx1 vector)
%                abs_cvtr    = absolute error for each of k cv training sets (kx1 vector)
%                lpd_tr      = log predictive density for the full trainng set
%                rmse_tr     = root mean squared error for the full trainng set
%                abs_tr      = absolute error for the full trainng set
%                lpd_ccv     = log predictive density with corrected cross validation
%                rmse_ccv    = root mean squared error with corrected cross validation
%                abs_ccv     = absolute error with corrected cross validation
%                cpu_time    = The cpu time used for inferring the full data set
%    
%  
%	See also
%	     demo_modelassesment1, gp_peff, gp_dic
%   
%       References: 
%         Spiegelhalter, Best, Carlin and van der Linde (2002). Bayesian measures
%         of model complexity and fit. J. R. Statist. Soc. B, 64, 583-639.
%         
%         Gelman, Carlin, Stern and Rubin (2004) Bayesian Data Analysis, second 
%         edition. Chapman & Hall / CRC.
%
%         Aki Vehtari and Jouko Lampinen. Bayesian model assessment and comparison 
%         using crossvalidation predictive densities. Neural Computation, 
%         14(10):2439-2468, 2002.

% Copyright (c) 2009-2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

    
    
    [n,nin] = size(x);
    
    gp_orig = gp;
    
    if nargin < 5
        error('gp_kfcv: Not enough arguments.')
    end
        
    if nargin < 6 || isempty(param)
        param = 'covariance';
    end

    if nargin < 7 || isempty(SAVE)
        SAVE = 1;
    end
    
    if nargin < 8 || isempty(trindex)
        [trindex, tstindex] = cvitr(n, 10);
    end
    
    if nargin < 10
        folder = [];
    end
    
    % Check which energy and gradient function
    if ~isstruct(gp.likelih)   % a regression model
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
    
    % loop over the crossvalidation sets
    for i=1:length(trindex)
        
        fprintf('The CV-iteration number: %d \n', i)
        
        % Set the training and test sets for i'th cross-validation set
        xtr = x(trindex{i},:);
        ytr = y(trindex{i},:);
        xtst = x(tstindex{i},:);
        ytst = y(tstindex{i},:);
        gp = gp_orig;
        
        switch gp.type
          case 'FULL'
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
                gp = gp_init('set', gp, 'latent_method', {'Laplace', xtr, ytr, param});
              case 'EP'
                gp = gp_init('set', gp, 'latent_method', {'EP', xtr, ytr, param});
              case 'MCMC'
                gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(ytr))', gp_orig.fh_mc});
            end
        end
                        
        % Conduct inference
        switch inf_method
          case 'MAP_scg2'
            w=gp_pak(gp, param);
            w = scg2(fe, w, opt, fg, gp, xtr, ytr, param);
            gp=gp_unpak(gp,w, param);
          case 'MAP_fminunc'
            w=gp_pak(gp, param);
            mydeal = @(varargin)varargin{1:nargout};
            w = fminunc(@(ww) mydeal(fe(ww, gp, xtr, ytr, param), fg(ww, gp, xtr, ytr, param)), w0, opt);
            gp=gp_unpak(gp,w, param);
          case 'MCMC'
            gp = gp_mc(opt, gp, xtr, ytr);
          case 'IA'
            gp = gp_ia(opt, gp, xtr, ytr, [], param);
        end
            
        % make the prediction
        [Ef, Varf, Ey, Vary, py] = feval(fp, gp, xtr, ytr, x, param, [], tstind, y);
                
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
    
    % save results
    if SAVE == 1
        
        switch gp.type
          case 'FULL'
            tstind = [];
          case 'FIC'
            tstind = 1:n;
          case 'PIC'
            tstind = gp.tr_index;
        end
        
        % Evaluate the training utility
        fprintf('\n Evaluating the training utility \n')
        gp = gp_orig;
        
        % Conduct inference
        cpu_time = cputime;
        switch inf_method
          case 'MAP_scg2'
            w=gp_pak(gp, param);
            w = scg2(fe, w, opt, fg, gp, x, y, param);
            gp=gp_unpak(gp,w, param);
          case 'MAP_fminunc'
            w=gp_pak(gp, param);
            mydeal = @(varargin)varargin{1:nargout};
            w = fminunc(@(ww) mydeal(fe(ww, gp, x, y, param), fg(ww, gp, x, y, param)), w0, opt);
            gp=gp_unpak(gp,w, param);
          case 'MCMC'
            gp = gp_mc(opt, gp, xtr, ytr);
        end
        cpu_time = cputime - cpu_time;
        
        % make the prediction
        [Ef, Varf, Ey, Vary, py] = feval(fp, gp, x, y, x, param, [], tstind, y);
                
        lpd_tr(i) = mean(log(mean(py,2)));
        rmse_tr(i) = sqrt(mean((mean(Ey,2) - y).^2));
        abs_tr(i) = mean(abs(mean(Ey,2) - y));
        
        lpd_ccv =  mlpd_cv +  mean(lpd_tr) -  mean(lpd_cvtr);
        rmse_ccv =  mrmse_cv +  mean(rmse_tr) -  mean(rmse_cvtr);
        abs_ccv =  mabs_cv +  mean(abs_tr) -  mean(abs_cvtr);
        
        % Save the results
        parent_folder = pwd;
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
        end
        mkdir(parent_folder, folder);
        
        save([parent_folder '/' folder '/cv_results.mat'], 'lpd_cv', 'rmse_cv', 'abs_cv', 'mlpd_cv', 'mrmse_cv',...
             'mabs_cv','Var_lpd_cv', 'Var_rmse_cv', 'Var_abs_cv', 'trindex', 'tstindex', 'lpd_cvtr', 'rmse_cvtr',...
             'abs_cvtr', 'lpd_tr', 'rmse_tr', 'abs_tr', 'lpd_ccv', 'rmse_ccv', 'abs_ccv', 'cpu_time');
        
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
        fprintf(f,'trindex     = training indeces \n'); 
        fprintf(f,'tstindex    = test indeces \n'); 
        fprintf(f,'lpd_cvtr    = mean log predictive density for each of k cv trainng sets (kx1 vector) \n'); 
        fprintf(f,'rmse_cvtr   = root mean squared error for each of k cv trainng sets (kx1 vector) \n'); 
        fprintf(f,'abs_cvtr    = absolute error for each of k cv training sets (kx1 vector) \n'); 
        fprintf(f,'lpd_tr      = log predictive density for the full trainng set  \n'); 
        fprintf(f,'rmse_tr     = root mean squared error for the full trainng set  \n'); 
        fprintf(f,'abs_tr      = absolute error for the full trainng set  \n'); 
        fprintf(f,'lpd_ccv     = log predictive density with corrected cross validation   \n'); 
        fprintf(f,'rmse_ccv    = root mean squared error with corrected cross validation \n');
        fprintf(f,'abs_ccv     = absolute error with corrected cross validation \n');
        fprintf(f,'cpu_time    = The cpu time used for inferring the full data set \n');
        fclose(f);
    end          
end

