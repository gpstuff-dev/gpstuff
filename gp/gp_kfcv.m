function [mlpd_cv, Var_lpd_cv, mrmse_cv, Var_rmse_cv, mabs_cv, Var_abs_cv] = gp_kfcv(gp, x, y, inf_method, opt, trindex, tstindex, param, SAVE, folder)

    [n,nin] = size(x);
    
    gp_orig = gp;
        
    if nargin < 6
        [trindex, tstindex] = cvitr(n, 10);
    end
    
    if nargin < 8
        param = 'hyper';
    end
   
    if nargin < 9
        SAVE = 1;
    end

    if nargin < 10
        folder = [];
    end

    switch inf_method
      case {'MAP_scg2' 'MAP_fminunc'}
    
        % Check which energy and gradient function
        if ~isstruct(gp.likelih)   % a regression model
            fe=str2fun('gpla_e');
            fg=str2fun('gpla_g');
        else
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
        end
    
        % loop over the crossvalidation sets
        for i=1:length(trindex)
            
            fprintf('The CV-iteration number: %d \n', i)
            
            % Set the training and test sets for i'th cross-validation set
            xtr = x(trindex{i},:);
            ytr = y(trindex{i},:);
            xtst = x(tstindex{i},:);
            ytst = y(tstindex{i},:);
            
            if isstruct(gp.likelih)
                switch gp.latent_method
                  case 'Laplace'
                    gp = gp_init('set', gp_orig, 'latent_method', {'Laplace', xtr, ytr, param});
                  case 'EP'
                    gp = gp_init('set', gp_orig, 'latent_method', {'EP', xtr, ytr, param});
                end
            end
            
            switch gp.type
              case 'FULL'
                tstind = [];
              case 'FIC'
                tstind = trindex{i};
              case 'PIC'
                tstind = gp.tr_index;
            end
            
            % Conduct inference
            w=gp_pak(gp, param);
            switch inf_method
              case 'MAP_scg2'
                w = scg2(fe, w, opt, fg, gp, xtr, ytr, param);
              case 'MAP_fminunc'
                mydeal = @(varargin)varargin{1:nargout};
                w = fminunc(@(ww) mydeal(fe(ww, gp, xtr, ytr, param), fg(ww, gp, xtr, ytr, param)), w0, opt);
            end
            gp=gp_unpak(gp,w, param);
            
            % make the prediction
            [Ef, Varf, Ey, Vary, py] = feval(fp, gp, xtr, ytr, x, param, [], tstind, y);
            
            % Evaluate statistics
            lpd_cv(tstindex{i}) = log(py(tstindex{i}));
            lpd_cvtr(i) = mean(log(py(trindex{i})));
            
            rmse_cv(tstindex{i}) = (Ey(tstindex{i}) - ytst).^2;
            rmse_cvtr(i) = sqrt(mean((Ey(trindex{i}) - ytr).^2));
            
            abs_cv(tstindex{i}) = abs(Ey(tstindex{i}) - ytst);
            abs_cvtr(i) = mean(abs(Ey(trindex{i}) - ytr));
            
            lpd_cvm(i) = mean(log(py(tstindex{i})));
            rmse_cvm(i) = sqrt(mean((Ey(tstindex{i}) - ytst).^2));
            abs_cvm(i) = mean(abs(Ey(tstindex{i}) - ytst));
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
            w=gp_pak(gp, param);
            cpu_time = cputime;
            switch inf_method
              case 'MAP_scg2'
                w = scg2(fe, w, opt, fg, gp, x, y, param);
              case 'MAP_fminunc'
                mydeal = @(varargin)varargin{1:nargout};
                w = fminunc(@(ww) mydeal(fe(ww, gp, x, y, param), fg(ww, gp, x, y, param)), w0, opt);
            end
            cpu_time = cputime - cpu_time;
            gp=gp_unpak(gp,w, param);

            % make the prediction
            [Ef, Varf, Ey, Vary, py] = feval(fp, gp, x, y, x, param, [], tstind, y);

            lpd_tr(i) = mean(log(py));
            rmse_tr(i) = sqrt(mean((Ey - y).^2));
            abs_tr(i) = mean(abs(Ey - y));
                        
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
            
            fprintf('The results have been saved in the folder:\n, %s \n', [parent_folder folder]);
            
            f = fopen([folder '/description.txt'],'w');
            fprintf(f,'The cv results were the following: \n\n');
            fprintf(f,'mlpd_cv = %.4f (+/- %.4f) \n', mlpd_cv, Var_lpd_cv);
            fprintf(f,'mrmse_cv = %.4f (+/- %.4f)  \n', mrmse_cv, Var_rmse_cv);
            fprintf(f,'mabs_cv = %.4f (+/- %.4f) \n', mabs_cv, Var_abs_cv);
            fprintf(f,'cpu time = %.4f \n', cpu_time);
            fprintf(f,'\n\n');
            
            fprintf(f,'For more information see the file cv_results.mat, which contains');            
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
            fprintf(f,'abs_cvtr    = absolute error for each of k cv trainng sets (kx1 vector) \n'); 
            fprintf(f,'lpd_tr      = log predictive density for the full trainng set  \n'); 
            fprintf(f,'rmse_tr     = root mean squared error for the full trainng set  \n'); 
            fprintf(f,'abs_tr      = absolute error for the full trainng set  \n'); 
            fprintf(f,'lpd_ccv     = log predictive density with corrected cross validation   \n'); 
            fprintf(f,'rmse_ccv    = root mean squared error with corrected cross validation \n');
            fprintf(f,'abs_ccv     = absolute error with corrected cross validation \n');
            fprintf(f,'cpu_time    = The cpu time used for inferring the full data set \n');
            fclose(f);
        end          
        
        
      case 'MCMC'
        
      case 'IA'
            
    end
    
end
