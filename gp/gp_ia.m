function [gp_array, P_TH, th, Ef, Varf, x, fx, H] = gp_ia(opt, gp, xx, yy, tx, param, tstindex)
% GP_IA explores the hypeparameters around the mode and returns a
% list of GPs with different hyperparameters and corresponding weights
%
%       Description [GP_ARRAY, P_TH, EF, VARF, X, FX] = GP_IA(OPT,
%       GP, XX, YY, TX, PARAM, TSTINDEX) takes a gp data structure GP
%       with covariates XX and observations YY and returns an array of
%       GPs GP_ARRAY and corresponding weights P_TH. If test
%       covariates TX is included, GP_IA also returns corresponding
%       mean EF and variance VARF (FX is PDF evaluated at X). TSTINDEX
%       defines the test index for FIC, PIC and CS+FIC (see e.g. gp_pred).
%
%       Options structure OPT contains the integration and optimization options.
%         OPT.SCG        contains the options for scaled conjugate gradients
%         OPT.FMINUNC    contains the options for fminunc
%         OPT.INT_METHOD is the method used for integration
%                        'CCD' for circular composite design
%                        'grid' for grid search
%                        'is_normal' for sampling from gaussian appr
%                        'is_normal_qmc' for quasi monte carlo samples
%
%         see gp_iaopt for more detailed desription for options.

% Copyright (c) 2009-2010 Ville Pietiläinen, Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.


% ===========================
% Which latent method is used
% ===========================
    if isfield(gp, 'latent_method')
        switch gp.latent_method
          case 'EP'
            fh_e = @gpep_e;
            fh_g = @gpep_g;
            fh_p = @ep_pred;
          case 'Laplace'
            fh_e = @gpla_e;
            fh_g = @gpla_g;
            fh_p = @la_pred;
        end
    else 
        fh_e = @gp_e;
        fh_g = @gp_g;
        fh_p = @gp_pred;
        
    end

    if nargin < 6 | isempty(param)
        param = 'covariance';
    end
    
    if nargin < 7
        tstindex = [];
    end
    
    if ~isfield(opt, 'fminunc') && ~isfield(opt, 'scg')
        opt.scg = scg2_opt;
        opt.scg.tolfun = 1e-3;
        opt.scg.tolx = 1e-3;
        opt.scg.display = 1;
    end

    if ~isfield(opt,'int_method')
        opt.int_method = 'grid';
    end

    if ~isfield(opt,'threshold')
        opt.threshold = 2.5;
    end

    if ~isfield(opt,'step_size')
        opt.step_size = 1;
    end

    if ~isfield(opt,'rotation')
        opt.rotation = true;
    end

    if ~isfield(opt,'improved')
        opt.improved = '0';
    end

    if ~isfield(opt,'isscale')
        opt.isscale = 1;
    end

    if ~isfield(opt,'noisy')
        opt.noisy = 0;
    end
    
    if ~isfield(opt,'validate')
        opt.validate = 0;
    end
    
    if ~isfield(opt,'qmc')
        opt.qmc = 0;
    end
    
    % ====================================
    % Find the mode of the hyperparameters
    % ====================================

    w0 = gp_pak(gp, param);
% $$$ gradcheck(w0, fh_e, fh_g, gp, xx, yy, param)
    mydeal = @(varargin)varargin{1:nargout};

    % The mode and hessian at it 
    if isfield(opt, 'scg')
        w = scg2(fh_e, w0, opt.scg, fh_g, gp, xx, yy, param);
    elseif isfield(opt, 'fminunc')
        w = fminunc(@(ww) mydeal(feval(fh_e,ww, gp, xx, yy, param), feval(fh_g, ww, gp, xx, yy, param)), w0, opt.fminunc);
    end
    gp = gp_unpak(gp,w,param);

    % Number of parameters
    nParam = length(w);

    switch opt.int_method
      case {'grid', 'CCD'}
        
        % ===============================
        % New variable z for exploration
        % ===============================

        H = hessian(w);    
        Sigma = inv(H);
        
        % Some jitter may be needed to get positive semi-definite covariance
        if any(eig(Sigma)<0)
            jitter = 0;
            while any(eig(Sigma)<0)
                jitter = jitter + eye(size(H,1))*0.01;
                Sigma = Sigma + jitter;
            end
            warning('gp_ia -> singular Hessian. Jitter of %.4f added.', jitter)
        end

        if ~opt.rotation
            Sigma=diag(diag(Sigma));
        end
        
        [V,D] = eig(full(Sigma));
        z = (V*sqrt(D))'.*opt.step_size;

        % =======================================
        % Exploration of possible hyperparameters
        % =======================================
        
        checked = zeros(1,nParam); % List of locations already visited
        candidates = zeros(1,nParam); % List of locations with enough density
        gp_array={}; % Array of gp-models with different hyperparameters
        Ef_grid = []; % Predicted means with different hyperparameters
        Varf_grid = []; % Variance of predictions with different hyperparameters
        p_th=[]; % List of the weights of different hyperparameters (un-normalized)
        th=[]; % List of hyperparameters
        
        switch opt.int_method
          case 'grid'
            % Make the predictions in the mode if needed and estimate the density of the mode
            if exist('tx')
                if isfield(gp,'latent_method')
                    p_th(1) = -feval(fh_e,w,gp,xx,yy,param);
                    [Ef_grid(1,:), Varf_grid(end+1,:)]=feval(fh_p,gp,xx,yy,tx,param,[],tstindex);
                else
                    p_th(1) = -feval(fh_e,w,gp,xx,yy);
                    [Ef_grid(1,:), Varf_grid(end+1,:)]=feval(fh_p,gp,xx,yy,tx,[],tstindex);
                end
            else
                p_th(1) = -feval(fh_e,w,gp,xx,yy,param);
            end
            
            % Put the mode to th-array and gp-model in the mode to gp_array
            th(1,:) = w;
            gp = gp_unpak(gp,w,param);
            gp_array{end+1} = gp;

            
            while ~isempty(candidates) % Repeat until there are no hyperparameters
                                       % with enough density that are not
                                       % checked yet
                for i1 = 1 : nParam % Loop through the dimensions
                    pos = zeros(1,nParam); pos(i1)=1; % One step to the positive direction
                                                      % of dimension i1
                    
                    % Check if the neighbour in the direction of pos is already checked
                    if ~any(sum(abs(repmat(candidates(1,:)+pos,size(checked,1),1)-checked),2)==0) 
                        w_p = w + candidates(1,:)*z + z(i1,:); % The parameters in the neighbour
                                                               %p_th(end+1) = -feval(fh_e,w_p,gp,xx,yy,param);
                        th(end+1,:) = w_p; 
                        
                        gp = gp_unpak(gp,w_p,param);
                        gp_array{end+1} = gp;
                        
                        if exist('tx')
                            if isfield(gp,'latent_method')
                                p_th(end+1) = -feval(fh_e,w_p,gp,xx,yy,param);
                                [Ef_grid(end+1,:), Varf_grid(end+1,:)]=feval(fh_p,gp,xx,yy,tx,param,[],tstindex);
                            else
                                p_th(end+1) = -feval(fh_e,w_p,gp,xx,yy);
                                [Ef_grid(end+1,:), Varf_grid(end+1,:)]=feval(fh_p,gp,xx,yy,tx,[],tstindex);
                            end
                        else
                            p_th(end+1) = -feval(fh_e,w_p,gp,xx,yy,param);
                        end
                        
                        % If the density is enough, put the location in to the
                        % candidates list. The neighbours of that location
                        % will be studied lated
                        if (p_th(1)-p_th(end))<opt.threshold
                            candidates(end+1,:) = candidates(1,:)+pos;
                        end

                        % Put the recently studied point to the checked list
                        checked(end+1,:) = candidates(1,:)+pos;    
                    end
                    
                    neg = zeros(1,nParam); neg(i1)=-1;
                    if ~any(sum(abs(repmat(candidates(1,:)+neg,size(checked,1),1)-checked),2)==0)
                        w_n = w + candidates(1,:)*z - z(i1,:);
                        %p_th(end+1) = -feval(fh_e,w_n,gp,xx,yy,param);
                        th(end+1,:) = w_n;
                        
                        gp = gp_unpak(gp,w_n,param);
                        gp_array{end+1} = gp;
                        
                        if exist('tx')
                            if isfield(gp,'latent_method')
                                p_th(end+1) = -feval(fh_e,w_n,gp,xx,yy,param);
                                [Ef_grid(end+1,:), Varf_grid(end+1,:)]=feval(fh_p,gp,xx,yy,tx,param,[],tstindex);
                            else
                                p_th(end+1) = -feval(fh_e,w_n,gp,xx,yy);
                                [Ef_grid(end+1,:), Varf_grid(end+1,:)]=feval(fh_p,gp,xx,yy,tx,[],tstindex);
                            end
                        else
                            p_th(end+1) = -feval(fh_e,w_n,gp,xx,yy,param);
                        end
                        
                        if (p_th(1)-p_th(end))<opt.threshold
                            candidates(end+1,:) = candidates(1,:)+neg;
                        end
                        checked(end+1,:) = candidates(1,:)+neg;
                    end
                end
                candidates(1,:)=[];
            end    
            
            % Convert densities from the log-space and normalize them
            p_th = p_th(:)-min(p_th);
            P_TH = exp(p_th)/sum(exp(p_th));

          case 'CCD'
            % Walsh indeces (see Sanchez and Sanchez (2005))
            walsh = [1 2 4 8 15 16 32 51 64 85 106 128 150 171 219 237 ...
                     247 256 279 297 455 512 537 557 597 643 803 863 898 ...
                     1024 1051 1070 1112 1169 1333 1345 1620 1866 2048 ...
                     2076 2085 2158 2372 2456 2618 2800 2873 3127 3284 ...
                     3483 3557 3763 4096 4125 4135 4176 4435 4459 4469 ...
                     4497 4752 5255 5732 5801 5915 6100 6369 6907 7069 ...
                     8192 8263 8351 8422 8458 8571 8750 8858 9124 9314 ...
                     9500 10026 10455 10556 11778 11885 11984 13548 14007 ...
                     14514 14965 15125 15554 16384 16457 16517 16609 ...
                     16771 16853 17022 17453 17891 18073 18562 18980 ...
                     19030 19932 20075 20745 21544 22633 23200 24167 ...
                     25700 26360 26591 26776 28443 28905 29577 32705];
            
            % ERROR CHECK
            
            % How many design points
            ii = sum(nParam >= [1 2 3 4 6 7 9 12 18 22 30 39 53 70 93]);
            H0 = 1;
            
            % Design matrix
            for i1 = 1 : ii
                H0 = [H0 H0 ; H0 -H0];
            end
            
            % Radius of the sphere (greater than 1)
            if ~(isfield(opt, 'f0'))
                f0=1.3;
            else
                f0 = opt.f0;
            end
            
            % Design points
            points = H0(:,1+walsh(1:nParam));
            % Center point
            points = [zeros(1,nParam); points];
            % Points on the main axis
            for i1 = 1 : nParam
                points(end+1,:)=zeros(1,nParam);
                points(end,i1)=sqrt(nParam);
                points(end+1,:)=zeros(1,nParam);
                points(end,i1)=-sqrt(nParam);
            end
            design = points;
            
            switch opt.improved 
              case 0
                points = f0*points;
              case 1
                optim = optimset('Display','off');
                
% $$$             sigma = zeros(size(points));
                for j = 1 : nParam*2
                    % Here temp is one of the points on the main axis in either
                    % positive or negative direction.
                    temp = zeros(1,nParam);
                    if mod(j,2) == 1
                        dir = 1;
                    else dir = -1;
                    end
                    ind = ceil(j/2);
                    temp(ind)=dir;

                    % Find the scaling parameter so that when we move 2 stds from the
                    % mode, the log desity drops by 2
                    if isfield(opt, 'scg')
                        error('The improved CCD works only when the optimization method is fminunc.')
                    elseif isfield(opt, 'fminunc')
                        t = fminunc(@(x) abs(-feval(fh_e,x*temp*z+w,gp,xx,yy,param)+feval(fh_e,w,gp,xx,yy,param)+2), 1.3,optim);
                    end
                    sd(points(:,ind)*dir>0, ind) = 0.5*t/sqrt(nParam);
                end
                % Each points is scaled with corresponding scaling parameter and
                % desired radius
                points = f0*sd.*design;
            end

            % Put the points into hyperparameter-space
            th = points*z+repmat(w,size(points,1),1);

            p_th=[]; gp_array={};
            for i1 = 1 : size(th,1)
                gp = gp_unpak(gp,th(i1,:),param);
                gp_array{end+1} = gp;
                % Make predictions if needed

                if exist('tx')
                    if isfield(gp,'latent_method')
                        p_th(end+1) = -feval(fh_e,th(i1,:),gp,xx,yy,param);
                        if ~isempty(tstindex)
                            [Ef_grid(end+1,:), Varf_grid(end+1,:)]=feval(fh_p,gp,xx,yy,tx,param,[],tstindex);
                        else
                            [Ef_grid(end+1,:), Varf_grid(end+1,:)]=feval(fh_p,gp,xx,yy,tx,param,[]);
                        end
                    else
                        p_th(end+1) = -feval(fh_e,th(i1,:),gp,xx,yy,param);
                        if ~isempty(tstindex)
                            [Ef_grid(end+1,:), Varf_grid(end+1,:)]=feval(fh_p,gp,xx,yy,tx,[],tstindex);
                        else
                            [Ef_grid(end+1,:), Varf_grid(end+1,:)]=feval(fh_p,gp,xx,yy,tx,[]);
                        end
                    end
                else
                    p_th(end+1) = -feval(fh_e,th(i1,:),gp,xx,yy,param);
                end
            end
            
            p_th=p_th-min(p_th);
            p_th=exp(p_th);
            p_th=p_th/sum(p_th);
            
            
            % Calculate the area weights for the integration and scale densities
            % of the design points with these weights
            delta_k = 1/((2*pi)^(-nParam/2)*exp(-.5*nParam*f0^2)*(size(points,1)-1)*f0^2);
            delta_0 = (2*pi)^(nParam/2)*(1-1/f0^2);
            
            delta_k=delta_k/delta_0;
            delta_0=1;
            
            p_th=p_th.*[delta_0,repmat(delta_k,1,size(th,1)-1)];
            P_TH=p_th/sum(p_th);
            P_TH=P_TH(:);
        end
        
      case {'is_normal' 'is_normal_qmc' 'is_student-t'}
        
        % Covariance of the gaussian approximation
        H = full(hessian(w));
        Sigma = inv(H);
        Scale = Sigma;
        [V,D] = eig(full(Sigma));
        z = (V*sqrt(D))'.*opt.step_size;
        P0 =  -feval(fh_e,w,gp,xx,yy,param);
        
        % Some jitter may be needed to get positive semi-definite covariance
        if any(eig(Sigma)<0)
            jitter = 0;
            while any(eig(Sigma)<0)
                jitter = jitter + eye(size(H,1))*0.01;
                Sigma = Sigma + jitter;
            end
            warning('gp_ia -> singular Hessian. Jitter of %.4f added.', jitter)
        end
        
        % Number of samples
        if ~isfield(opt, 'nsamples')
            N = 20;
        else
            N = opt.nsamples;
        end
        
        switch opt.int_method
          case 'is_normal' 
            % Normal samples
            
            if opt.qmc
                th  = repmat(w,N,1)+(chol(Sigma)'*(sqrt(2).*erfinv(2.*hammersley(size(Sigma,1),N) - 1)))';
                p_th_appr = mvnpdf(th, w, Sigma);        
            else
                th = mvnrnd(w,Sigma,N);
                p_th_appr = mvnpdf(th, w, Sigma);        
            end
            
            
            if opt.improved
                
                if opt.qmc
                    e = (sqrt(2).*erfinv(2.*hammersley(size(Sigma,1),N) - 1))';
                else
                    e = randn(N,size(Sigma,1));
                end
                
                % Scaling of the covariance (see Geweke, 1989, Bayesian inference in econometric models using monte carlo integration
                delta = -6:.5:6;
                for i0 = 1 : nParam
                    for i1 = 1 : length(delta)
                        ttt = zeros(1,nParam);
                        ttt(i0)=1;
                        phat = (-feval(fh_e,w+(delta(i1)*chol(Sigma)'*ttt')',gp,xx,yy,param));
                        fi(i1) = abs(delta(i1)).*(2.*(P0-phat)).^(-.5);
                        
                        pp(i1) = exp(phat);
                        pt(i1) = mvnpdf(delta(i1)*chol(Sigma)'*ttt', 0, Sigma);
                    end
                    
                    q(i0) = max(fi(delta>0));
                    r(i0) = max(fi(delta<0));
                    
% $$$                     scl = ones(1,length(delta));
% $$$                     scl(1:floor(length(delta)/2))=repmat(r(i0),1,floor(length(delta)/2));
% $$$                     scl(ceil(length(delta)/2):end)=repmat(q(i0),1,ceil(length(delta)/2));
                    
% $$$                     in{i0} = scl.*delta;
% $$$                     out{i0} = pt/max(pt);
% $$$                     in2off = delta;
                    
                end

                %% Samples one by one
                for i3 = 1 : N
                    C = 0;
                    for i2 = 1 : nParam
                        if e(i3,i2)<0
                            eta(i3,i2) = e(i3,i2)*r(i2);
                            C = C + log(r(i2));
                        else
                            eta(i3,i2) = e(i3,i2)*q(i2);
                            C = C + log(q(i2));
                        end
                        
                    end
                    p_th_appr(i3) = exp(-C-.5*e(i3,:)*e(i3,:)');
                    th(i3,:)=w+(chol(Scale)'*eta(i3,:)')';
                end
            end
            
            

          case 'is_student-t'
            % Student-t Samples
            if isfield(opt, 'nu')
                nu = opt.nu;
            else
                nu = 4;
            end
            chi2 = repmat(chi2rnd(nu, [1 N]), nParam, 1);
            Scale = (nu-2)./nu.*Sigma;
            Scale = Sigma;

            if opt.qmc == 1
                e = (sqrt(2).*erfinv(2.*hammersley(size(Sigma,1),N) - 1))';
                th = repmat(w,N,1) + ( chol(Scale)' * e' .* sqrt(nu./chi2) )';                
            else                                    
                th = repmat(w,N,1) + ( chol(Scale)' * randn(nParam, N).*sqrt(nu./chi2) )';
            end

            p_th_appr = mt_pdf(th - repmat(w,N,1), Sigma, nu);
            
            if opt.improved 
                delta = -6:.5:6;
                for i0 = 1 : nParam
                    ttt = zeros(1,nParam);
                    ttt(i0)=1;
                    for i1 = 1 : length(delta)
                        phat = exp(-feval(fh_e,w+(delta(i1)*chol(Scale)'*ttt')',gp,xx,yy,param));
                        
                        fi(i1) = nu^(-.5).*abs(delta(i1)).*(((exp(P0)/phat)^(2/(nu+nParam))-1).^(-.5));
                        rel(i1) = (exp(-feval(fh_e,w+(delta(i1)*chol(Scale)'*ttt')',gp,xx,yy,param)))/ ...
                                  mt_pdf((delta(i1)*chol(Scale)'*ttt')', Scale, nu);
                        pp(i1) = phat;
                        pt(i1) = mt_pdf((delta(i1)*chol(Scale)'*ttt')', Scale, nu);
                    end
                    
                    q(i0) = max(fi(delta>0));
                    r(i0) = max(fi(delta<0));
                    
                    scl = ones(1,length(delta));
                    scl(1:floor(length(delta)/2))=repmat(r(i0),1,floor(length(delta)/2));
                    scl(ceil(length(delta)/2):end)=repmat(q(i0),1,ceil(length(delta)/2));
                end
                
                %% Samples
                if opt.qmc
                    e = (sqrt(2).*erfinv(2.*hammersley(size(Sigma,1),N) - 1))';
                else
                    e = randn(N,size(Sigma,1));
                end
                
                for i3 = 1 : N
                    C = 0;
                    for i2 = 1 : nParam
                        chi(i2) = chi2rnd(nu);
                        if e(i3,i2)<0
                            eta(i3,i2) = e(i3,i2)*r(i2)*(sqrt(nu/chi(i2)));
                            C = C -log(r(i2));
                        else
                            eta(i3,i2) = e(i3,i2)*q(i2)*(sqrt(nu/chi(i2)));
                            C = C  -log(q(i2));
                        end
                    end
                    p_th_appr(i3) = exp(C - ((nu+nParam)/2)*log(1+sum((e(i3,:)./sqrt(chi)).^2)));
                    th(i3,:)=w+(chol(Scale)'*eta(i3,:)')';
                end
            end
        end
        gp_array=cell(N,1);
        
        % Densities of the samples in target distribution and predictions, if needed.
        for j = 1 : N
            gp_array{j}=gp_unpak(gp,th(j,:),param);
            if exist('tx')
                p_th(j) = -feval(fh_e,th(j,:),gp_array{j},xx,yy,param);
                if ~isempty(tstindex)
                    [Ef_grid(j,:), Varf_grid(j,:)]=feval(fh_p,gp_array{j},xx,yy,tx,param,[],tstindex);
                else
                    if isfield(gp, 'latent_method')
                        [Ef_grid(j,:), Varf_grid(j,:)]=feval(fh_p,gp_array{j},xx,yy,tx,param);
                    else 
                        [Ef_grid(j,:), Varf_grid(j,:)]=feval(fh_p,gp_array{j},xx,yy,tx);
                    end
                end
            else
                p_th(j) = -feval(fh_e,th(j,:),gp_array{j},xx,yy,param);
% $$$                 p_th(j) = -feval(fh_e,th(j,:),gp_array{j},xx,yy);
            end
        end
        p_th = exp(p_th-min(p_th));
        p_th = p_th/sum(p_th);

        % (Scaled) Densities of the samples in the approximation of the target distribution
        p_th_appr = p_th_appr/sum(p_th_appr);
                
        % Importance weights for the samples
        iw = p_th(:)./p_th_appr(:);
        iw = iw/sum(iw);
        
        % Return the importance weights
        P_TH = iw;
        
      case {'mcmc_hmc' 'mcmc_sls'}
        
        if isfield(opt, 'hmc_opt')
            if isfield(opt.hmc_opt, 'rstate')
                if ~isempty(opt.hmc_opt.rstate)
                    hmc_rstate = opt.hmc_opt.rstate;
                else
                    hmc2('state', sum(100*clock))
                    hmc_rstate=hmc2('state');
                end
            else
                hmc2('state', sum(100*clock))
                hmc_rstate=hmc2('state');
            end
        end    
        
        ri = 0;
        % -------------- Start sampling ----------------------------
        for j=1:opt.nsamples
            
            if isfield(opt, 'hmc_opt')                        
                if opt.hmc_opt.persistence_reset
                    hmc_rstate.mom = [];
                end
            end
            
            hmcrej = 0;
            for l=1:opt.repeat
                
                % ----------- Sample hyperparameters with HMC --------------------- 
                if isfield(opt, 'hmc_opt')
                    ww = gp_pak(gp, param);
                    hmc2('state',hmc_rstate)              % Set the state
                    [ww, energies, diagnh] = hmc2(fh_e, ww, opt.hmc_opt, fh_g, gp, xx, yy, param);
                    hmc_rstate=hmc2('state');             % Save the current state
                    hmcrej=hmcrej+diagnh.rej/opt.repeat;
                    if isfield(diagnh, 'opt')
                        opt.hmc_opt = diagnh.opt;
                    end
                    opt.hmc_opt.rstate = hmc_rstate;
                    ww=ww(end,:);
                    gp = gp_unpak(gp, ww, param);

                    etr = feval(fh_e,ww,gp,xx,yy,param);
                end
                
% $$$             % ----------- Sample hyperparameters with SLS --------------------- 
% $$$             if isfield(opt, 'sls_opt')
% $$$                 ww = gp_pak(gp, 'hyper');
% $$$                 [ww, energies, diagns] = sls(me, w, opt.sls_opt, mg, gp, xx, yy, 'hyper', varargin{:});
% $$$                 if isfield(diagns, 'opt')
% $$$                     opt.sls_opt = diagns.opt;
% $$$                 end
% $$$                 w=w(end,:);
% $$$                 gp = gp_unpak(gp, w, 'hyper');
% $$$             end
                
            end % ------------- for l=1:opt.repeat -------------------------  

            th(j,:) = ww;
            gp_array{j} = gp_unpak(gp, ww, param);
            
            if exist('tx')
                p_th(j) = 1;
                [Ef_grid(j,:), Varf_grid(j,:)]=feval(fh_p,gp_array{j},xx,yy,tx,[]);
            else
                p_th(j) = 1;
            end
            
            % ----------- Set record -----------------------    
            ri=ri+1;
            
            % Display some statistics  THIS COULD BE DONE NICER ALSO...
            if opt.display
                fprintf(' %4d  %.3f  ',ri, etr);
                if isfield(opt, 'hmc_opt')                
                    fprintf(' %.1e  ',hmcrej);
                end
                if isfield(opt, 'sls_opt')
                    fprintf('sls  ');
                end
                fprintf('\n');
            end
        end
        P_TH = p_th(:)./length(p_th);
    end    

    % =================================================================
    % If targets are given as inputs, make predictions to those targets
    % =================================================================

    if exist('tx') && nargout > 2
        
        % ====================================================================
        % Grid of 501 points around 10 stds to both directions around the mode
        % ====================================================================
        x = zeros(size(Ef_grid,2),501);
        
        if opt.noisy == 0
            for j = 1 : size(Ef_grid,2);
                x(j,:) = Ef_grid(1,j)-10*sqrt(Varf_grid(1,j)) : 20*sqrt(Varf_grid(1,j))/500 : Ef_grid(1,j)+10*sqrt(Varf_grid(1,j));  
            end
        end
        
        
        if opt.noisy == 1
            for i1 = 1 : length(gp_array)
                obsnoise(i1,1) = gp_array{i1}.noise{1}.noiseSigmas2;
            end
            
            for j = 1 : size(Ef_grid,2);
                x(j,:) = Ef_grid(1,j)-10*sqrt(Varf_grid(1,j)+obsnoise(i1,1)) : 20*sqrt(Varf_grid(1,j)+obsnoise(i1,1))/500 : Ef_grid(1,j)+10*sqrt(Varf_grid(1,j)+obsnoise(i1,1));  
            end
            
        end
        
        % Calculate the density in each grid point by integrating over
        % different models
        fx = zeros(size(Ef_grid,2),501);
        for j = 1 : size(Ef_grid,2)
            fx(j,:) = sum(normpdf(repmat(x(j,:),size(Ef_grid,1),1), repmat(Ef_grid(:,j),1,size(x,2)), repmat(sqrt(Varf_grid(:,j)),1,size(x,2))).*repmat(P_TH,1,size(x,2))); 
        end

       
        
        if opt.noisy == 1
            fx = zeros(size(Ef_grid,2),501);
            for j = 1 : size(Ef_grid,2)
                fx(j,:) = sum(normpdf(repmat(x(j,:),size(Ef_grid,1),1), repmat(Ef_grid(:,j),1,size(x,2)), repmat(sqrt(Varf_grid(:,j)+obsnoise),1,size(x,2))).*repmat(P_TH,1,size(x,2))); 
            end
        end
        % Normalize distributions
        fx = fx./repmat(sum(fx,2),1,size(fx,2));

        % Widths of each grid point
        dx = diff(x,1,2);
        dx(:,end+1)=dx(:,end);

        % Calculate mean and variance of the disrtibutions
        Ef = sum(x.*fx,2)./sum(fx,2);
        Varf = sum(fx.*(repmat(Ef,1,size(x,2))-x).^2,2)./sum(fx,2);
    end
    
    % ====================================================================
    % If validation of the approximation is used perform tests
    % ====================================================================
    % - validate the integration over hyperparameters
    % - Check the number of effective parameters in GP:s
    % - Check the normal approximations if Laplace approximation or EP has been used
    if opt.validate == 1 && size(Ef_grid,1) > 1
        % Check the importance weights if used 
        % Check also that the integration over theta has converged 
        switch opt.int_method
          case {'is_normal' 'is_normal_qmc' 'is_student-t'}
            pth_w = P_TH./sum(P_TH);
            meff = 1./sum(pth_w.^2);
            
            figure
            plot(cumsum(sort(pth_w)))
            title('The cumulative mass of importance weights')
            ylabel('cumulative weight, \Sigma_{i=1}^k w_i')
            xlabel('i, the i''th largest integration point')
            
            fprintf('\n \n')
            
            fprintf('The effective number of importance samples is %.2f out of total %.2f samples \n', meff, length(P_TH))
            
% $$$                 fprintf('Validating the integration over hyperparameters...  \n')
% $$$                 
% $$$                 Ef2(:,1) = Ef;
% $$$                 Varf2(:,1) = Varf;
% $$$                 
% $$$                 for i3 = 1:floor(size(Ef_grid,1)./2)
% $$$                     fx2 = zeros(size(Ef_grid,2),501);
% $$$                     for j = 1 : size(Ef_grid,2)
% $$$                         fx2(j,:) = sum(normpdf(repmat(x(j,:),size(Ef_grid,1)-i3,1), repmat(Ef_grid(1:end-i3,j),1,size(x,2)), repmat(sqrt(Varf_grid(1:end-i3,j)),1,size(x,2))).*repmat(P_TH(1:end-i3)./sum(P_TH(1:end-i3)),1,size(x,2))); 
% $$$                     end
% $$$                     
% $$$                     
% $$$                     % Normalize distributions
% $$$                     fx2 = fx2./repmat(sum(fx2,2),1,size(fx2,2));
% $$$                     
% $$$                     % Widths of each grid point
% $$$                     dx = diff(x,1,2);
% $$$                     dx(:,end+1)=dx(:,end);
% $$$                     
% $$$                     % Calculate mean and variance of the disrtibutions
% $$$                     Ef2(:,i3+1) = sum(x.*fx2,2)./sum(fx2,2);
% $$$                     Varf2(:,i3+1) = sum(fx2.*(repmat(Ef2(:,i3),1,size(x,2))-x).^2,2)./sum(fx2,2);
% $$$                 end
% $$$                 for i = 1:size(Varf2,2)-1
% $$$                     KL(:,i) = 0.5*log(Varf2(:,i+1)./Varf2(:,i)) + 0.5 * ( (Ef2(:,i) - Ef2(:,i+1)).^2 + Varf2(:,i) - Varf2(:,i+1) )./Varf2(:,i+1);
% $$$                 end
% $$$                 
% $$$                 KL = fliplr(KL);
% $$$                 if sum(KL(:,end)) < 0.5 % This is a limit whithout good justification
% $$$                     fprintf('The sum of KL-divergences between latent value marginals with %d and %d \n', size(Ef_grid,1), size(Ef_grid,1)-1)
% $$$                     fprintf('integration points is:  %.4e\n', sum(KL(:,end)));
% $$$                     fprintf('The integration seems to have converged.\n')
% $$$                 else
% $$$                     fprintf('The sum of KL-divergences between latent value marginals with %d and %d \n', size(Ef_grid,1), size(Ef_grid,1)-1)
% $$$                     fprintf('integration points is:  %.4e\n', sum(KL(:,end)));
% $$$                     fprintf('Check the integration. There might be problems. \n')
% $$$                 end                
% $$$                             figure
% $$$             plot(size(Ef_grid,1)-size(KL,2)+1:size(Ef_grid,1), sum(KL))
% $$$             title('The convergence of the integration over hyperparameters')
% $$$             ylabel('\Sigma_{i=1}^n KL(q_m(f_i)||q_{m-1}(f_i))')
% $$$             xlabel('m, the number of integration points in hyperparam. space')
        end
            
        % Evaluate the number of effective latent variables in GPs
        for i3 = 1:length(gp_array)
            p_eff(i3) = gp_peff(gp_array{i3}, xx, yy, param);
        end
            
        figure
        plot(p_eff./size(xx,1))
        title('The number of effective latent variables vs. number of latent variables')
        ylabel('p_{eff} / size(f,1)')
        xlabel('m, the index of integration point in hyperparam. space')
        
        fprintf('\n \n')
        
        if max(p_eff./size(xx,1)) < 0.5 % This is a limit whithout good justification
            fprintf('The maximum number of effective latent variables vs. the number of latent \n')
            fprintf('variables is %.4e at integration point %d .\n', max(p_eff./size(xx,1)), find(p_eff == max(p_eff)))
            fprintf('The Normal approximations for the conditional posteriors seem reliable.\n')
        else
            fprintf('The maximum number of effective latent variables vs. the number of latent \n')
            fprintf('variables is %.4e at integration point %d .\n', max(p_eff./size(xx,1)), find(p_eff == max(p_eff)))
            fprintf('The Normal approximations for the conditional posteriors should be checked.\n')
        end
        
        fprintf('\n \n')
    end
    

    % Add the integration weights into the gp_array    
    for i = 1:length(gp_array)
        gp_array{i}.ia_weight = P_TH(i);
    end
    
    function p = mt_pdf(x,Sigma,nu)
        d = length(Sigma);
        for i1 = 1 : size(x,1);
            p(i1) = gamma((nu+1)/2) ./ gamma(nu/2) .* nu^(d/2) .* pi^(d/2) ...
                    .* det(Sigma)^(-.5) .* (1+(1/nu) .* (x(i1,:))*inv(Sigma)*(x(i1,:))')^(-.5*(nu+d));
        end
    end

    function H = hessian(w0)
        
        m = length(w);
        e0 = feval(fh_e,w0,gp,xx,yy,param);
        delta = 1e-4;
        H = -1*ones(m,m);

        % Compute first using gradients
        % If Hessian is singular try computing with 
        % larger step-size
        while any(eig(H)<0) && delta < 1e-2
            for i = 1:m
                for j = i:m
                    w1 = w0; w2 = w0;
                    w1(j) = w1(j) + delta;
                    w2(j) = w2(j) - delta;
                    
                    g1 = feval(fh_g,w1,gp,xx,yy,param);
                    g2 = feval(fh_g,w2,gp,xx,yy,param);
                    
                    H(i,j) = (g1(i)-g2(i))./(2.*delta);
                    H(j,i) = H(i,j);
                end
            end
            delta = delta + 1e-3;
        end
        
        % If the hessian is still singular or the delta is too large 
        % try to compute with finite differences for energies.
        if any(eig(H)<0) || delta > 1e-2
            delta = 1e-4;
            for i=1:m
                w1 = w0; w4 = w0;
                w1(i) = [w1(i)+2*delta];
                w4(i) = [w4(i)-2*delta];
                
                e1 = feval(fh_e,w1,gp,xx,yy,param);
                e4 = feval(fh_e,w4,gp,xx,yy,param);
                
                H(i,i) = (e1 - 2*e0 + e4)./(4.*delta.^2);
                for j = i+1:m
                    w1 = w0; w2 = w0; w3 = w0; w4 = w0;
                    w1([i j]) = [w1(i)+delta w1(j)+delta];
                    w2([i j]) = [w2(i)-delta w2(j)+delta];
                    w3([i j]) = [w3(i)+delta w3(j)-delta];
                    w4([i j]) = [w4(i)-delta w4(j)-delta];
                    
                    e1 = feval(fh_e,w1,gp,xx,yy,param);
                    e2 = feval(fh_e,w2,gp,xx,yy,param);
                    e3 = feval(fh_e,w3,gp,xx,yy,param);
                    e4 = feval(fh_e,w4,gp,xx,yy,param);
                    
                    H(i,j) = (e1 - e2 - e3 + e4)./(4.*delta.^2);
                    H(j,i) = H(i,j);
                end
            end    
        end
        
        % even this does not work so print error
        if any(eig(H)<0)
            warning('gp_ia -> hessian: the Hessian matrix is singular. Check the optimization.')
        end
        
    end

end


% $$$ if ~exist('idx'), idx=size(th,1)+1; end;
% $$$     
% $$$     Ef_grid(idx:end,:)=[];
% $$$     Varf_grid(idx:end,:)=[];
% $$$     th(idx:end,:)=[];
% $$$     P_TH=exp(p_th-min(p_th))./sum(exp(p_th-min(p_th)));
% $$$     
% $$$     clear PPP PP
% $$$     x = -2.5 : 0.01 : 2.5;
% $$$     px = zeros(size(Ef_grid,2),numel(x));
% $$$     for j = 1 : numel(x);
% $$$         px(:,j)= (sum(normpdf(x(j), Ef_grid, sqrt(Varf_grid)).*repmat(P_TH(:),1,size(Ef_grid,2))))';
% $$$     end
% $$$ 
% $$$     clear diff;
% $$$ 
% $$$     px = px./repmat(sum(px,2),1,size(px,2));
% $$$     PPP = px;
% $$$     dx = diff(repmat(x,size(PPP,1),1),1,2);
% $$$     dx(:,end+1)=dx(:,end);
% $$$ 
% $$$     px = px./repmat(sum(px,2),1,size(px,2));
% $$$ 
% $$$     Ef = sum(repmat(x,size(px,1),1).*PPP,2)./sum(PPP,2);
% $$$     Varf = sum(PPP.*(repmat(Ef,1,size(x,2))-repmat(x,size(Ef,1),1)).^2,2)./sum(PPP,2);
% $$$ 
% $$$     fx = PPP; 

% $$$ % =========================================================
% $$$ % Check potential hyperparameter combinations (to be explained)
% $$$ % =========================================================
% $$$ 
% $$$ % Possible steps from the mode
% $$$ pot_dirs = (unique(nchoosek(repmat(1:steps,1,nParam), nParam),'rows')-floor(steps/2)-1);
% $$$ % Corresponding possible hyperparameters
% $$$ pot = (unique(nchoosek(repmat(1:steps,1,nParam),nParam),'rows')-floor(steps/2)-1)*z+repmat(w,size(pot_dirs,1),1);
% $$$ 
% $$$ if nParam == 2
% $$$     dirs = [1 0 ; 0 1 ; -1 0 ; 0 -1 ; 1 1 ; -1 -1 ; 1 -1 ; -1 1];
% $$$ elseif nParam == 3
% $$$     dirs = [1 0 0 ; 0 1 0 ; 0 0 1 ; -1 0 0 ; 0 -1 0 ; 0 0 -1 ; ...
% $$$             1 1 0 ; 1 0 1 ; 0 1 1 ; 1 -1 0 ; -1 1 0 ; -1 0 1 ; 1 0 -1; ...
% $$$             0 -1 1 ; 0 1 -1; -1 -1 0 ; -1 0 -1 ; 0 -1 -1 ;  1 1 1 ; -1 1 1 ; ...
% $$$             1 -1 1 ; 1 1 -1 ; -1 -1 1 ; -1 1 -1 ; 1 -1 -1 ; -1 -1 -1];
% $$$ end
% $$$ 
% $$$ candidates = w;
% $$$ idx = 1;
% $$$ loc = zeros(1,nParam);
% $$$ ind = zeros(1,nParam);
% $$$ checked=candidates;
% $$$ while ~isempty(candidates)
% $$$     wd = candidates(1,:);
% $$$     th(idx,:) = wd;
% $$$     loc = ind(1,:);
% $$$     
% $$$     pot(all(repmat(loc,size(pot_dirs,1),1)==pot_dirs,2),:)=[];
% $$$     pot_dirs(all(repmat(loc,size(pot_dirs,1),1)==pot_dirs,2),:)=[];
% $$$     
% $$$     gp = gp_unpak(gp,wd,param);
% $$$     gp_array{idx} = gp;
% $$$     
% $$$     p_th(idx) = -feval(fh_e,wd,gp,xx,yy,param);
% $$$     
% $$$     if exist('tx')
% $$$         if exist('tstindex')
% $$$             [Ef_grid(idx,:), Varf_grid(idx,:)]=feval(fh_p,gp,xx,yy,tx,param,[],tstindex);
% $$$         else   
% $$$             [Ef_grid(idx,:),Varf_grid(idx,:)] = feval(fh_p,gp,xx,yy,tx,param);
% $$$         end
% $$$     end
% $$$     
% $$$     
% $$$     [I,II]=sort(sum((repmat(loc,size(pot_dirs,1),1)-pot_dirs).^2,2));
% $$$     neigh = pot(II(1:3^nParam-1),:);
% $$$     
% $$$     for j = 1 : size(neigh,1)
% $$$         tmp = neigh(j,:);
% $$$         if ~any(sum(abs(repmat(tmp,size(checked,1),1)-checked),2)==0),
% $$$             error = -feval(fh_e,tmp,gp,xx,yy,param);
% $$$             if -feval(fh_e,w,gp,xx,yy,param) - error < 2.5, 
% $$$                 candidates(end+1,:) = tmp;
% $$$                 ind(end+1,:) = loc+dirs(j,:);
% $$$             end
% $$$             checked(end+1,:) = tmp;
% $$$         end
% $$$     end    
% $$$     candidates(1,:)=[];
% $$$     idx = idx+1;
% $$$     ind(1,:) = [];
% $$$     if isP,candidates,end
% $$$ end

% $$$     P_TH=ones(N,1)/N;
% $$$     p_th = P_TH;
% $$$     
% $$$     if exist('tx')
% $$$         if exist('tstindex')
% $$$             for j = 1 : N
% $$$                 p_th(j) = -feval(fh_e,th(j,:),gp,xx,yy,param);
% $$$                 [Ef_grid(j,:), Varf_grid(j,:)]=feval(fh_p,gp_array{j},xx,yy,tx,param,[],tstindex);
% $$$             end
% $$$         else
% $$$             for j = 1 : N
% $$$                 p_th(j) = -feval(fh_e,th(j,:),gp,xx,yy,param);
% $$$                 [Ef_grid(j,:),Varf_grid(j,:)] = feval(fh_p,gp_array{j},xx,yy,tx,param);
% $$$             end
% $$$         end
% $$$     else
% $$$         p_th(j) = -feval(fh_e,th(j,:),gp,xx,yy,param);
% $$$     end


% $$$     P_TH=ones(N,1)/N;
% $$$     p_th = P_TH;
% $$$     
% $$$     if exist('tx')
% $$$         if exist('tstindex')
% $$$             for j = 1 : N
% $$$                 if isP, j, end;
% $$$                 [Ef_grid(j,:), Varf_grid(j,:)]=feval(fh_p,gp_array{j},xx,yy,tx,param,[],tstindex);
% $$$             end
% $$$         else
% $$$             for j = 1 : N
% $$$                 if isP, j, end;
% $$$                 [Ef_grid(j,:),Varf_grid(j,:)] = feval(fh_p,gp_array{j},xx,yy,tx,param);
% $$$             end
% $$$         end
% $$$     end

% $$$ % ===================
% $$$     % Positive direction
% $$$     % ===================
% $$$ 
% $$$     for dim = 1 : nParam
% $$$         diffe = 0;
% $$$         iter = 1;
% $$$         while diffe < 2.5
% $$$             diffe = -feval(fh_e,w,gp,xx,yy,param) + feval(fh_e,w+z(dim,:)*iter,gp,xx,yy,param);
% $$$             iter = iter + 1;
% $$$             if isP, diffe, end
% $$$         end
% $$$         delta_pos(dim) = iter;
% $$$     end
% $$$     % ==================
% $$$     % Negative direction
% $$$     % ==================
% $$$ 
% $$$     for dim = 1 : nParam
% $$$         diffe = 0;
% $$$         iter = 1;
% $$$         while diffe < 2.5
% $$$             diffe = -feval(fh_e,w,gp,xx,yy,param) + feval(fh_e,w-z(dim,:)*iter, ...
% $$$                                                           gp,xx,yy,param);
% $$$             iter = iter + 1;
% $$$             if isP, diffe, end
% $$$         end
% $$$         delta_neg(dim) = iter;
% $$$     end
% $$$ 
% $$$ 
% $$$ 
% $$$     for j = 1 : nParam
% $$$         delta{j} = -delta_neg(j):1:delta_pos(j);
% $$$         temp(j) = numel(delta{j});
% $$$     end
% $$$     comb = prod(temp);
% $$$ 
% $$$     steps = max(delta_pos+delta_neg+1);
% $$$     p_th = zeros(comb,1);
% $$$     Ef_grid = zeros(comb, numel(yy));
% $$$     Varf_grid = zeros(comb, numel(yy));
% $$$     th = zeros(comb,nParam);
% $$$     pot_dirs = zeros(comb, nParam);
% $$$     pot = zeros(comb, nParam);
% $$$ 
% $$$     % =========================================================
% $$$     % Check potential hyperparameter combinations (to be explained)
% $$$     % =========================================================
% $$$     idx = ones(1,nParam);
% $$$     stepss = delta_pos+delta_neg+1;
% $$$     ind = 1;
% $$$     while any(idx~=stepss)
% $$$         for j = 1 : nParam
% $$$             pot_dirs(ind,j)=delta{j}(idx(j)); 
% $$$         end
% $$$         
% $$$         ind=ind+1;
% $$$         idx(end)=idx(end)+1;
% $$$         while any(idx>stepss)
% $$$             t = find(idx>stepss);
% $$$             idx(t)=1;
% $$$             idx(t-1)=idx(t-1)+1;
% $$$         end
% $$$     end
% $$$ 
% $$$     for j = 1 : nParam
% $$$         pot_dirs(ind,j)=delta{j}(idx(j)); 
% $$$     end
% $$$ 
% $$$     
% $$$     
% $$$     
% $$$     
% $$$     % Possible steps from the mode
% $$$     % pot_dirs = (unique(nchoosek(repmat(1:steps,1,nParam), nParam),'rows')-floor(steps/2)-1);
% $$$     % Corresponding possible hyperparameters
% $$$     pot = pot_dirs*z+repmat(w,size(pot_dirs,1),1);
% $$$ 
% $$$     candidates = w;
% $$$     idx = 1;
% $$$     candit=[];
% $$$     loc = zeros(1,nParam);
% $$$     ind = zeros(1,nParam);
% $$$     checked=candidates;
% $$$     while ~isempty(candidates)
% $$$         wd = candidates(1,:);
% $$$         th(idx,:) = wd;
% $$$         loc = ind(1,:);
% $$$         
% $$$         pot(all(repmat(loc,size(pot_dirs,1),1)==pot_dirs,2),:)=[];
% $$$         pot_dirs(all(repmat(loc,size(pot_dirs,1),1)==pot_dirs,2),:)=[];
% $$$         
% $$$         gp = gp_unpak(gp,wd,param);
% $$$         gp_array{idx} = gp;
% $$$         
% $$$         p_th(idx) = -feval(fh_e,wd,gp,xx,yy,param);
% $$$         
% $$$         if exist('tx')
% $$$             if exist('tstindex')
% $$$                 [Ef_grid(idx,:), Varf_grid(idx,:)]=feval(fh_p,gp,xx,yy,tx,param,[],tstindex);
% $$$             else   
% $$$                 [Ef_grid(idx,:),Varf_grid(idx,:)] = feval(fh_p,gp,xx,yy,tx,param);
% $$$             end
% $$$         end
% $$$         
% $$$         
% $$$         [I,II]=sort(sum((repmat(loc,size(pot_dirs,1),1)-pot_dirs).^2,2));
% $$$         neigh = pot(II(1:3^nParam-1),:);
% $$$         
% $$$         for j = 1 : size(neigh,1)
% $$$             tmp = neigh(j,:);
% $$$             if ~any(sum(abs(repmat(tmp,size(checked,1),1)-checked),2)==0),
% $$$                 error = -feval(fh_e,tmp,gp,xx,yy,param);
% $$$                 if -feval(fh_e,w,gp,xx,yy,param) - error < 2.5, 
% $$$                     candidates(end+1,:) = tmp;
% $$$                     candit(end+1,:)=tmp;
% $$$                     %ind(end+1,:) = loc+dirs(j,:);
% $$$                     ind(end+1,:) = pot_dirs(II(j),:);
% $$$                 end
% $$$                 checked(end+1,:) = tmp;
% $$$             end
% $$$         end    
% $$$         candidates(1,:)=[];
% $$$         idx = idx+1;
% $$$         ind(1,:) = [];
% $$$         if isP,candidates,end
% $$$     end
% $$$ 
% $$$ 
% $$$     p_th(idx:end)=[];
% $$$     P_TH=exp(p_th-min(p_th))./sum(exp(p_th-min(p_th)));

% $$$           case 'is_normal_qmc' 
% $$$             delta = -2:.1:2;
% $$$             for i0 = 1 : nParam
% $$$                 for i1 = 1 : length(delta)
% $$$                     ttt = zeros(1,nParam);
% $$$                     ttt(i0)=1;
% $$$                     phat = (-feval(fh_e,w+(delta(i1)*chol(Sigma)'*ttt')',gp,xx,yy,param));
% $$$                     fi(i1) = abs(delta(i1)).*(2.*(P0-phat)).^(-.5);
% $$$                     
% $$$                     pp(i1) = exp(phat);
% $$$                     pt(i1) = mvnpdf(delta(i1)*chol(Sigma)'*ttt', 0, Sigma);
% $$$                 end
% $$$                 
% $$$                 origo = (length(delta)-1)/2;
% $$$                 
% $$$                 q(i0) = max(fi(delta>0));
% $$$                 r(i0) = max(fi(delta<0));
% $$$                 
% $$$                 scl = ones(1,length(delta));
% $$$                 scl(1:floor(length(delta)/2))=repmat(r(i0),1,floor(length(delta)/2));
% $$$                 
% $$$                 scl(ceil(length(delta)/2):end)=repmat(q(i0),1,ceil(length(delta)/2));
% $$$                 
% $$$                 in{i0} = scl.*delta;
% $$$                 out{i0} = pt/max(pt);
% $$$                 in2off = delta;
% $$$ % $$$                 figure; plot(delta,pp/max(pp),'linewidth',2); hold on; plot(scl.*delta,pt/max(pt),'r','linewidth',2)
% $$$ % $$$                 legend('posterior','importance')
% $$$ % $$$                 set(gca,'fontsize',16,'fontweight','bold','ytick',[],'xlim',[-6,6]);
% $$$ % $$$                 figure; plot(delta,pp/max(pp),'linewidth',2); hold on; plot(delta,pt/max(pt),'r','linewidth',2)
% $$$ % $$$                 legend('posterior','importance')
% $$$ % $$$                 set(gca,'fontsize',16,'fontweight','bold','ytick',[],'xlim',[-6,6]);
% $$$                 
% $$$             end
% $$$             % Quasi MC samples
% $$$             th  = repmat(w,N,1)+(chol(Sigma)'*(sqrt(2).*erfinv(2.*hammersley(size(Sigma,1),N) - 1)))';
% $$$             p_th_appr = mvnpdf(th, w, Sigma);        
% $$$             if strcmp(opt.improved,'on');
% $$$                 e=(sqrt(2).*erfinv(2.*hammersley(size(Sigma,1),N) - 1))';
% $$$                 for i3 = 1 : N
% $$$                     for i2 = 1 : nParam
% $$$                         if e(i3,i2)<0
% $$$                             eta(i3,i2) = e(i3,i2)*r(i2);
% $$$                         else
% $$$                             eta(i3,i2) = e(i3,i2)*q(i2);
% $$$                         end
% $$$                         p_th_appr(i3) = mvnpdf(e(i3,:));
% $$$                     end
% $$$                     %                if i3 == 1,keyboard, end
% $$$                     th(i3,:)=w+(chol(Scale)'*eta(i3,:)')';
% $$$                 end
% $$$                 p_th_appr = mvnpdf(th, w, Sigma);        
% $$$             end
