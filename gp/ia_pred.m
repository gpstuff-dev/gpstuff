function [Ef, Varf, Ey, Vary, py, f, ff] = ia_pred(gp_array, x, y, xt, varargin) 
%IA_PRED	Prediction with Gaussian Process IA solution.
%
%	Description
%	[Ef, Varf] = IA_PRED(GP_ARRAY, X, Y, XT, OPTIONS) takes a Gaussian 
%       processes record array RECGP (returned by gp_ia) together with a matrix XT 
%       of input vectors, matrix X of training inputs and vector Y of training targets. 
%       Returns the predictive mean and variance, Ef and Varf, for test inputs XT. 
%
%
%
%     OPTIONS is optional parameter-value pair
%       'predcf' is index vector telling which covariance functions are 
%                used for prediction. Default is all (1:gpcfn)
%       'tstind' is a vector defining, which rows of X belong to which 
%                training block in *IC type sparse models. Default is [].
%       'yt' is optional observed yt in test points (see below)
%       'z' is optional observed quantity in triplet (x_i,y_i,z_i)
%         Some likelihoods may use this. For example, in case of Poisson
%         likelihood we have z_i=E_i, that is, expected value for ith case. 
%       'zt' is optional observed quantity in triplet (xt_i,yt_i,zt_i)
%         Some likelihoods may use this. For example, in case of Poisson
%         likelihood we have z_i=E_i, that is, expected value for ith case. 
%       
%       [Ef, Varf, Ey, Vary] = IA_PREDS(GP, X, Y, XT, OPTIONS) returns also the 
%       predictive means and variances for observations at input locations XT. That is,
%
%                    Ey(:,i) = E[y | xt, x, y]
%                  Vary(:,i) = Var[y | xt, x, y]
%    
%       [Ef, Varf, Ey, Vary, py] = IA_PREDS(GP, X, Y, XT, 'yt', YT, OPTIONS) 
%       returns also the predictive density py of test outputs YT, that is py(i) = p(YT_i).
%
%       [Ef, Varf, Ey, Vary, py, f, ff] = IA_PREDS(GP, X, Y, XT, OPTIONS) returns also the 
%       numerical representation of the marginal posterior of latent variables at each XT. f is
%       a vector of latent values and ff_i = p(f_i) is the posterior density for f_i.

%
%	See also
%	GP, GP_PAK, GP_UNPAK, GP_PRED

        
% Copyright (c) 2009 Ville Pietiläinen
% Copyright (c) 2010 Jarno Vanhatalo    

% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.    

    
    ip=inputParser;
    ip.FunctionName = 'IA_PRED';
    ip.addRequired('gp_array', @iscell);
    ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addRequired('xt', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                     isvector(x) && isreal(x) && all(isfinite(x)&x>0))
    ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                     (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
    ip.parse(gp_array, x, y, xt, varargin{:});
    yt=ip.Results.yt;
    z=ip.Results.z;
    zt=ip.Results.zt;
    predcf=ip.Results.predcf;
    tstind=ip.Results.tstind;
    
    % pass these forward
    options=struct();
    if ~isempty(ip.Results.yt);options.yt=ip.Results.yt;end
    if ~isempty(ip.Results.z);options.z=ip.Results.z;end
    if ~isempty(ip.Results.predcf);options.predcf=ip.Results.predcf;end
    if ~isempty(ip.Results.tstind);options.tstind=ip.Results.tstind;end
    
    if nargout > 4 && isempty(yt)
        py = NaN;
    end
        
    nGP = numel(gp_array);
    
    for i=1:nGP
        P_TH(i,:) = gp_array{i}.ia_weight;
    end

    % =======================================================
    % Select the functions corresponding to the latent_method
    % =======================================================

    if isfield(gp_array{1}, 'latent_method')
        switch gp_array{1}.latent_method
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
    
    % ==================================================
    % Make predictions with different models in gp_array
    % ==================================================

    for j = 1 : nGP
        if isempty(yt)
            [Ef_grid(j,:), Varf_grid(j,:), Ey_grid(j,:), Vary_grid(j,:)]=feval(fh_p,gp_array{j},x,y,xt,options);
        else
            [Ef_grid(j,:), Varf_grid(j,:), Ey_grid(j,:), Vary_grid(j,:), py_grid(j,:)]=feval(fh_p,gp_array{j},x,y,xt, options);
        end
    end
    
    % ====================================================================
    % Grid of 501 points around 10 stds to both directions around the mode
    % ====================================================================

    % ==============================
    % Latent variables f
    % ==============================
    
    f = zeros(size(Ef_grid,2),501);
    for j = 1 : size(Ef_grid,2);
        f(j,:) = Ef_grid(1,j)-10*sqrt(Varf_grid(1,j)) : 20*sqrt(Varf_grid(1,j))/500 : Ef_grid(1,j)+10*sqrt(Varf_grid(1,j));  
    end
    
    % Calculate the density in each grid point by integrating over
    % different models
    ff = zeros(size(Ef_grid,2),501);
    for j = 1 : size(Ef_grid,2)
        ff(j,:) = sum(normpdf(repmat(f(j,:),size(Ef_grid,1),1), repmat(Ef_grid(:,j),1,size(f,2)), repmat(sqrt(Varf_grid(:,j)),1,size(f,2))).*repmat(P_TH,1,size(f,2)),1); 
    end

    % Normalize distributions
    ff = ff./repmat(sum(ff,2),1,size(ff,2));

    % Widths of each grid point
    df = diff(f,1,2);
    df(:,end+1)=df(:,end);

    % Calculate mean and variance of the disrtibutions
    Ef = sum(f.*ff,2)./sum(ff,2);
    Varf = sum(ff.*(repmat(Ef,1,size(f,2))-f).^2,2)./sum(ff,2);
    
    Ey = sum(Ey_grid.*repmat(P_TH,1,size(Ey_grid,2)),1);
    Vary = sum(Vary_grid.*repmat(P_TH,1,size(Ey_grid,2)),1) + sum( (Ey_grid - repmat(Ey,nGP,1)).^2, 1);
    
    if ~isempty(yt)
        py = sum(py_grid.*repmat(P_TH,1,size(Ey_grid,2)),1);
        py = py';
    end
    
    % Take transposes
    Ef = Ef';
    Varf = Varf';
    Ey = Ey';    
    Vary = Vary';