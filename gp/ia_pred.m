function [Ef, Varf, Ey, Vary, py, f, ff] = ia_pred(gp_array, tx, ty, x, param, predcf, tstind, Y) 
%IA_PRED	Prediction with Gaussian Process IA solution.
%
%	Description
%	[Ef, Varf] = IA_PRED(GP_ARRAY, TX, TY, X, PREDCF, TSTIND) takes a Gaussian 
%       processes record array RECGP (returned by gp_ia) together with a matrix X 
%       of input vectors, matrix TX of training inputs and vector TY of training targets. 
%       Returns the predictive mean and variance, Ef and Varf, for test inputs X. 
%
%       Each row of X corresponds to one input vector and each row of Y corresponds to one 
%       output. PREDCF is an array specifying the indexes of covariance functions, which 
%       are used for making the prediction (others are considered noise). TSTIND is, in 
%       case of PIC, a cell array containing index vectors specifying the blocking 
%       structure for test data, or in FIC and CS+FI a vector of length n that points out 
%       the test inputs that are also in the training set (if none, set TSTIND = []).
%       
%       [Ef, Varf, Ey, Vary] = IA_PREDS(GP, TX, TY, X, PREDCF, TSTIND) returns also the 
%       predictive means and variances for observations at input locations X. That is,
%
%                    Ey(:,i) = E[y | x, tx, ty]
%                  Vary(:,i) = Var[y | x, tx, ty]
%    
%       [Ef, Varf, Ey, Vary, py] = IA_PREDS(GP, TX, TY, X, PREDCF, TSTIND, Y) 
%       returns also the predictive density py of test output Y, that is py = p(Y).
%
%       [Ef, Varf, Ey, Vary, py, f, ff] = IA_PREDS(GP, TX, TY, X, PREDCF, TSTIND) returns also the 
%       numerical representation of the marginal posterior of latent variables at each X. f is
%       a vector of latent values and ff_i = p(f_i) is the posterior density for f_i.

%
%	See also
%	GP, GP_PAK, GP_UNPAK, GP_PRED

        
% Copyright (c) 2009 Ville Pietiläinen
% Copyright (c) 2010 Jarno Vanhatalo    

% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.    
    
    tn = size(tx,1);
    if nargin < 4
        error('Requires at least 4 arguments');
    end

    if nargin < 8 
        Y = [];
    end
    
    if nargout > 4 && isempty(Y)
        py = NaN;
    end
    
    if nargin < 7
        tstind = [];
    end
    
    if nargin < 6
        predcf = [];
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
        if isempty(Y)
            [Ef_grid(j,:), Varf_grid(j,:), Ey_grid(j,:), Vary_grid(j,:)]=feval(fh_p,gp_array{j},tx,ty,x,param,[],tstind);
        else
            [Ef_grid(j,:), Varf_grid(j,:), Ey_grid(j,:), Vary_grid(j,:), py_grid(j,:)]=feval(fh_p,gp_array{j},tx,ty,x, param, [],tstind, Y);
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
    
    if ~isempty(Y)
        py = sum(py_grid.*repmat(P_TH,1,size(Ey_grid,2)),1);
        py = py';
    end
    
    % Take transposes
    Ef = Ef';
    Varf = Varf';
    Ey = Ey';    
    Vary = Vary';