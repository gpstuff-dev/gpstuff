function [Ef, Varf, x, fx] = gp_int(gp_array, xx, yy, tx,param, tstindex)
% GP_INT        Predictions by integrating over GP models
%
%       Description 
%       [EF, VARF, X, FX] = GP_INT(GP_ARRAY, XX, YY,TX) takes an array
%       of gp data structures GP_ARRAY together with a matrix TX of
%       input vectors, matrix XX of training inputs and vector YY of 
%       training targets, and evaluates the predictive distribution at 
%       inputs by integrating over models in GP_ARRAY. Returns a matrix 
%       EF of (noiseless) output vectors. Each row of TX corresponds to
%       one input vector and each row of EF corresponds to one output 
%       vector. VARF is the variance of the predictions. FX is the 
%       probability density of output evaluated in points in X. Each
%       row of X and FX correspond to one output vector.
%    
%       [EF, VARF, X, FX] = GP_INT(GP_ARRAY, XX, YY,TX,'PARAM')
%       in the case of sparse model, takes also string PARAM defining,  
%       which parameters have been optimized 
%       
%       [EF, VARF, X, FX] = GP_INT(GP_ARRAY, XX, YY,TX,'PARAM', TSTINDEX)
%       TSTINDEX is needed with FIC model when inputs are same as training inputs     
        
% Copyright (c) 2009 Ville Pietiläinen

% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.

    if numel(gp_array)~=numel(P_TH)
        disp('Error Error')
        return
    end
    
    nGP = numel(gp_array);
    
    for i=1:nGP
        P_TH(i) = gp_array{i}.ia_weight;
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
        if exist('tstindex')
            if ~isfield(gp_array{1}, 'latent_method')
                [Ef_grid(j,:), Varf_grid(j,:)]=feval(fh_p,gp_array{j},xx,yy,tx,[],tstindex);
            else
                [Ef_grid(j,:), Varf_grid(j,:)]=feval(fh_p,gp_array{j},xx,yy,tx,param,[],tstindex);
            end
        else
            if isfield(gp_array{1}, 'latent_method')   
                [Ef_grid(j,:),Varf_grid(j,:)]=feval(fh_p,gp_array{j}, xx, yy, tx, param);
            else
                [Ef_grid(j,:),Varf_grid(j,:)]=feval(fh_p,gp_array{j}, xx, yy, tx);
            end
        end
    end

    % ====================================================================
    % Grid of 501 points around 10 stds to both directions around the mode
    % ====================================================================

    x = zeros(size(Ef_grid,2),501);
    for j = 1 : size(Ef_grid,2);
        x(j,:) = Ef_grid(1,j)-10*sqrt(Varf_grid(1,j)) : 20*sqrt(Varf_grid(1,j))/500 : Ef_grid(1,j)+10*sqrt(Varf_grid(1,j));  
    end
    
    % Calculate the density in each grid point by integrating over
    % different models
    fx = zeros(size(Ef_grid,2),501);
    for j = 1 : size(Ef_grid,2)
        fx(j,:) = sum(normpdf(repmat(x(j,:),size(Ef_grid,1),1), repmat(Ef_grid(:,j),1,size(x,2)), repmat(sqrt(Varf_grid(:,j)),1,size(x,2))).*repmat(P_TH,1,size(x,2))); 
    end

    % Normalize distributions
    fx = fx./repmat(sum(fx,2),1,size(fx,2));

    % Widths of each grid point
    dx = diff(x,1,2);
    dx(:,end+1)=dx(:,end);

    % Calculate mean and variance of the disrtibutions
    Ef = sum(x.*fx,2)./sum(fx,2);
    Varf = sum(fx.*(repmat(Ef,1,size(x,2))-x).^2,2)./sum(fx,2);

