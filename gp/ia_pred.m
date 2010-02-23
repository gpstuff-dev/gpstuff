function [Ef, Varf, f, ff, Ey, Vary, y, fy, py] = ia_pred(gp_array, tx, ty, x, predcf, tstind, Y) 
    
    
    
% Copyright (c) 2009 Ville Pietiläinen

% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.    
    
    tn = size(tx,1);
    if nargin < 4
        error('Requires at least 4 arguments');
    end

    if nargin < 7
        Y = [];
    end
    
    if nargout > 8 && isempty(Y)
        error('gp_pred -> If py is wanted you must provide the vector y as 7''th input.')
    end
    
    if nargin < 6
        tstind = [];
    end
    
    if nargin < 5
        predcf = [];
    end
    
% $$$     nin  = gp.nin;
% $$$     nout = gp.nout;

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
        if exist('tstindex')
            if ~isfield(gp_array{1}, 'latent_method')
                [Ef_grid(j,:), Varf_grid(j,:)]=feval(fh_p,gp_array{j},tx,ty,x,[],tstindex);
                Ey_grid(j,:) = Ef_grid(j,:);
                Vary_grid(j,:) = Varf_grid(j,:)+gp_array{j}.noise{1}.noiseSigmas2;
            else
                [Ef_grid(j,:), Varf_grid(j,:)]=feval(fh_p,gp_array{j},tx,ty,x,param,[],tstindex);
                Ey_grid(j,:) = Ef_grid(j,:);
                Vary_grid(j,:) = Varf_grid(j,:)+gp_array{j}.noise{1}.noiseSigmas2;
            end
        else
            if isfield(gp_array{1}, 'latent_method')   
                [Ef_grid(j,:),Varf_grid(j,:)]=feval(fh_p,gp_array{j}, tx, ty, x, param);
                Ey_grid(j,:) = Ef_grid(j,:);
                Vary_grid(j,:) = Varf_grid(j,:)+gp_array{j}.noise{1}.noiseSigmas2;
            else
                [Ef_grid(j,:), Varf_grid(j,:)]=feval(fh_p,gp_array{j}, tx, ty, x);
                Ey_grid(j,:) = Ef_grid(j,:);
                Vary_grid(j,:) = Varf_grid(j,:)+gp_array{j}.noise{1}.noiseSigmas2;
            end
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
    
    % ==============================
    % Observations y
    % ==============================
    
    y = zeros(size(Ey_grid,2),501);
    for j = 1 : size(Ey_grid,2);
        y(j,:) = Ey_grid(1,j)-10*sqrt(Vary_grid(1,j)) : 20*sqrt(Vary_grid(1,j))/500 : Ey_grid(1,j)+10*sqrt(Vary_grid(1,j));  
    end
    
    % Calculate the density in each grid point by integrating over
    % different models
    fy = zeros(size(Ey_grid,2),501);
    for j = 1 : size(Ey_grid,2)
        fy(j,:) = sum(normpdf(repmat(y(j,:),size(Ey_grid,1),1), repmat(Ey_grid(:,j),1,size(y,2)), repmat(sqrt(Vary_grid(:,j)),1,size(y,2))).*repmat(P_TH,1,size(y,2)),1); 
    end

    % Normalize distributions
    fy = fy./repmat(sum(fy,2),1,size(fy,2));

    % Widths of each grid point
    dy = diff(y,1,2);
    dy(:,end+1)=dy(:,end);

    % Calculate mean and variance of the disrtibutions
    Ey = sum(y.*fy,2)./sum(fy,2);
    Vary = sum(fy.*(repmat(Ey,1,size(y,2))-y).^2,2)./sum(fy,2);
    
    if nargin == 7
        for i1 = 1 : length(Y)
            py(i1) = sum(normpdf(repmat(Y(i1),size(Ey_grid,1),1), Ey_grid(:,i1), sqrt(Vary_grid(:,i1))).*P_TH); 
        end
    end