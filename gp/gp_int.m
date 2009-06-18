function [Ef, Varf, x, fx] = gp_int(gp_array, P_TH, xx, yy, tx,param, tstindex)
% Averages over models in gp_array weighting with probabilities in P_TH
    if numel(gp_array)~=numel(P_TH)
        disp('Error Error')
        return
    end
    
    nGP = numel(gp_array)
    
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
    
    for j = 1 : nGP
        if exist('tstindex')
            [Ef_grid(j,:), Varf_grid(j,:)]=feval(fh_p,gp_array{j},xx,yy,tx,param,[],tstindex);
        else   
            [Ef_grid(j,:),Varf_grid(j,:)]=feval(fh_p,gp_array{j}, xx, yy, tx, param);
        end
    end
        
    x = -2.5 : 0.01 : 2.5;
    fx = zeros(size(Ef_grid,2),numel(x));
    for j = 1 : numel(x);
        fx(:,j)= (sum(normpdf(x(j), Ef_grid, sqrt(Varf_grid)).*repmat(P_TH(:),1,size(Ef_grid,2))))';
    end

    fx = fx./repmat(sum(fx,2),1,size(fx,2));

    dx = diff(repmat(x,size(fx,1),1),1,2);
    dx(:,end+1)=dx(:,end);

    Ef = sum(repmat(x,size(fx,1),1).*fx,2)./sum(fx,2);
    Varf = sum(fx.*(repmat(Ef,1,size(x,2))-repmat(x,size(Ef,1),1)).^2,2)./sum(fx,2);

    fx = fx; 