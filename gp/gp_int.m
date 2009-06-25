function [Ef, Varf, x, fx] = gp_int(gp_array, P_TH, xx, yy, tx,param, tstindex)
% Averages over models in gp_array weighting with probabilities in P_TH
    if numel(gp_array)~=numel(P_TH)
        disp('Error Error')
        return
    end
    
    nGP = numel(gp_array)
    
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

%keyboard
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

x = zeros(size(Ef_grid,2),501);
for j = 1 : size(Ef_grid,2);
    x(j,:) = Ef_grid(1,j)-10*sqrt(Varf_grid(1,j)) : 20*sqrt(Varf_grid(1,j))/500 : Ef_grid(1,j)+10*sqrt(Varf_grid(1,j));  
end
% x = -2.5 : 0.01 : 2.5;
fx = zeros(size(Ef_grid,2),501);

for j = 1 : size(Ef_grid,2)
    fx(j,:) = sum(normpdf(repmat(x(j,:),size(Ef_grid,1),1), repmat(Ef_grid(:,j),1,size(x,2)), repmat(sqrt(Varf_grid(:,j)),1,size(x,2))).*repmat(P_TH,1,size(x,2))); 
end


% $$$     for j = 1 : numel(x);
% $$$         fx(:,j)= (sum(normpdf(x(j), Ef_grid, sqrt(Varf_grid)).*repmat(P_TH(:),1,size(Ef_grid,2))))';
% $$$     end

fx = fx./repmat(sum(fx,2),1,size(fx,2));

dx = diff(x,1,2);
dx(:,end+1)=dx(:,end);

Ef = sum(x.*fx,2)./sum(fx,2);
Varf = sum(fx.*(repmat(Ef,1,size(x,2))-x).^2,2)./sum(fx,2);

fx = fx; 