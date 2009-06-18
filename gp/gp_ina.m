function [gp_array, P_TH, Ef, Varf, x, fx] = gp_ina(opt, gp, xx, yy, tx, param, tstindex)
% GP_INA explores the hypeparameters around the mode and returns a
% list of GPs with different hyperparameters and corresponding weights
%
%       Description [GP_ARRAY, P_TH, EF, VARF, X, FX] = GP_INA(OPT,
%       GP, XX, YY, TX, PARAM, TSTINDEX) takes a gp data structure GP
%       with covariates XX and observations YY and returns an array of
%       GPs GP_ARRAY and corresponding weights P_TH. Iff test
%       covariates TX is included, GP_INA also returns corresponding
%       mean EF and variance VARF (FX is PDF evaluated at X). TSTINDEX
%       is for FIC. 

% isP = true for some more or less (probably less) useful printing during the
% algorithm. This print makes it easier to see how long the algortihm will run
isP=logical(1);

% ===========================
% Which latent method is used
% ===========================
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

if nargin < 6
    param = 'hyper';
end

opt=optimset('GradObj','on');
% opt=optimset(opt,'TolX', 1e-3);
opt=optimset(opt,'LargeScale', 'on');
opt=optimset(opt,'Display', 'iter');

% ====================================
% Find the mode of the hyperparameters
% ====================================

w0 = gp_pak(gp, param);
gradcheck(w0, fh_e, fh_g, gp, xx, yy, param)
mydeal = @(varargin)varargin{1:nargout};

% The mode and hessian at it 
[w,fval,exitflag,output,grad,H] = fminunc(@(ww) mydeal(feval(fh_e,ww, gp, xx, yy, param), feval(fh_g, ww, gp, xx, yy, param)), w0, opt);
gp = gp_unpak(gp,w,param);

% Number of parameters
nParam = size(H,1);

% ===============================
% New variable z for exploration
% ===============================

Sigma = inv(H);
[V,D] = eig(full(Sigma));
z = (V*sqrt(D))';

% ===================
% Positive direction
% ===================

for dim = 1 : nParam
    diffe = 0;
    iter = 1;
    while diffe < 2.5
        diffe = -feval(fh_e,w,gp,xx,yy,param) + feval(fh_e,w+z(dim,:)*iter,gp,xx,yy,param);
        iter = iter + 1;
        if isP, diffe, end
    end
    delta_pos(dim) = iter;
end
% ==================
% Negative direction
% ==================

for dim = 1 : nParam
    diffe = 0;
    iter = 1;
    while diffe < 2.5
        diffe = -feval(fh_e,w,gp,xx,yy,param) + feval(fh_e,w-z(dim,:)*iter, ...
                                                      gp,xx,yy,param);
        iter = iter + 1;
        if isP, diffe, end
    end
    delta_neg(dim) = iter;
end

steps = max(delta_pos+delta_neg+1);
p_th = zeros(steps^nParam,1);
Ef_grid = zeros(steps^nParam, numel(yy));
Varf_grid = zeros(steps^nParam, numel(yy));
th = zeros(steps^nParam,nParam);

% =========================================================
% Check potential hyperparameter combinations (to be explained)
% =========================================================

% Possible steps from the mode
pot_dirs = (unique(nchoosek(repmat(1:steps,1,nParam), nParam),'rows')-floor(steps/2)-1);
% Corresponding possible hyperparameters
pot = (unique(nchoosek(repmat(1:steps,1,nParam),nParam),'rows')-floor(steps/2)-1)*z+repmat(w,size(pot_dirs,1),1);

candidates = w;
idx = 1;
candit=[];
loc = zeros(1,nParam);
ind = zeros(1,nParam);
checked=candidates;
while ~isempty(candidates)
    wd = candidates(1,:);
    th(idx,:) = wd;
    loc = ind(1,:);
    
    pot(all(repmat(loc,size(pot_dirs,1),1)==pot_dirs,2),:)=[];
    pot_dirs(all(repmat(loc,size(pot_dirs,1),1)==pot_dirs,2),:)=[];
    
    gp = gp_unpak(gp,wd,param);
    gp_array{idx} = gp;
    
    p_th(idx) = -feval(fh_e,wd,gp,xx,yy,param);
    
    if exist('tx')
        if exist('tstindex')
            [Ef_grid(idx,:), Varf_grid(idx,:)]=feval(fh_p,gp,xx,yy,tx,param,[],tstindex);
        else   
            [Ef_grid(idx,:),Varf_grid(idx,:)] = feval(fh_p,gp,xx,yy,tx,param);
        end
    end
    
    
    [I,II]=sort(sum((repmat(loc,size(pot_dirs,1),1)-pot_dirs).^2,2));
    neigh = pot(II(1:3^nParam-1),:);
    
    for j = 1 : size(neigh,1)
        tmp = neigh(j,:);
        if ~any(sum(abs(repmat(tmp,size(checked,1),1)-checked),2)==0),
            error = -feval(fh_e,tmp,gp,xx,yy,param);
            if -feval(fh_e,w,gp,xx,yy,param) - error < 2.5, 
                candidates(end+1,:) = tmp;
                candit(end+1,:)=tmp;
                %ind(end+1,:) = loc+dirs(j,:);
                ind(end+1,:) = pot_dirs(II(j),:);
            end
            checked(end+1,:) = tmp;
        end
    end    
    candidates(1,:)=[];
    idx = idx+1;
    ind(1,:) = [];
    if isP,candidates,end
end

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

p_th(idx:end)=[];
P_TH=exp(p_th-min(p_th))./sum(exp(p_th-min(p_th)));

if exist('tx')
    
    Ef_grid(idx:end,:)=[];
    Varf_grid(idx:end,:)=[];
    th(idx:end,:)=[];
    P_TH=exp(p_th-min(p_th))./sum(exp(p_th-min(p_th)));
    
    clear PPP PP
    x = -2.5 : 0.01 : 2.5;
    px = zeros(size(Ef_grid,2),numel(x));
    for j = 1 : numel(x);
        px(:,j)= (sum(normpdf(x(j), Ef_grid, sqrt(Varf_grid)).*repmat(P_TH(:),1,size(Ef_grid,2))))';
    end

    clear diff;

    px = px./repmat(sum(px,2),1,size(px,2));
    PPP = px;
    dx = diff(repmat(x,size(PPP,1),1),1,2);
    dx(:,end+1)=dx(:,end);

    px = px./repmat(sum(px,2),1,size(px,2));

    Ef = sum(repmat(x,size(px,1),1).*PPP,2)./sum(PPP,2);
    Varf = sum(PPP.*(repmat(Ef,1,size(x,2))-repmat(x,size(Ef,1),1)).^2,2)./sum(PPP,2);

    fx = PPP; 
end
