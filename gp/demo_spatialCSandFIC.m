function demo_spatialCSandFIC
%   Author: Jarno Vanhatalo <jarno.vanhatalo@tkk.fi>
%   Last modified: 2008-05-02 11:19:08 EEST
    
% First load the data
%=======================================================================
    load /proj/finnwell/spatial/data/tilastok2007/testdata/aivoverisuonitaudit911_9600.mat
    xxa=data(:,1:2);
    yna=data(:,6);
    xx=unique(xxa,'rows');
    for i1=1:size(xx,1)
        xxi{i1,1}=find(xxa(:,1)==xx(i1,1)&xxa(:,2)==xx(i1,2));
        yn(i1,1)=sum(data(xxi{i1},6));
        yy(i1,1)=sum(data(xxi{i1},7));
    end
    xxii=sub2ind([60 35],xx(:,2),xx(:,1));
    [X1,X2]=meshgrid(1:35,1:60);
    N=zeros(60,35);
    Y=zeros(60,35);
    N(xxii)=yn;
    Y(xxii)=yy;

    gxa=data(:,3:5);
    gxx=unique(gxa,'rows');
    gxx=gxx(gxx(:,1)>=4,:);
    for i1=1:size(gxx,1)
        gxxi{i1}=find(gxa(:,1)==gxx(i1,1)&gxa(:,2)==gxx(i1,2)&gxa(:,3)==gxx(i1,3));
        gyn(i1,1)=sum(data(gxxi{i1},6));
        gyy(i1,1)=sum(data(gxxi{i1},7));
    end
    ra=sum(yy)./sum(yn); % average risk
    gye=ra.*gyn;
    gra=gyy./gyn; % average risk for each group

    ea=zeros(size(xxa,1),1);
    for i1=1:numel(gxxi)
        ea(gxxi{i1})=gra(i1).*yna(gxxi{i1});
    end
    EA=zeros(60,35)+NaN;
    for i1=1:numel(xxi)
        EA(xxii(i1))=sum(ea(xxi{i1}));
    end
    ye=EA(xxii);
    %=======================================================================

    xx = xx(1:100,:);
    yy = yy(1:100,:);
    ye = ye(1:100,:);

        
    

    % Set the inducing inputs
    bls = 4; indtype = 'corners';
    [blockindex, Xu] = set_PIC(xx, dims, cellsize, bls, indtype, 1);

    % Create the model
    [n, nin] = size(xx);

    gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 20, 'magnSigma2', 0.1);
    gpcf1.p.lengthScale = t_p({1 4});
    gpcf1.p.magnSigma2 = t_p({0.3 4});

    gpcf2 = gpcf_ppcs2('init', nin, 'lengthScale', 3, 'magnSigma2', 0.1);
    gpcf2.p.lengthScale = t_p({1 4});
    gpcf2.p.magnSigma2 = t_p({0.3 4});

    likelih = likelih_poisson('init', yy, ye);
    gp = gp_init('init', 'CS+FIC', nin, likelih, {gpcf1, gpcf2}, [], 'jitterSigmas', 0.01, 'X_u', Xu);   %{gpcf2}
    gp.avgE = ye; 
    gp = gp_init('set', gp, 'latent_method', {'EP', xx, yy, 'hyper'});

    e = gpep_e(gp_pak(gp,'hyper'), gp, xx, yy, 'hyper')
    g = gpep_g(gp_pak(gp,'hyper'), gp, xx, yy, 'hyper')

    
    likelih = likelih_poisson('init', yy, ye);
    gp = gp_init('init', 'CS+FIC', nin, likelih, {gpcf1, gpcf2}, [], 'jitterSigmas', 0.01, 'X_u', Xu);   %{gpcf2}
    gp.avgE = ye; 
    gp = gp_init('set', gp, 'blocks', {'manual', xx, blockindex},'latent_method', {'EP', xx, yy, 'hyper'});
    
    
    %gradcheck(gp_pak(gp,'hyper'), @gpep_e, @gpep_g, gp, xx, yy, 'hyper') 
    
    % Find the posterior mode of hyperparameters
    opt=optimset('GradObj','on');
    opt=optimset(opt,'TolX', 1e-3);
    opt=optimset(opt,'LargeScale', 'off');
    opt=optimset(opt,'Display', 'iter');
    param = 'hyper'
    
    tic
    w0 = gp_pak(gp, param);
    mydeal = @(varargin)varargin{1:nargout};
    [w,fval,exitflag,output,grad,hessian] = fminunc(@(ww) mydeal(gpep_e(ww, gp, xx, yy, param), gpep_g(ww, gp, xx, yy, param)), w0, opt);
    gp = gp_unpak(gp,w,param);
    toc

    % Make predictions
    [Ef, Varf] = ep_pred(gp, xx, yy, xx, param);

    % Plot maps
    figure(1)
    G=repmat(NaN,size(Y));
    G(xxii)=exp(Ef+Varf./2);
    pcolor(X1,X2,G),shading flat
    colormap(mapcolor(G)),colorbar
    axis equal
    title('mean relative risk, mean(\mu)')

    figure(2)
    Gp=repmat(NaN,size(Y));
    Gp(xxii)=1-normcdf(0,Ef,sqrt(Varf));
    pcolor(X1,X2,Gp),shading flat
    colormap(mapcolor(Gp, [0.5 0.5])),colorbar
    axis equal
    title('p(\mu>1)')
