function demo_nb_FIC

    
% First load the data
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
    ye = max(ye,1e-3);

    %=======================================================================
% $$$     xx = xx(1:400,:);
% $$$     yy = yy(1:400,:);
% $$$     ye = ye(1:400,:);

    bls = 3; indtype = 'corners';
    [blockindex, Xu] = set_PIC(xx, dims, bls, indtype, 1);
    
    [n, nin] = size(xx);

    %gpcf1 = gpcf_sexp('init', 'lengthScale', 2, 'magnSigma2', 0.01);
    %gpcf1 = gpcf_exp('init', 'lengthScale', 2, 'magnSigma2', 0.01);
    gpcf1 = gpcf_matern32('init', 'lengthScale', 2, 'magnSigma2', 0.01);
    %gpcf1 = gpcf_matern52('init', 'lengthScale', 2, 'magnSigma2', 0.01);

    
    gpcf1.p.lengthScale = t_p({1 4});
    gpcf1.p.magnSigma2 = t_p({0.3 4});

    likelih = likelih_negbin('init', yy, ye, 10);
    %    likelih.p.disper = gamma_p({1.5, 0.005});
    
    
    %==========================
    % MCMC 
    %==========================
    
    gp = gp_init('init', 'FIC', likelih, {gpcf1}, []);   %{gpcf2}
    gp = gp_init('set', gp, 'X_u', Xu);
    gp = gp_init('set', gp, 'latent_method', {'MCMC', zeros(size(yy))'});
    
    gradcheck(gp_pak(gp,'hyper'), @gp_e, @gp_g, gp, xx, yy, 'hyper')
    
    % Set the parameters 
    opt=gp_mcopt;
    opt.nsamples=1;
    opt.repeat=1;

    % HMC-hyper
    %opt.hmc_opt.steps=3;
    opt.hmc_opt.steps=1;
    opt.hmc_opt.stepadj=0.01;
    opt.hmc_opt.nsamples=1;
    opt.hmc_opt.persistence=0;
    opt.hmc_opt.decay=0.8;

    % HMC-latent
    opt.latent_opt.nsamples=1;
    opt.latent_opt.nomit=0;
    opt.latent_opt.persistence=0;
    %opt.latent_opt.repeat=20;
    %opt.latent_opt.steps=20;
    opt.latent_opt.repeat=1;
    opt.latent_opt.steps=6;
    opt.latent_opt.stepadj=0.15;
    opt.latent_opt.window=5;
    
    opt.nb_sls_opt = sls_opt;
    opt.nb_sls_opt.maxiter = 400;
    opt.nb_sls_opt.mmlimits = [0;1000];
    opt.nb_sls_opt.nsamples = 20;
    opt.nb_sls_opt.method = 'minmax';
    opt.nb_sls_opt.display = 0;
    
    [rgp, gp, opt]=gp_mc(opt, gp, xx, yy);

    opt.latent_opt.repeat=1;
    opt.latent_opt.steps=3;
    opt.latent_opt.window=1;
    opt.hmc_opt.persistence=0;
    opt.hmc_opt.stepadj=0.005;
    opt.hmc_opt.steps=1;

    opt.display = 1;
    opt.hmc_opt.display = 0;
    opt.latent_opt.display=0;

    opt.repeat = 1;
    opt.hmc_opt.nsamples=1;
    opt.nsamples=1;
    while length(rgp.edata)<2000 %   1000
        [rgp,gp,opt]=gp_mc(opt, gp, xx, yy, [], [], rgp);
        fprintf('  hmcrejects=%.3f   \n',mean(rgp.hmcrejects))
        fprintf('dispersion=%.4f\n',gp.likelih.disper)
        fprintf('length1=%.4f, magnitude1=%.4f\n',gp.cf{1}.lengthScale, sqrt(gp.cf{1}.magnSigma2))
        subplot(2,2,1)
        plot(rgp.cf{1}.lengthScale,sqrt(rgp.cf{1}.magnSigma2) ,rgp.cf{1}.lengthScale(end),sqrt(rgp.cf{1}.magnSigma2(end)),'r*')
        subplot(2,2,3)
        plot(1:length(rgp.edata),rgp.likelih.disper, length(rgp.edata),rgp.likelih.disper(end),'r*')
        
        subplot(2,2,[2 4])
        G=repmat(NaN,size(Y));
        G(xxii)=exp(rgp.latentValues(end,:));
        pcolor(X1,X2,G),shading flat
        colormap(mapcolor(G)),colorbar
        %colorbar
        drawnow
    end




    %==========================================
    % EP optimization
    %==========================================
    gp = gp_init('init', 'FIC', likelih, {gpcf1}, [], 'X_u', Xu);   %{gpcf2}
    gp = gp_init('set', gp, 'latent_method', {'EP', xx, yy, 'hyper'});

    gradcheck(gp_pak(gp,'hyper'), @gpep_e, @gpep_g, gp, xx, yy, 'hyper')

    
    gp = gp_init('init', 'FIC', likelih, {gpcf1}, [], 'X_u', Xu);   %{gpcf2}
    gp = gp_init('set', gp, 'latent_method', {'EP', xx, yy, 'hyper+likelih'});
        
    opt=optimset('GradObj','on');
    opt=optimset(opt,'TolX', 1e-3);
    opt=optimset(opt,'LargeScale', 'off');
    opt=optimset(opt,'Display', 'iter');
    param = 'hyper+likelih';

    gpep_g(gp_pak(gp,'hyper+likelih'), gp, xx, yy, 'hyper+likelih')    
    gradcheck(gp_pak(gp,'hyper+likelih'), @gpep_e, @gpep_g, gp, xx, yy, 'hyper+likelih')

    w0 = gp_pak(gp, param);
    mydeal = @(varargin)varargin{1:nargout};
    w = fminunc(@(ww) mydeal(gpep_e(ww, gp, xx, yy, param), gpep_g(ww, gp, xx, yy, param)), w0, opt);
    gp = gp_unpak(gp,w,param);
    

    
    
    y = yy;
    z = randn(size(y));
    w = 10;
    fe = @(w, likelih) (- feval(likelih.fh_e, feval(likelih.fh_unpak, w, likelih), y, z));
    fg = @(w, likelih) (- feval(likelih.fh_g, feval(likelih.fh_unpak, w, likelih), y, z, 'hyper'));

    gradcheck(w, fe, fg, gp.likelih)






















% $$$ opt.repeat = 10;
% $$$ opt.hmc_opt.nsamples=1;
% $$$ alku = length(rgp.edata);
% $$$ for i=alku:1000 %   1000
% $$$     [rgp,gp,opt]=gp_mc(opt, gp, xx, yy, [], [], rgp);
% $$$     save(name,'rgp', 'gp' ,'opt', 'xx', 'yy', 'ye', 'Y', 'dims', 'xxii');
% $$$     err = lasterror;
% $$$     if ~isempty(err.message)
% $$$         save(['error_' name], 'err')
% $$$     end
% $$$ end

    % rgp.tr_index = blockindex;
    save(name,'rgp', 'gp' ,'opt', 'xx', 'yy', 'ye', 'Y', 'dims', 'xxii');


    rgp = rmfield(rgp, 'tr_index')

    rt=thin(rgp,400);
    %rt =rgp;
    plot(log(rt.cf{1}.lengthScale))
    plot(rt.cf{1}.lengthScale)
    plot(sqrt(rt.cf{1}.magnSigma2))
    plot(log(rt.cf{1}.lengthScale),log(rt.cf{1}.magnSigma2))
    plot(rt.cf{1}.lengthScale,rt.cf{1}.magnSigma2)

    % Convergence testing
    mean(rgp.hmcrejects)
    rt=thin(rgp, 100)
    geyer_imse(rt.cf{1}.lengthScale)
    geyer_imse(rt.cf{1}.magnSigma2)

    %rt=thin(rgp, 300,70);
    subplot(1,2,1)
    G=repmat(NaN,size(Y));
    G(xxii)=mean(exp(rt.Ef));
    pcolor(X1,X2,G),shading flat
    colormap(mapcolor(G)),colorbar
    title('median relative risk, median(\mu)')

    subplot(1,2,2)
    Gp=repmat(NaN,size(Y));
    %Gp(ii)=sum(rt.latentValues>0)/size(rt.latentValues,1);
    Gp(xxii)=sum(exp(rt.latentValues)>1)/size(rt.latentValues,1);
    pcolor(X1,X2,Gp),shading flat
    %set(gca,'ydir','reverse')
    colormap(mapcolor(G)),colorbar
    %colormap(mapcolor2(Gp,[.05 .1 .2 .8 .9 .95])),colorbar
    drawnow
    title('probability p(\mu>1)')


    % T�st� alasp�in kuvat suuremmalla resoluutiolla. 
    % Aja joko alla oleva for-silmukka tai lataa apu_spatial1r2, jonne on
    % tallennettu for-silmukan tulokset.
    [X1t,X2t]=meshgrid(.5:.5:35,.5:.5:60);
    for i1=1:35
        for i2=1:60
            if isnan(EA(i2,i1))
                qii=find(X1t>i1-1&X1t<=i1&X2t>i2-1&X2t<=i2);
                X1t(qii)=NaN;
                X2t(qii)=NaN;
            end
        end
    end
    tii=~isnan(X1t);
    xxt=[X1t(tii) X2t(tii)];
    [X1t,X2t]=meshgrid(.5:.5:35,.5:.5:60);

    %rt=thin(rgp,150,15);
    gts=zeros(size(xxt,1),numel(rt.cf{1}.magnSigma2));
    pts=zeros(size(xxt,1),numel(rt.cf{1}.magnSigma2));
    for i1=1:numel(rt.cf{1}.magnSigma2)
        rti=thin(rt,i1-1,1,i1);
        rti.tr_index = gp.tr_index;
        rti.X_u = reshape(rti.X_u, length(rti.X_u)/2,2);
        [gts(:,i1)] = gp_fwd(rti,xx,rti.latentValues',xxt, gp.tr_index);
% $$$   s = sqrt(s);
% $$$   pts(:,i1)=normcdf(zeros(size(s)),gts(:,i1),s);
    end
    %save apu_spatial1r2 rgp gp opt tcpu gts pts


    gt=mean(gts,2);
    pt=mean(pts,2);
    Gt=repmat(NaN,size(X1t));
    Pt=repmat(NaN,size(X1t));
    Gt(tii)=exp(gt);
    Pt(tii)=pt;
    Gt(Pt>0.05&Pt<0.95)=1;

    title('Full Gaussian process with data915_aivoveri.mat')
    hold on
    subplot(1,3,3)
    pcolor(X1t,X2t,Gt),shading flat,set(gca,'YDir','reverse')
    colormap(mapcolor(Gt,[max(Gt(Gt<1)) min(Gt(Gt>1))]))
    colormap(mapcolor(Gt))
    colorbar
    title('Gt')
end

