function demo_ep_spatialPIC_mcmc
%   Author: Jarno Vanhatalo <jarno.vanhatalo@tkk.fi>
%   Last modified: 2007-08-23 09:45:01 EEST

% $$$ addpath /proj/finnwell/spatial/testdata
% $$$ addpath /proj/finnwell/spatial/jpvanhat/model_comp

    
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
    ye = max(ye,1e-3);
    %=======================================================================

    [blockindex, Xu] = set_PIC(xx, dims, cellsize, 4, 'corners', 1);
    
    [n, nin] = size(xx);

    gpcf1 = gpcf_exp('init', nin, 'lengthScale', 7, 'magnSigma2', 0.5);
    gpcf1.p.lengthScale = t_p({1 4});
    gpcf1.p.magnSigma2 = t_p({0.3 4});

    gp = gp_init('init', 'PIC_BLOCK', nin, 'poisson', {gpcf1}, [], 'X_u', Xu, 'blocks', {'manual', xx, blockindex});   %{gpcf2} , 'jitterSigmas', 0.01
    gp.avgE = ye; 
    tic
        gp = gp_init('set', gp, 'latent_method', {'EP', xx, yy, 'hyper'});
    toc
    gradcheck(gp_pak(gp,'hyper'), @gpep_e, @gpep_g, gp, xx, yy, 'hyper')

    for i=1:length(ind); 
        %        [min(diag(La2{i})) max(diag(La2{i}))]
        [min(diag(D{i})) max(diag(D{i}))]
    end
    
    
    % Find the mode by optimization
    %===============================
    w0 = gp_pak(gp,'hyper');
    % Uncomment desired lines
    % no gradients provided (can't use LargeSacle)
    %opt=optimset('LargeScale','off');
    % gradients provided
    opt=optimset('GradObj','on');
    opt=optimset(opt,'TolX', 1e-3);
    opt=optimset(opt,'Display', 'iter');
    % Hessian provided
% $$$     opt=optimset('GradObj','on','Hessian','off');
    % if gradients provided and you want to check your gradients
    % DerivativeCheck is not allowed with LargeScale
    %opt=optimset(opt,'LargeScale','off','DerivativeCheck','on');
    % optimize and get also Hessian H
% $$$     thefunction = @(ww) {gpep_e(ww, gp, xx, yy, 'hyper') gpep_g(ww, gp, xx, yy, 'hyper')}
    [w,fval,exitflag,output,g,H]=fminunc(@(ww) energy_grad(ww, gp, xx, yy, 'hyper'), w0, opt); 
    save EP_PIC_20
    %% If using LargeScale without Hessian given, Hessian computed is sparse 
    H=full(H);
    S=inv(H);
    exp(w)

    gp = gp_unpak(gp,w,'hyper');
    [Ef, Varf] = ep_pred(gp, xx, yy, xx, blockindex);

    % Plot the maps and the Normal approximation of the 
    % hyperparameter posterior
    figure(1)
    G=repmat(NaN,size(Y));
    G(xxii)=exp(Ef);
    pcolor(X1,X2,G),shading flat
    colormap(mapcolor(G)),colorbar
    axis equal
    axis([0 35 0 60])
    drawnow
    title('EP approximated median/mean relative risk')
    
    figure(2)
    Gp=repmat(NaN,size(Y));
    Gp(xxii)=1-normcdf(0,Ef,sqrt(Varf));
    pcolor(X1,X2,Gp),shading flat
    colormap(mapcolor(Gp)),colorbar
    axis equal
    axis([0 35 0 60])
    drawnow
    title('EP approximated probability p(\mu>1)')

    figure(3)
    [m, l] = meshgrid(linspace(1, 2.2 ,40),linspace(-1.4, 1.4, 40));
    loghyper = [m(:) l(:)];
    const = - log(2*pi) -0.5*log(det(S));
    for i=1:length(loghyper)
        loghyp_post(i) = exp( const - 0.5*(loghyper(i,:) - w)*(S\(loghyper(i,:) - w)')).*prod(1./exp(loghyper(i,:)));
    end
    contour(m, l, reshape(loghyp_post,40,40))

    log_l = linspace(-1.4,1.4,40);
    lmarg_post = norm_pdf(log_l,w(2),sqrt(S(2,2)))./exp(log_l);    
    plot(exp(log_l), lmarg_post)
    
    log_m = linspace(1,2.2,40);
    lmarg_post = norm_pdf(log_m,w(1),sqrt(S(1,1)))./exp(log_m);
    plot(exp(log_m), lmarg_post)

    %    mapcolor(G, [0.975 1.025])
    
% $$$     gp = gp_init('init', 'FULL', nin, 'poisson', {gpcf1}, [], 'jitterSigmas', 0.01);   %{gpcf2}
% $$$     gp.avgE = ye; 
% $$$     tic
% $$$     gp = gp_init('set', gp, 'latent_method', {'EP', xx, yy, 'hyper'});
% $$$     toc
    
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

    opt.fh_e = @gpep_e;
    opt.fh_g = @gpep_g;
    [rgp,gp]=gp_mc(opt, gp, xx, yy);

    opt.hmc_opt.persistence=0;
    opt.hmc_opt.stepadj=0.01;
    opt.hmc_opt.steps=1;

    opt.display = 1;
    opt.hmc_opt.display = 0;

    opt.repeat = 1;
    opt.hmc_opt.nsamples=1;
    opt.nsamples=1;
    while length(rgp.edata)<200 %   1000
        [rgp,gp,opt]=gp_mc(opt, gp, xx, yy, [], [], rgp);
        fprintf('  hmcrejects=%.3f   \n',mean(rgp.hmcrejects))
        fprintf('length1=%.4f, magnitude1=%.4f\n',gp.cf{1}.lengthScale, sqrt(gp.cf{1}.magnSigma2)) 
        subplot(2,2,1)
        plot(rgp.cf{1}.lengthScale,sqrt(rgp.cf{1}.magnSigma2) ,rgp.cf{1}.lengthScale(end),sqrt(rgp.cf{1}.magnSigma2(end)),'r*')
        subplot(2,2,[2 4])
        G=repmat(NaN,size(Y));
        G(xxii)=exp(rgp.Ef(end,:));
        pcolor(X1,X2,G),shading flat
        colormap(mapcolor(G)),colorbar
        drawnow
    end

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


    % Tästä alaspäin kuvat suuremmalla resoluutiolla. 
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











    
% $$$     subplot(3,1,1); plot([lambdaconf(1):0.01:lambdaconf(2)], feval(zm,[lambdaconf(1):0.01:lambdaconf(2)]))
% $$$     subplot(3,1,2); plot([lambdaconf1(1):0.01:lambdaconf1(2)], feval(fm,[lambdaconf1(1):0.01:lambdaconf1(2)]))
% $$$     subplot(3,1,3); plot([lambdaconf2(1):0.01:lambdaconf2(2)], feval(sm,[lambdaconf2(1):0.01:lambdaconf2(2)]))
% $$$ 
% $$$     f = [-5:0.01:5];
% $$$     figure(1)
% $$$     subplot(2,1,1)
% $$$     plot(f, poiss_pdf(repmat(0,1,size(f,2)), exp(f).*gp.avgE(i1)))
% $$$     subplot(2,1,2)
% $$$     plot(f, log(poiss_pdf(repmat(0,1,size(f,2)), exp(f).*gp.avgE(i1))))
% $$$     figure(2)
% $$$     subplot(2,1,1)
% $$$     plot(f, poiss_pdf(repmat(1,1,size(f,2)), exp(f).*gp.avgE(i1)))
% $$$     subplot(2,1,2)
% $$$     plot(f, log(poiss_pdf(repmat(1,1,size(f,2)), exp(f).*gp.avgE(i1))))
% $$$     figure(3)
% $$$     subplot(2,1,1)
% $$$     plot(f, poiss_pdf(repmat(5,1,size(f,2)), exp(f).*gp.avgE(i1)))
% $$$     subplot(2,1,2)
% $$$     plot(f, log(poiss_pdf(repmat(5,1,size(f,2)), exp(f).*gp.avgE(i1))))
