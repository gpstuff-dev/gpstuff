
% $$$    load /proj/finnwell/spatial/data/cancer2007/testdata/melanoma_female_10years.mat
% $$$    load /proj/finnwell/spatial/data/cancer2007/testdata/cervix_female_10years.mat
   load /proj/finnwell/spatial/data/cancer2007/testdata/lungcancer_female_10years.mat

   name = 'lungcancer_10yrs_FIC_3exp.mat';

   
   [n, nin] = size(xx);

   % Create covariance functions
   gpcf1 = gpcf_st_exp('init',2,1, 'lengthScale', [200 26], 'magnSigma2', 0.04);
   gpcf1.p.lengthScale = t_p({1 4});
   gpcf1.p.magnSigma2 = t_p({0.3 4});  

   gpcf2 = gpcf_t_exp('init',3, 'lengthScale', [150], 'magnSigma2', 0.11);
   gpcf2.p.lengthScale = t_p({1 4});
   gpcf2.p.magnSigma2 = t_p({0.3 4});     
   
   gpcf3 = gpcf_s_exp('init',3, 'lengthScale', [150], 'magnSigma2', 0.11);
   gpcf3.p.lengthScale = t_p({1 4});
   gpcf3.p.magnSigma2 = t_p({0.3 4});     

   % Initialize GP
   gp = gp_init('init', 'FIC', nin, 'poisson', {gpcf1,gpcf2,gpcf3}, [], 'jitterSigmas', 0.01);   %{gpcf2}
   gp.avgE = ye; 
   gp = gp_init('set', gp, 'latent_method', {'MCMC', @latent_hmcr, zeros(size(yy))'});
   
   % Meshgrid for inducing points
   [u1,u2]=meshgrid(-10:70:650,0:90:1300);

   % Create the (approximate) convex full of Finland from
   % the coordinates of municipals.
   KH = convhull(xx(1:431,1),xx(1:431,2));
   khx = xx(KH,1);
   khy = xx(KH,2);
   % Calculate the center of KH.
   cx = mean(khx);
   cy = mean(khy);
   % Increase the size of KH a bit.
   gain = 70;
   for i = 1:length(khx)
       len = sqrt((khx(i)-cx)^2+(khy(i)-cy)^2);
       khx(i) = khx(i) + gain/len*(khx(i)-cx);
       khy(i) = khy(i) + gain/len*(khy(i)-cy);
   end

   % Include the points which are inside the convex hull
   IN = inpolygon(u1,u2,khx,khy);
   u1 = u1(IN);
   u2 = u2(IN);
   
   t_u = [1957.5-5:10:1997.5+5];
   U = zeros(length(t_u)*length(u1(:)),3);   
   for i = 1:length(t_u);
       U((i-1)*length(u1(:))+1:i*length(u1(:)),:)=[u1(:) u2(:) t_u(i)*ones(length(u1(:)),1)];    
   end

   % plot the inducing inputs and data points
   plot3(U(:,1), U(:,2), U(:,3), 'kX', 'MarkerSize', 12, 'LineWidth', 2)
   hold on
   plot3(xx(:,1), xx(:,2), xx(:,3), 'ro', 'MarkerSize', 12);
   
   gp = gp_init('set', gp, 'X_u', U);
   
   gradcheck(gp_pak(gp,'hyper'), @gp_e, @gp_g, gp, xx, yy, 'hyper')
   
   % Optimize the starting values with SCG2
   %s_opt = scg2_opt;
   %w = gp_pak(gp,'hyper');
   %w = scg2(@gp_e,w,s_opt,@gp_g, gp, xx, yy, 'hyper')
   
   % Set the parameters 
   opt=gp_mcopt;
   opt.nsamples=1;
   opt.repeat=1;

   % HMC-hyper
   %opt.hmc_opt.steps=3;
   opt.hmc_opt.steps=4;
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
   opt.latent_opt.steps=4;
   opt.latent_opt.stepadj=0.10;
   opt.latent_opt.window=5;
   
   [rgp,gp, opt]=gp_mc(opt, gp, xx, yy);

   opt.latent_opt.repeat=1;
   opt.latent_opt.steps=5;
   opt.latent_opt.window=1;
   opt.hmc_opt.persistence=0;
   opt.hmc_opt.stepadj=0.015;
   opt.hmc_opt.steps=5;

   opt.display = 1;
   opt.hmc_opt.display = 0;
   opt.latent_opt.display=0;

   opt.repeat = 2;
   opt.hmc_opt.nsamples=1;
   opt.nsamples=1;
   while length(rgp.edata)<10000 %   1000
       [rgp,gp,opt]=gp_mc(opt, gp, xx, yy, [], [], rgp);
       %gradcheck(gp_pak(gp,'hyper'), @gp_e, @gp_g, gp, xx, yy, 'hyper')
       fprintf('  hmcrejects=%.3f   \n',mean(rgp.hmcrejects))
       %fprintf('length1=%.4f, magnitude1=%.4f\n',gp.cf{1}.lengthScale(1), sqrt(gp.cf{1}.magnSigma2))

       fprintf('length1=%.4f length2=%.4f, magnitude1=%.4f\n',gp.cf{1}.lengthScale(1), gp.cf{1}.lengthScale(2), sqrt(gp.cf{1}.magnSigma2))
       fprintf('length1=%.4f, magnitude1=%.4f\n',gp.cf{2}.lengthScale(1), sqrt(gp.cf{2}.magnSigma2))
       fprintf('length1=%.4f, magnitude1=%.4f\n',gp.cf{3}.lengthScale(1), sqrt(gp.cf{3}.magnSigma2))

       % Plot the hyperparameters
       nsamp = length(rgp.hmcrejects);
       subplot(7,1,1)
       plot(1:nsamp,rgp.cf{1}.lengthScale(:,1));
       subplot(7,1,2)
       plot(1:nsamp,rgp.cf{1}.lengthScale(:,2));
       subplot(7,1,3)
       plot(1:nsamp,rgp.cf{1}.magnSigma2);
       subplot(7,1,4)
       plot(1:nsamp,rgp.cf{2}.lengthScale(:,1));
       subplot(7,1,5)
       plot(1:nsamp,rgp.cf{2}.magnSigma2);
       subplot(7,1,6)
       plot(1:nsamp,rgp.cf{3}.lengthScale(:,1));
       subplot(7,1,7)
       plot(1:nsamp,rgp.cf{3}.magnSigma2);

       drawnow
   end

   % Save results
   save(name,'rgp', 'gp' ,'opt', 'xx', 'yy', 'ye');


% $$$ 
% $$$    rt=thin(rgp,200,5);
% $$$    rt = rgp;
% $$$    %rt =rgp;
% $$$    plot(log(rt.cf{1}.lengthScale))
% $$$    plot(rt.cf{1}.lengthScale)
% $$$    plot(sqrt(rt.cf{1}.magnSigma2))
% $$$    plot(log(rt.cf{1}.lengthScale),log(rt.cf{1}.magnSigma2))
% $$$    plot(rt.cf{1}.lengthScale,rt.cf{1}.magnSigma2)
% $$$ 
% $$$    % Convergence testing
% $$$    mean(rgp.hmcrejects)
% $$$    rt=thin(rgp, 400)
   geyer_imse(rt.cf{1}.lengthScale)
   geyer_imse(rt.cf{1}.magnSigma2)

   plot_to_grid = 0;
   
   % Predict into a grid
   if plot_to_grid               
       % Create the prediction grid
       minXY = min(xx);
       maxXY = max(xx)+100;
       step = 5;
       [X1t,X2t]=meshgrid(minXY(1):step:maxXY(1),minXY(2):step:maxXY(2));
       IN_all = zeros(size(X1t));   
       
       xx_grid = [];
       
       load 'kunta_polygonit.mat'
       for i1 = 1:length(municipals)
           polygons = municipals{i1};
           for i2 = 1:length(polygons)
               IN = inpolygon(X1t,X2t,polygons{i2}(:,1),polygons{i2}(:,2));
               IN_all(IN) = 1;
           xx_grid = [xx_grid; X1t(IN) X2t(IN)];
           end
       end
       
       X1t(IN_all == 0) = NaN; 
       X2t(IN_all == 0) = NaN; 
       tii = find(IN_all);
       
       xx_grid = unique(xx_grid,'rows');
       xx_grid(:,1) = xx_grid(:,1) - 10;
       plot(xx_grid(:,1),xx_grid(:,2),'.b',xx(:,1),xx(:,2),'.k')
   % Predict into original municipal cooordinates
   else                     
       xx_grid = [xx(1:431,1) xx(1:431,2)];
   end
   
   % Make the predictions
   predict_years = [1955:2004]';
   gtt = zeros(length(predict_years), size(xx_grid,1));
   % Calculate the predictions
   for i = 1:length(predict_years)
       xxt = [xx_grid predict_years(i)*ones(size(xx_grid,1),1)];
       
       gts=zeros(size(xxt,1),numel(rt.cf{1}.magnSigma2));
       pts=zeros(size(xxt,1),numel(rt.cf{1}.magnSigma2));
       for i1=1:numel(rt.cf{1}.magnSigma2)
           rti=thin(rt,i1-1,1,i1);
           rti.X_u = reshape(rti.X_u, length(rti.X_u)/3,3);
           %[gts(:,i1)] = gp_fwdc(rti,xx,rti.latentValues',xxt,components);
           [gts(:,i1)] = gp_fwds(rti,xx,rti.latentValues',xxt);
       end
       
       gtt(i,:) = mean(gts,2);
       i
   end
   
   % Save results
   save('lungcancer_10yrs_exp_predm.mat','gtt','xxt')
   load lungcancer_10yrs_exp_predm.mat
     
   % Make animations for grid
   if plot_to_grid
       for i = 1:length(test_years)   
           pt=mean(pts,2);
           Gt=repmat(NaN,size(X1t));
           Pt=repmat(NaN,size(X1t));
           Gt(tii)=exp(gtt(i,:));
           Pt(tii)=pt;
           Gt(Pt>0.05&Pt<0.95)=1;
           
           pcolor(X1t,X2t,Gt),shading flat
           %,set(gca,'YDir','reverse')
           %colormap(mapcolor(Gt,[max(Gt(Gt<1)) min(Gt(Gt>1))]))
           %colormap(mapcolor(Gt))
           %colormap(mapcolor2(Gt,[0.64 0.7 0.78 1.157 1.28 1.44]))
           caxis([0.3 2.1])
           hold on
           plot(xx(:,1),xx(:,2),'k.')
           colorbar
           title(sprintf('Year %d',i));
           F(i) = getframe;
           hold off
       end
   % Plot to municipals
   else
       % Make KML animation
       addpath /proj/finnwell/spatial/data/cancer2007/gearth
       colors = [min(min((exp(gtt))))-0.01 max(max((exp(gtt))))+0.01];
       caxis(colors);
       cmap = colormap;
       filename = '/proj/finnwell2/jmjharti/kml/lungcancer_3exp_s.kml';
       ge_munc_anim(exp(gtt),test_years, cmap, colors, filename);
       

       % Make animations in figures for municipals
       addpath /proj/finnwell/spatial/data/cancer2007/matlab
       scrsz = get(0,'ScreenSize');
       figure('Position',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)])
       set(gcf,'PaperPositionMode','auto')
       set(gcf,'Position',[0 0 360 466])
       axis off
       for i = 1:size(gtt,1)
           hold on
           draw_muncs(exp(gtt(i,:))',colors);
           title(sprintf('Relative risk on female lungcancer, year %d',i+1950));
           pause
% $$$        if i < 10
% $$$            filename = sprintf('lungcancer/00%d.jpg',i);
% $$$        else
% $$$            filename = sprintf('lungcancer/0%d.jpg',i);
% $$$        end
% $$$        print('-djpeg100', filename);
% $$$        %F(i) = getframe;
% $$$        hold off
       end
   end
   
   
 
   %title('Gt')
   %pause

end