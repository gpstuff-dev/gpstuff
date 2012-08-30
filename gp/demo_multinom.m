%DEMO_MULTINOM    
%

% Copyright (c) 2010 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% Simulate data 
% x = 5*rand(20,2);
% gpcf1 = gpcf_sexp('lengthScale', 2, 'magnSigma2', 2);
% gp = gp_set('lik', lik_gaussian, 'cf', gpcf1, 'jitterSigma2', 1e-4);
% K = gp_trcov(gp,x);
% 
% L = chol(K)';
% f = L*randn(length(x), 3);
% expf = exp(f);
% 
% p = expf./ repmat(sum(expf,2),1,size(expf,2));
% n = floor(1+ 100*rand(size(p,1),1));
% y = mnrnd(n,p);

% Simulate data
[x1,x2] = meshgrid(1:0.5:10,1:0.5:10);
xt = [x1(:), x2(:)];

gpcf1 = gpcf_sexp('lengthScale', 2, 'magnSigma2', 2);
gp = gp_set('lik', lik_gaussian, 'cf', gpcf1, 'jitterSigma2', 1e-4);
K = gp_trcov(gp,xt);

L = chol(K)';
f = L*randn(length(xt), 3);
expf = exp(f);

p = expf./ repmat(sum(expf,2),1,size(expf,2));
n = floor(1+ 1000*rand(size(p,1),1));
yt = mnrnd(n,p);

I = floor((size(xt,1)-1)*rand(80,1)+1);
I = unique(I,'rows');
y = yt(I,:);
x = xt(I,:);

% x=xt;
% y=yt;

% Plot the real surface
figure(1), set(gcf, 'color', 'w'), hold on
subplot(2,3,1);contour(x1, x2, reshape(f(:,1),size(x1)),'r', 'linewidth', 2)
subplot(2,3,2);contour(x1, x2, reshape(f(:,2),size(x1)),'b', 'linewidth', 2)
subplot(2,3,3);contour(x1, x2, reshape(f(:,3),size(x1)),'k', 'linewidth', 2)

figure(2), set(gcf, 'color', 'w'), hold on, mc1=mapcolor(f(:,1));mc2=mapcolor(f(:,2));mc3=mapcolor(f(:,3));
subplot(2,3,1);pcolor(x1, x2, reshape(f(:,1),size(x1))),shading flat,colormap(mc1),colorbar
subplot(2,3,2);pcolor(x1, x2, reshape(f(:,2),size(x1))),shading flat,colormap(mc2),colorbar
subplot(2,3,3);pcolor(x1, x2, reshape(f(:,3),size(x1))),shading flat,colormap(mc3),colorbar

% Do the data analysis
% ====================

% Create the model to infer the above data
lik = lik_multinom;

gpcf1 = gpcf_sexp('lengthScale', 2, 'magnSigma2', 2);
gp = gp_set('lik', lik_multinom, 'cf', gpcf1, 'jitterSigma2', 1e-4);

%opt=optimset('TolFun',1e-4,'TolX',1e-4,'Display','iter','MaxIter',100,'Derivativecheck','on');
opt=optimset('TolFun',1e-4,'TolX',1e-4,'Display','iter','MaxIter',100);

% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'opt',opt,'optimf',@fminunc);

% make the prediction for test points
[Eft] = gp_pred(gp, x, y, xt, 'yt', ones(size(yt)));

figure(1)
subplot(2,3,4);contour(x1, x2, reshape(Eft(:,1),size(x1)),'r', 'linewidth', 2)
hold on; plot(x(:,1),x(:,2),'.')
subplot(2,3,5);contour(x1, x2, reshape(Eft(:,2),size(x1)),'b', 'linewidth', 2)
hold on; plot(x(:,1),x(:,2),'.')
subplot(2,3,6);contour(x1, x2, reshape(Eft(:,3),size(x1)),'k', 'linewidth', 2)
hold on; plot(x(:,1),x(:,2),'.')

figure(2)
subplot(2,3,4);pcolor(x1, x2, reshape(Eft(:,1),size(x1))),shading flat,colormap(mc1),colorbar
hold on; plot(x(:,1),x(:,2),'k.')
subplot(2,3,5);pcolor(x1, x2, reshape(Eft(:,2),size(x1))),shading flat,colormap(mc2),colorbar
hold on; plot(x(:,1),x(:,2),'k.')
subplot(2,3,6);pcolor(x1, x2, reshape(Eft(:,3),size(x1))),shading flat,colormap(mc3),colorbar
hold on; plot(x(:,1),x(:,2),'k.')


% Predict also observations and plot the predictions with variance
% -------------------------------------------------

% % HUOM! Eyt ja Varyt predictiot on huonosti toteutettu. Koodaa z:lla, joka
% % kuvaa kokonaism��r��. pyt on pieless�
[Eft, Varft, lpyt, Eyt, Varyt] = gp_pred(gp, x, y, xt, 'yt', yt);
% 
% figure
% variance = [squeeze(Varft(1,1,:)) ; squeeze(Varft(3,3,:)) ; squeeze(Varft(3,3,:))];
% plot(Eft(:), 'k')
% hold on
% plot(f(:), 'r')
% plot(Eft(:)+2.*variance, 'b--')
% plot(Eft(:)-2.*variance, 'b--')
% 
% variance = [squeeze(Varyt(1,1,:)) ; squeeze(Varyt(2,2,:)) ; squeeze(Varyt(3,3,:))];
% figure
% plot(Eyt(:), 'k')
% hold on
% plot(y(:), 'r*')
% plot(Eyt(:)+2.*variance, 'b--')
% plot(Eyt(:)-2.*variance, 'b--')


pyt2=exp(Eft)./(sum(exp(Eft),2)*ones(1,3));

% Plot the relative abundances

figure, set(gcf, 'color', 'w'), hold on, mc1=mapcolor(p(:,1));mc2=mapcolor(p(:,2));mc3=mapcolor(p(:,3));
subplot(2,3,1);pcolor(x1, x2, reshape(p(:,1),size(x1))),shading flat,colormap(mc1),colorbar
title('Real relative abundance 1')
subplot(2,3,2);pcolor(x1, x2, reshape(p(:,2),size(x1))),shading flat,colormap(mc2),colorbar
title('Real relative abundance 2')
subplot(2,3,3);pcolor(x1, x2, reshape(p(:,3),size(x1))),shading flat,colormap(mc3),colorbar
title('Real relative abundance 3')

subplot(2,3,4);pcolor(x1, x2, reshape(pyt2(:,1),size(x1))),shading flat,colormap(mc1),colorbar
hold on; plot(x(:,1),x(:,2),'k.'), title('Model prediction 1')
subplot(2,3,5);pcolor(x1, x2, reshape(pyt2(:,2),size(x1))),shading flat,colormap(mc2),colorbar
hold on; plot(x(:,1),x(:,2),'k.'), title('Model prediction 2')
subplot(2,3,6);pcolor(x1, x2, reshape(pyt2(:,3),size(x1))),shading flat,colormap(mc3),colorbar
hold on; plot(x(:,1),x(:,2),'k.'), title('Model prediction 3')

figure
subplot(1,3,1);pcolor(x1, x2, reshape(squeeze(Varyt(1,1,:)),size(x1))),shading flat,colormap(mc1),colorbar
hold on; plot(x(:,1),x(:,2),'k.')
subplot(1,3,2);pcolor(x1, x2, reshape(squeeze(Varyt(2,2,:)),size(x1))),shading flat,colormap(mc2),colorbar
hold on; plot(x(:,1),x(:,2),'k.')
subplot(1,3,3);pcolor(x1, x2, reshape(squeeze(Varyt(3,3,:)),size(x1))),shading flat,colormap(mc3),colorbar
hold on; plot(x(:,1),x(:,2),'k.')
