function demo_ep_clPIC_opt
%DEMO_GP2CLASS    Classification problem demonstration for 2
%                 classes. 
%
%      Description
%      The demonstration program is based on synthetic two 
%      class data used by B.D. Ripley (Pattern Regocnition and
%      Neural Networks, 1996}. The data consists of 2-dimensional
%      vectors that are divided into to classes, labeled 0 or 1.
%      Each class has a bimodal distribution generated from equal
%      mixtures of Gaussian distributions with identical covariance
%      matrices. A Bayesian aprouch is used to find the decision
%      line and predict the classes of new data points.
%
%      The demonstration program does not sample for real, because
%      it would require so much time. The main sampling state is
%      commented out from the program and instead a saved network
%      structure is loaded and used to make predictions (see lines
%      143-146).
%

% Copyright (c) 2005 Jarno Vanhatalo, Aki Vehtari 

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% This demonstration is based on the dataset used in the book Pattern Recognition and
% Neural Networks by B.D. Ripley (1996) Cambridge University Press ISBN 0 521
% 46986 7

disp(' ')
disp(' The demonstration program is based on synthetic two ')
disp(' class data used by B.D. Ripley (Pattern Regocnition and')
disp(' Neural Networks, 1996}. The data consists of 2-dimensional')
disp(' vectors that are divided into to classes, labeled 0 or 1.')
disp(' Each class has a bimodal distribution generated from equal')
disp(' mixtures of Gaussian distributions with identical covariance')
disp(' matrices. A Gaussian process is used to find the decision')
disp(' line and predict the classes of new data points.')
disp(' ')
disp(' ')

  
% Load the data
S = which('demo_classific2');
L = strrep(S,'demo_classific2.m','demos/synth.tr');
x=load(L);
y=x(:,end);

% Set the targets to {-1,1}. NOTE! Compare with logistic with which we need {0,1}.
y=y*2-1;
x(:,end)=[];
[n, nin] = size(x);

disp(' ')
disp(' First we create a Gaussian process for classification problem. ')
disp(' A Gaussian multivariate hierarchical prior with ARD is created')
disp(' for GP. ')
disp(' ')

% Create covariance functions
% Create covariance functions
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 1, 'magnSigma2', 2^2);
%gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

%gpcf1 = gpcf_exp('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_exp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

%gpcf1 = gpcf_matern32('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_matern32('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

%gpcf1 = gpcf_matern52('init', nin, 'lengthScale', 1, 'magnSigma2', 0.2^2);
%gpcf1 = gpcf_matern52('init', nin, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

% Set the prior for the parameters of covariance functions 
gpcf1.p.lengthScale = gamma_p({3 7 3 7});
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});


% Set the options for scges
fe=str2fun('gpep_e');
fg=str2fun('gpep_g');
n=length(y);
itr=1:2:n-1;     % training set of data for early stop
its=2:2:n;   % test set of data for early stop
optes=scges_opt;
optes.display=1;
optes.tolfun=1e-1;
optes.tolx=1e-1;

% Set the blocks and the inducing inputs
b1 = linspace(-1.25, 0.9, 5);
b2 = linspace(-0.2,  1.1, 5);
xx = x(itr,:);
xx2 = x(its,:);
for i1=1:4
    for i2=1:4
        ind = 1:size(xx,1);
        ind = ind(: , b1(i1)<=xx(ind',1) & xx(ind',1) < b1(i1+1));
        ind = ind(: , b2(i2)<=xx(ind',2) & xx(ind',2) < b2(i2+1));        
        ind2 = 1:size(xx2,1);
        ind2 = ind2(: , b1(i1)<=xx2(ind2',1) & xx2(ind2',1) < b1(i1+1));
        ind2 = ind2(: , b2(i2)<=xx2(ind2',2) & xx2(ind2',2) < b2(i2+1));
        index{4*(i1-1)+i2} = ind';
        index2{4*(i1-1)+i2} = ind2';
    end
end

index_tmp = {index{1:3}};
index_tmp = {index_tmp{:} index{5:7}};
index_tmp = {index_tmp{:} [index{8} ; index{9}] };
index_tmp = {index_tmp{:} index{10:16}};
index = index_tmp;

index_tmp = {index2{1:3}};
index_tmp = {index_tmp{:} index2{5:7}};
index_tmp = {index_tmp{:} [index2{8} ; index2{9}] };
index_tmp = {index_tmp{:} index2{10:16}};
index2 = index_tmp;

% Set the inducing inputs
[u1,u2]=meshgrid(linspace(-1.25, 0.9,6),linspace(-0.2, 1.1,6));
U=[u1(:) u2(:)];
U = U([3 4 7:18 20:24 26:30 33:36],:);

% plot the data points in each block with different colors and marks
figure
col = {'b*','g*','r*','c*','m*','y*','k*','b*','b.','g.','r.','c.','m.','y.','k.','b.'};
hold on
for i=1:14
    plot(xx(index{i},1),xx(index{i},2),col{i})
    plot(xx(index2{i},1),xx(index2{i},2),col{i})
end
% plot the inducing inputs
plot(U(:,1), U(:,2), 'kX', 'MarkerSize', 12, 'LineWidth', 2)
title('Blocks and inducing inputs')

% Initiliaze the likelihood
likelih = likelih_probit('init', y);

% Initialize the GP
gp = gp_init('init', 'PIC_BLOCK', nin, likelih, {gpcf1}, [])
gp = gp_init('set', gp, 'X_u', U, 'blocks', {'manual', x(itr,:), index});
gp.ep_opt.display = 1;
gp = gp_init('set', gp, 'latent_method', {'EP', x(itr,:), y(itr,:), 'hyper'});



% Initialize the test GP used in the scges
gptst = gp_init('init', 'PIC_BLOCK', nin, 'probit', {gpcf1}, [])
gptst = gp_init('set', gptst, 'X_u', U, 'blocks', {'manual', x(its,:), index2});
gptst = gp_init('set', gptst, 'latent_method', {'EP', x(its,:), y(its,:), 'hyper'});


% Intialize weights to zero and set the optimization parameters...
w=randn(size(gp_pak(gp,'hyper')))*0.01;

% do scaled conjugate gradient optimization with early stopping.
gp.ep_opt.display = 1;
[w,fs,vs]=scges(fe, w, optes, fg, gp, x(itr,:), y(itr,:),'hyper', gptst ,x(its,:), y(its,:),'hyper');
gp=gp_unpak(gp,w,'hyper');

% Print the hyperparameter values
fprintf(' The point estimate of length-scale is: %.3f \n The point estimate of magnitude sigma is: %.3f \n',...
        gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

% Print some figures that show results
% First create test inputs
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% ... and set the block indices for them
for i1=1:4
    for i2=1:4
        ind = 1:size(xstar,1);
        ind = ind(: , b1(i1)<=xstar(ind',1) & xstar(ind',1) < b1(i1+1));
        ind = ind(: , b2(i2)<=xstar(ind',2) & xstar(ind',2) < b2(i2+1));        
        index3{4*(i1-1)+i2} = ind';
    end
end

index_tmp = {index3{1:3}};
index_tmp = {index_tmp{:} index3{5:7}};
index_tmp = {index_tmp{:} [index3{8} ; index3{9}] };
index_tmp = {index_tmp{:} index3{10:16}};
index3 = index_tmp;

[Ef, Varf, p1] = ep_pred(gp, x(itr,:),y(itr,:), xstar, index3);
%Ef = ep_pred(gp, x(itr,:),y(itr,:), xstar, index3);

% visualise predictive probability  p(ystar = 1)
figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases', 'fontsize', 14)

% Visualize the predictive variance
figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(Varf,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('marginal predictive latent variance', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with contours
figure, hold on
[cs,h]=contour(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20),[0.025 0.25 0.5 0.75 0.975], 'linewidth', 3);
text_handle = clabel(cs,h);
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==1,1), x(y==1,2), 'rx', 'markersize', 8, 'linewidth', 2),
plot(x(y==-1,1), x(y==-1,2), 'bo', 'markersize', 8, 'linewidth', 2)
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('predictive probability contours with training cases', 'fontsize', 14)

% test how well the network works for the test data. 
L = strrep(S,'demo_clFull.m','demos/synth.ts');
tx=load(L);
ty_temp=tx(:,end);
ty = 2*ty_temp-1;
tx(:,end)=[];

[Eftest, Varftest, p1test] = ep_pred(gp, x(itr,:),y(itr,:), tx);

% calculate the percentage of misclassified points
missed = sum(abs(round(p1test)-ty_temp))/size(ty,1)*100

% Plot the training and test cases in the same figure
figure, hold on;
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
plot(tx(ty==-1,1),tx(ty==-1,2),'go', 'markersize', 8, 'linewidth', 2);
plot(tx(ty==1,1),tx(ty==1,2),'cx', 'markersize', 8, 'linewidth', 2);
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('training and test cases', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with the test cases
figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(tx(ty==-1,1),tx(ty==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(tx(ty==1,1),tx(ty==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and test cases', 'fontsize', 14)



















% $$$ gp.ep_opt.display = 1;
% $$$ gp = gp_init('set', gp, 'latent_method', {'EP', x, y, 'hyper'});
% $$$ 
% $$$ [e, edata, eprior] = gpep_e(gp_pak(gp,'hyper'), gp, x, y, 'hyper')
% $$$ 
% $$$ [g, gdata, gprior] = gpep_g(gp_pak(gp,'hyper'), gp, x(itr,:), y(itr,:), 'hyper')
% $$$ 
% $$$ gp.ep_opt.display = 0;
% $$$ gradcheck(gp_pak(gp,'hyper'), @gpep_e, @gpep_g, gp, x(itr,:), y(itr,:), 'hyper')


% $$$ [e, edata, eprior] = gpep_e(gp_pak(gp,'hyper'), gp, x(itr,:), y(itr,:), 'hyper')


   1.0e+02 *
  -0.023860495207817  -0.042038094605346   0.018177599397528
   1.125051639202053   1.125051639121466   0.000000000080587
   0.009689170941641   0.016481840816596  -0.006792669874954
