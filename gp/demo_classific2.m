function demo_classific2
%DEMO_CLAASIFIC2    Classification problem demonstration for 2 classes via Laplace 
%                   approximation and EP
%
%      Description
%      The demonstration program is based on synthetic two 
%      class data used by B.D. Ripley (Pattern Regocnition and
%      Neural Networks, 1996}. The data consists of 2-dimensional
%      vectors that are divided into to classes, labeled -1 or 1.
%      Each class has a bimodal distribution generated from equal
%      mixtures of Gaussian distributions with identical covariance
%      matrices. A Bayesian aproach is used to find the decision
%      line and predict the classes of new data points.
%
%      The probability of y being one is assumed to be 
%
%            p(y=1|f) = int_{-inf}^{yf} N(x|0,1) dx
%
%      (Compare this to logistic likelihood in demo_classific1 and see 
%      Rasmussen and Williams (2006) for details). 
%
%      The latent values f are given a zero mean Gaussian process prior. 
%      This implies that at the observed input locations latent values
%      have prior 
%
%         f ~ N(0, K),
%
%      where K is the covariance matrix, whose elements are given as 
%      K_ij = k(x_i, x_j | th). The function k(x_i, x_j | th) is covariance 
%      function and th its parameters, hyperparameters. 
% 
%      Here we Laplace approximation and expectation propagation to find normal 
%      approximation for the posterior of the latent values and optimize the
%      hyperparameters in their MAP point. See Rasmussen and Williams (2006) for 
%      details.
%
%      NOTE! The class labels have to be {-1,1} for probit likelihood 
%      (different from the logit likelihood).

% Copyright (c) 2008 Jarno Vanhatalo

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

% This demonstration is based on the dataset used in the book Pattern Recognition and
% Neural Networks by B.D. Ripley (1996) Cambridge University Press ISBN 0 521
% 46986 7

%==================================================================
% PART 1 data analysis with full GP model and Laplace approximation
%==================================================================

S = which('demo_classific1');
L = strrep(S,'demo_classific1.m','demos/synth.tr');
x=load(L);
y=x(:,end);
y=y*2-1;
x(:,end)=[];
[n, nin] = size(x);

% Create covariance functions
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [0.6 0.9], 'magnSigma2', 0.5);

% Set the prior for the parameters of covariance functions 
% $$$ gpcf1.p.lengthScale = gamma_p({3 7 3 7});
% $$$ gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});
gpcf1.p.lengthScale = unif_p;
gpcf1.p.magnSigma2 = unif_p;


% Create the likelihood structure
likelih = likelih_probit('init', y);

% Create the GP data structure
gp = gp_init('init', 'FULL', nin, likelih, {gpcf1}, {}, 'jitterSigmas', 0.001);   %{gpcf2}

% Set the approximate inference method
tt = cputime;
gp = gp_init('set', gp, 'latent_method', {'Laplace', x, y, 'hyper'});
cputime - tt

gp.laplace_opt.optim_method = 'newton';

w = gp_pak(gp, 'hyper');
gradcheck(w, @gpla_e, @gpla_g, gp, x, y, 'hyper')

fe=str2fun('gpla_e');
fg=str2fun('gpla_g');
n=length(y);
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do scaled conjugate gradient optimization 
w=gp_pak(gp, 'hyper');
[w, opt, flog]=scg2(fe, w, opt, fg, gp, x, y, 'hyper');
gp=gp_unpak(gp,w, 'hyper');

% Print some figures that show results
% First create data for predictions
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% make the prediction
[Ef, Varf, p1] = la_pred(gp, x, y, xstar, 'hyper');

figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases, full GP with Laplace', 'fontsize', 14)

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
set(gcf, 'color', 'w'), title('predictive probability contours, full GP with Laplace', 'fontsize', 14)


%========================================================
% PART 2 data analysis with FIC GP model and Laplace approximation
%========================================================
S = which('demo_classific1');
L = strrep(S,'demo_classific1.m','demos/synth.tr');
x=load(L);
y=x(:,end);
y=y*2-1;
x(:,end)=[];
[n, nin] = size(x);

% Create covariance functions
%gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [0.6 0.9], 'magnSigma2', 0.5);
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [0.6 0.8], 'magnSigma2', 0.2);
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [1 1], 'magnSigma2', 1);

% Set the prior for the parameters of covariance functions 
% $$$ gpcf1.p.lengthScale = gamma_p({3 7 3 7});
% $$$ gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});
gpcf1.p.lengthScale = unif_p;
gpcf1.p.magnSigma2 = unif_p;

% Create the likelihood structure
likelih = likelih_probit('init', y);

% Set the inducing inputs
[u1,u2]=meshgrid(linspace(-1.25, 0.9,6),linspace(-0.2, 1.1,6));
Xu=[u1(:) u2(:)];
Xu = Xu([3 4 7:18 20:24 26:30 33:36],:);

% Create the GP data structure
gp_fic = gp_init('init', 'FIC', nin, likelih, {gpcf1}, [], 'jitterSigmas', 0, 'X_u', Xu);   %{gpcf2}

% Set the approximate inference method
gp_fic = gp_init('set', gp_fic, 'latent_method', {'Laplace', x, y, 'hyper'});

gp_fic.laplace_opt.optim_method = 'newton';
%gp_fic.laplace_opt.optim_method = 'fminunc_large';

w = gp_pak(gp_fic, 'hyper');
gradcheck(w, @gpla_e, @gpla_g, gp_fic, x, y, 'hyper')

fe=str2fun('gpla_e');
fg=str2fun('gpla_g');
n=length(y);
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do scaled conjugate gradient optimization 
w=gp_pak(gp_fic, 'hyper');
[w, opt, flog]=scg2(fe, w, opt, fg, gp_fic, x, y, 'hyper');
gp_fic=gp_unpak(gp_fic,w, 'hyper');

% Print some figures that show results
% First create data for predictions
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% make the prediction
[Ef_fic, Varf_fic, p1_fic] = la_pred(gp_fic, x, y, xstar, 'hyper');

figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_fic,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases, FIC and Laplace', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with contours
figure, hold on
[cs,h]=contour(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_fic,20,20),[0.025 0.25 0.5 0.75 0.975], 'linewidth', 3);
text_handle = clabel(cs,h);
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==1,1), x(y==1,2), 'rx', 'markersize', 8, 'linewidth', 2),
plot(x(y==-1,1), x(y==-1,2), 'bo', 'markersize', 8, 'linewidth', 2)
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('predictive probability contours, FIC and Laplace', 'fontsize', 14)

%========================================================
% PART 3 data analysis with PIC GP model and Laplace approximation
%========================================================
S = which('demo_classific1');
L = strrep(S,'demo_classific1.m','demos/synth.tr');
x=load(L);
y=x(:,end);
y=y*2-1;
x(:,end)=[];
[n, nin] = size(x);

% Create covariance functions
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [0.2 0.2], 'magnSigma2', 2);

% Set the prior for the parameters of covariance functions 
gpcf1.p.lengthScale = gamma_p({3 7 3 7});
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% Create the likelihood structure
likelih = likelih_probit('init', y);

% Set the blocks and the inducing inputs
[u1,u2]=meshgrid(linspace(-1.25, 0.9,7),linspace(-0.2, 1.1,7));
Xu=[u1(:) u2(:)];
%Xu = Xu([3 4 7:18 20:24 26:30 33:36],:);

b1 = linspace(-1.25, 0.9, 5);
b2 = linspace(-0.2,  1.1, 5);
for i1=1:4
    for i2=1:4
        ind = 1:size(x,1);
        ind = ind(: , b1(i1)<=x(ind',1) & x(ind',1) < b1(i1+1));
        ind = ind(: , b2(i2)<=x(ind',2) & x(ind',2) < b2(i2+1));        
        index{4*(i1-1)+i2} = ind';
    end
end
index = {index{[1:3 5:16]}};

% $$$ for i=1:250
% $$$     index{i} = i;
% $$$ end

% Create the GP data structure
gp_pic = gp_init('init', 'PIC', nin, likelih, {gpcf1}, [], 'jitterSigmas', 0.01, 'X_u', Xu);   %{gpcf2}
gp_pic = gp_init('set', gp_pic, 'blocks', {'manual', x, index});

% Set the approximate inference method
gp_pic = gp_init('set', gp_pic, 'latent_method', {'Laplace', x, y, 'hyper'});

% $$$ figure
% $$$ col = {'b*','g*','r*','c*','m*','y*','k*','b*','b.','g.','r.','c.','m.','y.','k.','b.'};
% $$$ hold on
% $$$ for i=1:length(index)
% $$$     plot(x(index{i},1),x(index{i},2),col{i})
% $$$ end 

fe=str2fun('gpla_e');
fg=str2fun('gpla_g');
n=length(y);
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do scaled conjugate gradient optimization 
w=gp_pak(gp_pic, 'hyper');
[w, opt, flog]=scg2(fe, w, opt, fg, gp_pic, x, y, 'hyper');
gp_pic=gp_unpak(gp_pic,w, 'hyper');

% Print some figures that show results
% First create data for predictions
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% make the prediction
[Ef_pic, Varf_pic, p1_pic] = la_pred(gp_pic, x, y, xstar, 'hyper');

figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_pic,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases, FIC and Laplace', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with contours
figure, hold on
[cs,h]=contour(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_pic,20,20),[0.025 0.25 0.5 0.75 0.975], 'linewidth', 3);
text_handle = clabel(cs,h);
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==1,1), x(y==1,2), 'rx', 'markersize', 8, 'linewidth', 2),
plot(x(y==-1,1), x(y==-1,2), 'bo', 'markersize', 8, 'linewidth', 2)
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('predictive probability contours, FIC and Laplace', 'fontsize', 14)



%==================================================================
% PART 4 data analysis with full GP model and expectation propagation
%==================================================================

S = which('demo_classific1');
L = strrep(S,'demo_classific1.m','demos/synth.tr');
x=load(L);
y=x(:,end);
y=y*2-1;
x(:,end)=[];
[n, nin] = size(x);

% Create covariance functions
%gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [0.9 0.9], 'magnSigma2', 1);
gpcf1 = gpcf_ppcs2('init', nin, 'lengthScale', [0.3 0.3], 'magnSigma2', 1);

% Set the prior for the parameters of covariance functions 
gpcf1.p.lengthScale = gamma_p({3 7 3 7});
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% Create the likelihood structure
likelih = likelih_probit('init', y);

% Create the GP data structure
gp = gp_init('init', 'FULL', nin, likelih, {gpcf1}, [], 'jitterSigmas', 0.0001);   %{gpcf2}

% Set the approximate inference method
tt = cputime;
gp = gp_init('set', gp, 'latent_method', {'EP', x, y, 'hyper'});

fe=str2fun('gpep_e');
fg=str2fun('gpep_g');
n=length(y);
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do scaled conjugate gradient optimization 
gp.ep_opt.display = 1;
w=gp_pak(gp, 'hyper');
[w, opt, flog]=scg2(fe, w, opt, fg, gp, x, y, 'hyper');
gp=gp_unpak(gp,w, 'hyper');
cputime - tt

% Print some figures that show results
% First create data for predictions
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% make the prediction
[Ef, Varf, p1] = ep_pred(gp, x, y, xstar, 'hyper');

figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases, full GP with EP', 'fontsize', 14)

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
set(gcf, 'color', 'w'), title('predictive probability contours, full GP with EP', 'fontsize', 14)


%========================================================
% PART 5 data analysis with FIC GP model and expectation propagation
%========================================================

% Set the inducing inputs
[u1,u2]=meshgrid(linspace(-1.25, 0.9,6),linspace(-0.2, 1.1,6));
Xu=[u1(:) u2(:)];
Xu = Xu([3 4 7:18 20:24 26:30 33:36],:);

% Create the GP data structure
gp_fic = gp_init('init', 'FIC', nin, likelih, {gpcf1}, [], 'jitterSigmas', 0.01, 'X_u', Xu);   %{gpcf2}

% Set the approximate inference method
gp_fic = gp_init('set', gp_fic, 'latent_method', {'EP', x, y, 'hyper'});

fe=str2fun('gpep_e');
fg=str2fun('gpep_g');
n=length(y);
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do scaled conjugate gradient optimization 
gp_fic.ep_opt.display = 1;
w=gp_pak(gp_fic, 'hyper');
[w, opt, flog]=scg2(fe, w, opt, fg, gp_fic, x, y, 'hyper');
gp_fic=gp_unpak(gp_fic,w, 'hyper');

% Print some figures that show results
% First create data for predictions
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% make the prediction
[Ef_fic, Varf_fic, p1_fic] = ep_pred(gp_fic, x, y, xstar, 'hyper');

figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_fic,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases, FIC with EP', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with contours
figure, hold on
[cs,h]=contour(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_fic,20,20),[0.025 0.25 0.5 0.75 0.975], 'linewidth', 3);
text_handle = clabel(cs,h);
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==1,1), x(y==1,2), 'rx', 'markersize', 8, 'linewidth', 2),
plot(x(y==-1,1), x(y==-1,2), 'bo', 'markersize', 8, 'linewidth', 2)
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('predictive probability contours, FIC with EP', 'fontsize', 14)

































%========================================================
% PART  data analysis with CS+FIC GP model and expectation propagation
%========================================================
S = which('demo_classific1');
L = strrep(S,'demo_classific1.m','demos/synth.tr');
x=load(L);
y=x(:,end);
y=y*2-1;
x(:,end)=[];
[n, nin] = size(x);

% Create covariance functions
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', [0.9 0.9], 'magnSigma2', 1);
gpcf1.p.lengthScale = gamma_p({3 7 3 7});
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

gpcf2 = gpcf_ppcs2('init', nin, 'lengthScale', 3, 'magnSigma2', 0.03);
gpcf2.p.lengthScale = t_p({1 4});
gpcf2.p.magnSigma2 = t_p({0.3 4});

% Set the inducing inputs
[u1,u2]=meshgrid(linspace(-1.25, 0.9,6),linspace(-0.2, 1.1,6));
Xu=[u1(:) u2(:)];
Xu = Xu([3 4 7:18 20:24 26:30 33:36],:);

% Create the likelihood structure
likelih = likelih_probit('init', y);

% Create the GP data structure
gp = gp_init('init', 'CS+FIC', nin, likelih, {gpcf1, gpcf2}, [], 'X_u', Xu, 'jitterSigmas', 0.0001);   %{gpcf2}

% Set the approximate inference method
gp = gp_init('set', gp, 'latent_method', {'EP', x, y, 'hyper+inducing'});
gp = gp_init('set', gp, 'latent_method', {'Laplace', x, y, 'hyper+inducing'});



fe=str2fun('gpep_e');
fg=str2fun('gpep_g');
n=length(y);
opt = scg2_opt;
opt.tolfun = 1e-3;
opt.tolx = 1e-3;
opt.display = 1;

% do scaled conjugate gradient optimization 
gp_fic.ep_opt.display = 1;
w=gp_pak(gp_fic, 'hyper');
[w, opt, flog]=scg2(fe, w, opt, fg, gp_fic, x, y, 'hyper');
gp_fic=gp_unpak(gp_fic,w, 'hyper');

% Print some figures that show results
% First create data for predictions
xt1=repmat(linspace(min(x(:,1)),max(x(:,1)),20)',1,20);
xt2=repmat(linspace(min(x(:,2)),max(x(:,2)),20)',1,20)';
xstar=[xt1(:) xt2(:)];

% make the prediction
[Ef_fic, Varf_fic, p1_fic] = ep_pred(gp_fic, x, y, xstar, 'hyper');

figure, hold on;
n_pred=size(xstar,1);
h1=pcolor(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_fic,20,20))
set(h1, 'edgealpha', 0), set(h1, 'facecolor', 'interp')
colormap(repmat(linspace(1,0,64)', 1, 3).*repmat(ones(1,3), 64,1))
axis([-inf inf -inf inf]), axis off
plot(x(y==-1,1),x(y==-1,2),'o', 'markersize', 8, 'linewidth', 2);
plot(x(y==1,1),x(y==1,2),'rx', 'markersize', 8, 'linewidth', 2);
set(gcf, 'color', 'w'), title('predictive probability and training cases, FIC with EP', 'fontsize', 14)

% visualise predictive probability  p(ystar = 1) with contours
figure, hold on
[cs,h]=contour(reshape(xstar(:,1),20,20),reshape(xstar(:,2),20,20),reshape(p1_fic,20,20),[0.025 0.25 0.5 0.75 0.975], 'linewidth', 3);
text_handle = clabel(cs,h);
set(text_handle,'BackgroundColor',[1 1 .6],'Edgecolor',[.7 .7 .7],'linewidth', 2, 'fontsize',14)
c1=[linspace(0,1,64)' 0*ones(64,1) linspace(1,0,64)'];
colormap(c1)
plot(x(y==1,1), x(y==1,2), 'rx', 'markersize', 8, 'linewidth', 2),
plot(x(y==-1,1), x(y==-1,2), 'bo', 'markersize', 8, 'linewidth', 2)
plot(xstar(:,1), xstar(:,2), 'k.'), axis([-inf inf -inf inf]), axis off
set(gcf, 'color', 'w'), title('predictive probability contours, FIC with EP', 'fontsize', 14)


