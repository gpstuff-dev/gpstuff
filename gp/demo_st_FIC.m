function demo_st_FIC
%DEMO_GPREGR    Regression problem demonstration for 3-input 
%               function with Gaussian process
%
%    Description
%    The problem consist of a data with two input variables
%    and one output variable with Gaussian noise. 
%

% Copyright (c) 2005-2006 Jarno Vanhatalo, Aki Vehtari 
% Copyright (c) 2007      Jouni Hartikainen 
                   

% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.

t_max = 5;
t_0 = 0;
nin = 3;
nin_no_ard = 2;
nin_ard = 1;
[X Y] = meshgrid(-2:0.35:2, -2:0.35:2);
X = X(:);
Y = Y(:);
x_grid = [X Y];
n = size(x_grid,1);
x = zeros(n*(1+t_max),3);
x(:,1:2) = repmat(x_grid,1+t_max,1);
 

axis tight
set(gca,'nextplot','replacechildren')
xlim([-2 2])
ylim([-2 2])
zlim([-1 2])

% Generate the training data from a radial wave equation
phi0 = 0.5;
k = 6;
phase = 0;
t = [t_0:1:t_max];
omega = 1;
phi = zeros(length(X),length(t));
for j = 1:length(t);
    for i = 1:length(X)
        x_1 = X(i);
        y_1 = Y(i);
        r = sqrt(x_1^2+y_1^2);
        if r ~= 0
            phi(i,j) = phi0/r*cos(omega*t(j)-k*r+phase);    
        else
            phi(i,j) = 0;
        end

    end
% $$$     [xi,yi,zi]=griddata(X,Y,phi(:,j),-2:0.35:2,[-2:0.35:2]');
% $$$     surf(xi,yi,zi);   
% $$$     pause
end


y = phi(:);
% Add noise
%y = y + 0.1*randn(length(y),1);
x(:,1:2) = repmat(x_grid,t_max+1,1);
for i = 0:t_max
    x(n*i+1:n*(i+1),3) = t(i+1);
end

% Create covariance functions
gpcf1 = gpcf_st_sexp('init', nin_no_ard, nin_ard, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);

gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 0.2^2);

% Set the prior for the parameters of covariance functions 
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

gp = gp_init('init', 'FIC', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.001);
[u1,u2]=meshgrid(linspace(-1.8,1.8,8),linspace(-1.8,1.8,8));
U = zeros((length(t)-2)*length(u1(:)),3);
for i = t_0+1:t_max-1
    U((i-1)*length(u1(:))+1:i*length(u1(:)),:)=[u1(:) u2(:) i*ones(length(u1(:)),1)];    
end

plot3(U(:,1), U(:,2), U(:,3), 'kX', 'MarkerSize', 12, 'LineWidth', 2)
hold on
plot3(x(:,1), x(:,2), x(:,3), 'ro', 'MarkerSize', 12);

gp = gp_init('set', gp, 'X_u', U);

gradcheck(gp_pak(gp,'hyper'), @gp_e, @gp_g, gp, x, y, 'hyper')

% find starting point using scaled conjucate gradient algorithm
% Intialize weights to zero and set the optimization parameters
w=gp_pak(gp, 'hyper');
fe=str2fun('gp_e');
fg=str2fun('gp_g');
n=length(y);
itr=1:floor(0.5*n);     % training set of data for early stop
its=floor(0.5*n)+1:n;   % test set of data for early stop
optes=scges_opt;
optes.display=1;
optes.tolfun=1e-1;
optes.tolx=1e-1;

% do scaled conjugate gradient optimization with early stopping.
% First for hyperparameters
[w,fs,vs]=scges(fe, w, optes, fg, gp, x(itr,:),y(itr,:), 'hyper', gp,x(its,:), y(its,:), 'hyper');
gp=gp_unpak(gp,w, 'hyper');


% Set the sampling options
opt=gp_mcopt;
opt.repeat=10;
opt.nsamples=10;
opt.hmc_opt.steps=3;
opt.hmc_opt.stepadj=0.01;
opt.hmc_opt.nsamples=1;
hmc2('state', sum(100*clock));

% Sample 
[r,g,rstate1]=gp_mc(opt, gp, x, y);

% These need some adjusting
opt.hmc_opt.stepadj=0.08;
opt.nsamples= 50;
opt.repeat=5;
opt.hmc_opt.steps=3;
opt.hmc_opt.stepadj=0.05;
opt.hmc_opt.persistence=0;
opt.hmc_opt.decay=0.6;

[r,g,rstate2]=gp_mc(opt, gp, x, y, [], [], r);

% Evaluate the MSE for the predictions
out=gp_fwds(r, x, y, x);
mout = mean(squeeze(out)');
pred = zeros(size(x,1),1);
pred(:)=mout;
(pred-y)'*(pred-y)/length(y)


% Calculate the predictions
axis tight
set(gca,'nextplot','replacechildren')
xlim([-2 2])
ylim([-2 2])
zlim([-1 2])
t_test = 0:0.1:5;
pred = cell(1,length(t));
[p1,p2]=meshgrid(-1.8:0.05:1.8,-1.8:0.05:1.8);
for i = 1:length(t)
    p = [p1(:) p2(:) t_test(i)*ones(length(p1(:)),1)];
    out  = gp_fwds(rr, x, y, p);
    mout = mean(squeeze(out)');
    pred{i} = zeros(size(p1));
    pred{i}(:)=mout;
end

% Plot the predictions
for i = 1:n_max
    surf(p1,p2,pred{i});   
    F(i) = getframe;
end

% Play movie
movie(F)

% Use this for predicting at wanted time step.
% t_test = 2;
t_p = t_test*ones(size(p1(:)));
p=[p1(:) p2(:) t_p];
out=gp_fwds(rr, x, y, p);
mout = mean(squeeze(out)');
pred = zeros(size(p1));
pred(:)=mout;
figure
mesh(p1,p2,pred);