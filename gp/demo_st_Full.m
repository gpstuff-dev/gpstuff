function demo_st_Full
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

t_max = 4;
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
x_grid = [X Y];

axis tight
set(gca,'nextplot','replacechildren')
xlim([-2 2])
ylim([-2 2])
zlim([-1 2])

% Generate the training data from a radial wave equation
phi0 = 0.5;
k = 6;
phase = 0;
t = [0:1:4];
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
    % Uncomment for visualization
    [xi,yi,zi]=griddata(X,Y,phi(:,j),-2:0.35:2,[-2:0.35:2]');
    surf(xi,yi,zi);   
    pause
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

%gpcf1 = gpcf_sexp_cs3('init', 3, 'lengthScale', [1 1], 'magnSigma2', 0.2^2);


gpcf2 = gpcf_noise('init', 'noiseSigmas2', 0.2^2);

% Set the prior for the parameters of covariance functions 
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

gp = gp_init('init', 'FULL', 'regr', {gpcf1}, {gpcf2}, 'jitterSigma2', 0.001.^2)

w=gp_pak(gp, 'hyper');
gp_e(w, gp, x, y, 'hyper')    % answer 370.9230

gradcheck(gp_pak(gp,'hyper'), @gp_e, @gp_g, gp, x, y, 'hyper')

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

figure
[xi,yi,zi]=griddata(data(:,1),data(:,2),pred,-1.8:0.01:1.8,[-1.8:0.01:1.8]');
mesh(xi,yi,zi)
title('The prediction');

(pred-y)'*(pred-y)/length(y)

% Calculate the predictions
[p1,p2]=meshgrid(-1.8:0.05:1.8,-1.8:0.05:1.8);
axis tight
set(gca,'nextplot','replacechildren')
xlim([-2 2])
ylim([-2 2])
zlim([-1 1])
pred = cell(1,length(t));
for i = 1:n_max
    p = [p1(:) p2(:) t(j)];
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

% Calculate predictions for wanted time step.
t_test = 2;
t = t_test*ones(size(p1(:)));
p=[p1(:) p2(:) t];
out=gp_fwds(rr, x, y, p);
mout = mean(squeeze(out)');
pred = zeros(size(p1));
pred(:)=mout;
figure
mesh(p1,p2,pred);
