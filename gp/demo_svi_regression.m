%DEMO_SVI_REGRESSION  A toy data regression example for sparse SVI GP

% Generate toy data
n = 2000;
nu = 50;
s2 = 0.01;

% Training samples
x = rand(n,2)*2*pi - pi;
y = sin(x(:,1)).*sin(x(:,2)) + sqrt(s2).*randn(n,1);

% Test samples
nt = 100;
xt = rand(nt,2)*2*pi - pi;
yt = sin(xt(:,1)).*sin(xt(:,2));

% Prediction grid
[X1,X2] = meshgrid(-4:0.2:4, -4:0.2:4);
grid = [X1(:), X2(:)];

% Build gp structure
gp = gp_set('lik',lik_gaussian, 'cf', gpcf_sexp, 'latent_method', 'SVI');
gp = gp_set('lik',lik_gaussian, 'cf', gpcf_sexp, 'latent_method', 'SVI');
[gp, diagnosis] = svigp(gp, x, y, 'xt',xt,'yt',yt, 'nu', nu, 'maxiter', 200);

% Predict
Eft = gpsvi_pred(gp,x,y,grid);

%% Plot
figure()
contour(X1, X2, reshape(Eft, size(X1)), 16)
hold on
scatter(x(:,1), x(:,2), 20, y, 'filled')
scatter(gp.X_u(:,1), gp.X_u(:,2), 30)
legend('Eft', 'data', 'Z')

figure()
subplot(3,1,1)
plot(mean(diagnosis.e,2))
title('energy')
subplot(3,1,2)
plot(diagnosis.mlpd)
title('mean log predictive density')
subplot(3,1,3)
plot(diagnosis.rmse)
title('root mean square error')
xlabel('iteration')