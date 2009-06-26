%% generate data
clear;
datai=1;

% true function f(x)
xx = linspace(-7,5,500)';
yy = 0.1+0.1*xx+0.2*sin(2.7*xx)+1./(1+xx.^2);

nu=3;
sigma=0.05;
Hmax=sqrt(3*nu)*sigma;
Hzero=sqrt(nu)*sigma;
a=0.3;
ylim=[-0.4 0.9];

x=[linspace(-5.4,-3-a,15) linspace(-3+a,-1-a,15) linspace(-1+a,4,30)];
xo=[-4 -3 -2 -1 2 3];
ii=length(xo)+[1:length(x)];
io=1:length(xo); % outlier indices
x=[xo x]'; % outlier x:s
y = 0.1 + 0.1*x + 0.2*sin(2.7*x) + 1 ./ (1+x.^2);
y(io)=y(io)+Hmax*[-3 -2 1 -1 -1 1]'; % outlier y:s
y(ii)=y(ii)+sigma*randn(size(y(ii)));

yt=y;
y(y<ylim(1))=ylim(1);
y(y>ylim(2))=ylim(2);

figure(1); clf
plot(xx,yy,'k',x,y,'b.',x(io),y(io),'ro')
hold on
plot(repmat(x(io)',2,1),[y(io)'-Hmax; y(io)'+Hmax],'r.-')
plot(repmat(x(io)',2,1),[y(io)'-Hzero; y(io)'+Hzero],'g.-')
hold off
%save(sprintf('data%d.mat',datai),'x','y','yt','xx','yy','io','ylim')

%% load data & create gp
%clear
%datai=1;
%load(sprintf('data%d.mat',datai));
nu=6;
sigma=0.1;
J=0.02;
x=[x randn(size(x))];
xx=[xx randn(size(xx))];
[n, nin] = size(x);
ylim=[-0.4 0.9];

% gpcf1 = gpcf_dotproduct('init', nin, 'constSigma2',10,'coeffSigma2',ones(1,nin));
% gpcf1.p.constSigma2 = logunif_p;
% gpcf1.p.coeffSigma2 = logunif_p;

gpcf1 = gpcf_sexp('init', nin, 'lengthScale',ones(1,nin),'magnSigma2',1);
gpcf1.p.lengthScale = logunif_p;
gpcf1.p.magnSigma2 = logunif_p;


% gpcf1 = gpcf_neuralnetwork('init',nin,'biasSigma2',0.1,'weightSigma2',ones(1,nin));
% gpcf1.p.weightSigma2 = logunif_p;
% gpcf1.p.biasSigma2 = logunif_p;

% Create the likelihood structure
%likelih = likelih_t('init', nu, sigma);
likelih = likelih_cen_t('init', nu, sigma, ylim);
likelih.p.nu = logunif_p;
likelih.p.sigma = logunif_p;
%likelih.freeze_nu=1;

% Laplace approximation Student-t likelihood
param = 'hyper+likelih';
gp_la = gp_init('init', 'FULL', nin, likelih, {gpcf1}, {}, 'jitterSigmas', J);
gp_la = gp_init('set', gp_la, 'latent_method', {'Laplace', x, y, param});
gp_la.laplace_opt.optim_method='likelih_specific';
%gp_la.laplace_opt.optim_method='fminunc_large';
[e, edata, eprior, f, L, a, La2] = gpla_e(gp_pak(gp_la,param), gp_la, x, y, param);

% gradient checking
w = gp_pak(gp_la,param);
w = w+ 0.1*randn(size(w));
gradcheck(w, @gpla_e, @gpla_g, gp_la, x, y, param)
 
opt=optimset('GradObj','on');
opt=optimset(opt,'TolX', 1e-3);
opt=optimset(opt,'LargeScale', 'off');
opt=optimset(opt,'Display', 'iter');
opt=optimset(opt,'Derivativecheck', 'on'); % 'iter'

w0 = gp_pak(gp_la, param);
mydeal = @(varargin)varargin{1:nargout};
w = fminunc(@(ww) mydeal(gpla_e(ww, gp_la, x, y, param), gpla_g(ww, gp_la, x, y, param)), w0, opt);
gp_la = gp_unpak(gp_la,w,param);

fprintf('\nnu=%.3f, sigma=%.3f \nhyper=%s\n',gp_la.likelih.nu,...
 gp_la.likelih.sigma,sprintf(' %.2f,',exp(gp_pak(gp_la,'hyper'))) )

figure(2)
[e, edata, eprior, f, L, a, La2] = gpla_e(gp_pak(gp_la,param), gp_la, x, y, param);
W=-gp_la.likelih.fh_g2(gp_la.likelih,y,f,'latent');
[foo,ii]=sort(W,'ascend');
ii=ii(1:5);
plot(xx(:,1),yy,'k',x(:,1),f,'b.',x(:,1),y,'go',x(ii,1),y(ii),'r.')

[Ef_la, Varf_la] = la_pred(gp_la, x, y, xx, param);
stdf_la = sqrt(Varf_la);

% plot the predictions and data
nu=gp_la.likelih.nu;
sigma=gp_la.likelih.sigma;
Hmax=sqrt(3*nu)*sigma;
Hzero=sqrt(nu)*sigma;

figure(1)
h1=plot(xx(:,1),yy,'k',xx(:,1),Ef_la,'b',xx(:,1),Ef_la-2*stdf_la, 'b--',xx(:,1), Ef_la+2*stdf_la, 'b--');
hold on
h1=[h1(1:2); plot(x(:,1),y,'k.')];
plot(repmat(x(io,1)',2,1),[y(io)'-Hmax; y(io)'+Hmax],'r.-')
plot(repmat(x(io,1)',2,1),[y(io)'-Hzero; y(io)'+Hzero],'g.-')
hold off
legend(h1,'True','Laplace','Data')

