%  DEMO_MODEL_COMPARISON  Model Assessment and Comparisons  
%
%  Description: 
%    
%    By using kfc-validation and bayesian bootstrap we compare the predictive 
%    ability of different models by estimating various assessment statistics 
%
%    We will compare two Cox proportional hazars model, the old model will
%    have less covariates than the second model.  
%   
%    The censoring indicator ye is
%    
%      ye = 0 for uncensored event
%      ye = 1 for right censored event.
% 
%    Example data set is leukemia survival data in Northwest England
%    presented in (Henderson, R., Shimakura, S., and Gorst, D. (2002).
%    Modeling spatial variation in leukemia survival data. Journal of the
%    American Statistical Association, 97:965–972). Data set was downloaded
%    from http://www.math.ntnu.no/%7Ehrue/r-inla.org/examples/leukemia/leuk.dat
%
%  See also  DEMO_MODELCOMPARISON2, DEMO_SURVIVAL_COXPH
%
% Copyright (c) 2012 
%
% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.


%% First load data
S = which('demo_survival_weibull');
L = strrep(S,'demo_survival_weibull.m','demos/leukemia.txt');
leukemiadata=load(L);

% leukemiadata consists of:
% 'time', 'cens', 'xcoord', 'ycoord', 'age', 'sex', 'wbc', 'tpi', 'district'

% survival times
y=leukemiadata(:,1);
% scale survival times
y=y/max(y);

ye=1-leukemiadata(:,2); % event indicator, ye = 0 for uncensored event
                  %                        ye = 1 for right censored event

%  we choose for the old model (for example): 'age' and 'sex'covariates
x01=leukemiadata(:,5:6);
x1=x01;
                        
%  we choose for the new model (for example): 'age', 'sex', 'wbc', and 'tpi' covariates
x02=leukemiadata(:,5:8);
x2=x02;

% normalize continuous covariates 

%x1(:,1)=bsxfun(@rdivide,bsxfun(@minus,x01(:,1),mean(x01(:,1),1)),std(x01(:,1),1));
x1(:,1)=normdata(x01(:,1));
%x2(:,[1 3:4])=bsxfun(@rdivide,bsxfun(@minus,x02(:,[1 3:4]),mean(x02(:,[1 3:4]),1)),std(x02(:,[1 3:4]),1));
x2(:,[1 3:4])=normdata(x02(:,[1 3:4]));

[n1, nin1]=size(x1);
[n2, nin2]=size(x2);

% number of time intervals
ntime=50;
% create finite partition of time axis
S=linspace(0,max(y)+0.001,ntime+1);

%% obtain predictions

% Create the covariance functions
pl = prior_t('s2',1, 'nu', 4);
pm = prior_t('s2',1, 'nu', 4); 

% covariance for hazard function
gpcfh1 = gpcf_sexp('lengthScale', 1, 'magnSigma2', 1.1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcfh2 = gpcf_sexp('lengthScale', 1, 'magnSigma2', 1.1, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% covariance for proportional part
gpcf1 = gpcf_sexp('lengthScale', ones(1,size(x1,2)), 'magnSigma2', 1.2, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
gpcf2 = gpcf_sexp('lengthScale', ones(1,size(x2,2)), 'magnSigma2', 1.2, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);


% Create the likelihood structure
lik = lik_coxph();

% use mean of time intervals in hazard function
xtmp=zeros(length(S)-1,1);
for i1=1:(length(S)-1)
    xtmp(i1,1)=mean([S(i1) S(i1+1)]);
end
lik.xtime=xtmp;
lik.stime=S;

% NOTE! if Multiple covariance functions per latent is used, define
% gp.comp_cf as follows:
% gp.comp_cf = {[1 2] [5 6]}
% where [1 2] are for hazard function, and [5 6] for proportional part
gp1 = gp_set('lik', lik, 'cf', {gpcfh1 gpcf2}, 'jitterSigma2', 1e-6);
gp1.comp_cf = {[1] [2]};

gp2 = gp_set('lik', lik, 'cf', {gpcfh1 gpcf2}, 'jitterSigma2', 1e-6);
gp2.comp_cf = {[1] [2]};

% Set the approximate inference method to Laplace
gp1 = gp_set(gp1, 'latent_method', 'Laplace');
gp2 = gp_set(gp2, 'latent_method', 'Laplace');

opt=optimset('TolFun',1e-2,'TolX',1e-4,'Display','iter','Derivativecheck','off');

% obtain predictions for both models using kfc-validation


%* first we set tau
tt=0.1:.1:1;


% set D event indicator vector for each time in tt (Di=0 if i experienced
% the event before tau and Di=1 otherwise)


% Also we set YY, the observed time vector for each time value in tt 

for i=1:size(tt,2)
    for i2=1:size(ye,1)
        if y(i2)>tt(i)
        yytemp(i2)=tt(i);
        Dtemp(i2)=1;   
        else
            if ye(i2)==1
            Dtemp(i2)=1;
            else  
            Dtemp(i2)=0;
            end
        yytemp(i2)=y(i2);
        end
    end
    yyi{i}=yytemp';
    Di{i}=Dtemp';
end


for i=1:size(Di,2)
    D(:,i)=Di{i};
end

for i=1:size(yyi,2)
    yy(:,i)=yyi{i};
end


% set time vector to make predictions
yt=bsxfun(@times,ones(size(y)),tt);

% Obtain predictions
[cdf1]=gp_kfcv_cdf(gp1,x1,y,'z',D,'yt',yt,'opt',opt);
[cdf2]=gp_kfcv_cdf(gp2,x2,y,'z',D,'yt',yt,'opt',opt);

crit1=cdf1;
crit2=cdf2;


  % Save results
  save results.mat crit1 crit2
  
  % Load results
  load results.mat

%% Calculate statics and compare models 
% MODEL COMPARISON

%% AUC 
% AUC for Binary outcomes P(Pi>Pj | Di=1,Dj=0)
[auc1,fps1,tps1]=aucs(crit1(:,length(tt)),D(:,length(tt)));
[auc2,fps2,tps2]=aucs(crit2(:,length(tt)),D(:,length(tt)));


st1=sprintf(['\n AUC at end of study for model 1:   ', num2str(auc1)]);
st2=sprintf(['\n AUC at end of study for model 2:   ', num2str(auc2)]);
display(st1)
display(st2)
hold on
plot(fps1,tps1,'b')
plot(fps2,tps2,'r')
title('ROC curve')
legend('model 1', 'model 2')
hold off

%% Harrell's C
% Obtain for both models Binary AUC(t) = P(Pi>Pj | Di(t)=1,Dj(t)=0) and
% Harrell's C(t) = P(Pi>Pj | Di(ti)=1, ti<tj, ti<tt) for every element of tt  
[aut,c]=assess(crit1,crit2,yy,D,tt);

% Plot for both models Harrells C in function of time
plot(tt,c(:,1),'r');
hold on;
plot(tt,c(:,2),'g');
legend('Old model','New model')
title('Harrolds C in function of time ');
xlabel('Time');
ylabel('Harrolds C');
hold off;

%% Estimated density
% Use bayesian bootsrap to obtain Harrells (C1-C2) statistic density at tt=1
[c1,bb1]=hcs(crit1(:,size(tt,2)),y,ye,1,'rsubstream',1);
[c2,bb2]=hcs(crit2(:,size(tt,2)),y,ye,1,'rsubstream',1);
title('Estimated density of C2-C1')
hold on 
lgpdens(bb2-bb1)
hold off 


% We integrate the (C1-C2) estimated density in the (0,inf) interval
zc=lgpdens_cum(bb2-bb1,0,inf);
st1=sprintf(['Estimated c statistics for model 1 and 2 respectively:   ', num2str(c1) '  ' num2str(c2)]);
st2=sprintf(['cumulative probability in the (0,inf) interval:   ', num2str(zc)]);
display(st1);
display(st2);

%% IDI
%Estimate r² for both models, idi, its density and the cumulative
%probability in the (0,inf) interval, al at time 1

[idi,bbid,r1,r2] = idis(crit1(:,size(tt,2)),crit2(:,size(tt,2)),'rsubstream',1);
zidi=lgpdens_cum(bbid,0,inf);
title('IDI estimated density')
hold on 
lgpdens(bbid)
hold off 

st1=sprintf(['\n R² statistic for model 1:   ', num2str(r1)]);
st2=sprintf(['\n R² statistic for model 2:   ', num2str(r2)]);
display(st1)
display(st2)

st1=sprintf(['Estimated idi: ', num2str(idi)]);
st2=sprintf(['cumulative probability in the (0,inf) interval: ', num2str(zidi)]);
display(st1)
display(st2)

%% EXT AUC

% Ext_AUC for different subsets of tt 
Indxtmp{1}=1:1:size(tt,2);
Indx{1}=1:1:size(tt,2);
j=2;
k=round(size(tt,2)/2); 
for i=2:k
    Indxtmp{i}=1:i:size(tt,2);
    if length(Indxtmp{i})~=length(Indxtmp{i-1})
        Indx{j}=Indxtmp{i};
        j=j+1;
    end
    
end


for i=1:size(Indx,2)
l(i)=length(Indx{i});
end

for i=1:size(Indx,2)
ea1(i) = ext_auc(crit1(:,Indx{i}),tt(:,Indx{i}),tt(:,Indx{i}(size(Indx{i},2))));
ea2(i) = ext_auc(crit2(:,Indx{i}),tt(:,Indx{i}),tt(:,Indx{i}(size(Indx{i},2))));
end

hold on
xlabel('Number of distinct time partitions')
ylabel('Extended AUC')
plot(wrev(l),ea1,'r')
plot(wrev(l),ea2,'b')
legend('Traditional model', 'New model')
hold off

st1=sprintf(['\n ExtAUC at end of study for model 1:   ', num2str(ea1(size(Indx,2)))]);
st2=sprintf(['\n ExtAUC at end of study for model 2:   ', num2str(ea2(size(Indx,2)))]);
display(st1)
display(st2)


%% plot predictions for average individual 
%********************  Superimpose a plot of a prediction for an average
% individual
% choose (for example) 'age', 'sex', 'wbc', and 'tpi' covariates
% average covariates except sex 
xa=mean(x2(:,[1,3,4]),1);
% -1 for female
xaf=[xa(1) -1 xa(2:3)];
%[Ef1, Varf1] = gp_pred(gp, x, y,xa ,'z', ye);
% 1 for male 
xam=[xa(1) 1 xa(2:3)];


% optimise parameters
opt=optimset('TolFun',1e-4,'TolX',1e-4,'Display','iter');
gp1=gp_optim(gp1,x1,y,'z',ye,'opt',opt);
gp2=gp_optim(gp2,x2,y,'z',ye,'opt',opt);


% obtain predictions 


% model 1

[h1f surv1f]=pred_coxphhs(gp1,x1,y,xaf(:,1:2),'z',ye);
[h1m surv1m]=pred_coxphhs(gp1,x1,y,xam(:,1:2),'z',ye);


% model 2

[h2f surv2f]=pred_coxphhs(gp2,x2,y,xaf,'z',ye);
[h2m surv2m]=pred_coxphhs(gp2,x2,y,xam,'z',ye);


% Calculate and plot empirical survival and confidence bounds
qf=1;
qm=1;
for i=1:size(y,1)
    if x2(i,2) == 1
        ym(qm) = y(i);
        yem(qm) = ye(i);
        qm=qm+1;
    else
        yf(qf) = y(i);
        yef(qf) = ye(i);
        qf=qf+1;
    end
end

[fm,zm] = ecdf(ym,'censoring',yem,'function','survivor');
clf
stairs(zm,fm,'LineWidth',2,'color','b')
hold on

[ff,zf] = ecdf(yf,'censoring',yef,'function','survivor');
stairs(zf,ff,'LineWidth',2,'color','y')

surv_1f=[1 surv1f];
surv_2f=[1 surv2f];

surv_1m=[1 surv1m];
surv_2m=[1 surv2m];

zz=[gp1.lik.stime];

plot(zz,surv_1f,'r--','LineWidth',2)
plot(zz,surv_1m,'r','LineWidth',2)
plot(zz,surv_2f,'g--','LineWidth',2)
plot(zz,surv_2m,'g','LineWidth',2)

legend('Empirical male ', 'Empirical female','CoxPh Model 1 Female', 'CoxPh Model 1 Male', 'CoxPh Model 2 Female', 'CoxPh Model 2 Male')
hold off






