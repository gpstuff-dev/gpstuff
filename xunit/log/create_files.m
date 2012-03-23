%   Author: Aki Vehtari <Aki.Vehtari@hut.fi>
%   Last modified: 2011-03-10 11:15:15 EET
diary off;clear;close all; 
load realValuesBinomial1.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_binomial1.txt');
disp('Running: demo_binomial1')
demo_binomial1
fprintf('\n gp hyperparameters: \n \n')
disp(gp_pak(gp))
diary off;
save('realValuesBinomial1.mat', char(var(1)));
for i=2:length(var)
  save('realValuesBinomial1.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesBinomial_apc.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_binomial_apc.txt');
disp('Running: demo_binomial_apc')
demo_binomial_apc
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesBinomial_apc.mat', char(var(1)));
for i=2:length(var)
  save('realValuesBinomial_apc.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesClassific.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_classific.txt');
disp('Running: demo_classific')
demo_classific
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesClassific.mat', char(var(1)));
for i=2:length(var)
  save('realValuesClassific.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesDerivativeobs.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_derivativeobs.txt');
disp('Running: demo_derivativeobs')
demo_derivativeobs
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesDerivativeobs.mat', char(var(1)));
for i=2:length(var)
  save('realValuesDerivativeobs.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesLgcp.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_lgcp.txt');
disp('Running: demo_lgcp')
demo_lgcp
diary off;
save('realValuesLgcp.mat', char(var(1)));
for i=2:length(var)
  save('realValuesLgcp.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesModelAssesment1.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_modelassesment1.txt');
disp('Running: demo_modelassesment1')
demo_modelassesment1
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesModelAssesment1.mat', char(var(1)));
for i=2:length(var)
  save('realValuesModelAssesment1.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesModelAssesment2.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_modelassesment2.txt');
disp('Running: demo_modelassesment2')
demo_modelassesment2
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesModelAssesment2.mat', char(var(1)));
for i=2:length(var)
  save('realValuesModelAssesment2.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesMulticlass.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_multiclass.txt');
disp('Running: demo_multiclass')
demo_multiclass
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesMulticlass.mat', char(var(1)));
for i=2:length(var)
  save('realValuesMulticlass.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesNeuralnetcov.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_neuralnetcov.txt');
disp('Running: demo_neuralnetcov')
demo_neuralnetcov
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesNeuralnetcov.mat', char(var(1)));
for i=2:length(var)
  save('realValuesNeuralnetcov.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesPeriodic.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_periodic.txt');
disp('Running: demo_periodic')
demo_periodic
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesPeriodic.mat', char(var(1)));
for i=2:length(var)
  save('realValuesPeriodic.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesRegression1.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_regression1.txt');
disp('Running: demo_regression1')
demo_regression1
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
% drawnow;clear;close all
% disp('Running: demo_regression2')
% demo_regression2
diary off;
save('realValuesRegression1.mat', char(var(1)));
for i=2:length(var)
  save('realValuesRegression1.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesRegression_additive1.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_regression_additive1.txt');
disp('Running: demo_regression_additive1')
demo_regression_additive1
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesRegression_additive1.mat', char(var(1)));
for i=2:length(var)
  save('realValuesRegression_additive1.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesRegression_additive2.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_regression_additive2.txt');
disp('Running: demo_regression_additive2')
demo_regression_additive2
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesRegression_additive2.mat', char(var(1)));
for i=2:length(var)
  save('realValuesRegression_additive2.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesRegression_hier.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_regression_hier.txt');
disp('Running: demo_regression_hier')
demo_regression_hier
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesRegression_hier.mat', char(var(1)));
for i=2:length(var)
  save('realValuesRegression_hier.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesRegression_meanf.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_regression_meanf.txt');
disp('Running: demo_regression_meanf')
demo_regression_meanf
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesRegression_meanf.mat', char(var(1)));
for i=2:length(var)
  save('realValuesRegression_meanf.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesRegression_ppcs.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_regression_ppcs.txt');
disp('Running: demo_regression_ppcs')
demo_regression_ppcs
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesRegression_ppcs.mat', char(var(1)));
for i=2:length(var)
  save('realValuesRegression_ppcs.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesRegression_robust.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_regression_robust.txt');
disp('Running: demo_regression_robust')
demo_regression_robust
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesRegression_robust.mat', char(var(1)));
for i=2:length(var)
  save('realValuesRegression_robust.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesRegression_sparse1.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_regression_sparse1.txt');
disp('Running: demo_regression_sparse1')
demo_regression_sparse1
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesRegression_sparse1.mat', char(var(1)));
for i=2:length(var)
  save('realValuesRegression_sparse1.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesRegression_sparse2.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_regression_sparse2.txt');
disp('Running: demo_regression_sparse2')
demo_regression_sparse2
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesRegression_sparse2.mat', char(var(1)));
for i=2:length(var)
  save('realValuesRegression_sparse2.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesSpatial1.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_spatial1.txt');
disp('Running: demo_spatial1')
demo_spatial1
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesSpatial1.mat', char(var(1)));
for i=2:length(var)
  save('realValuesSpatial1.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesSpatial2.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_spatial2.txt');
disp('Running: demo_spatial2')
demo_spatial2
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesSpatial2.mat', char(var(1)));
for i=2:length(var)
  save('realValuesSpatial2.mat', char(var(i)), '-append');
end
drawnow;clear;close all;

load realValuesSurvival_weibull.mat
var=who();
stream0 = RandStream('mt19937ar','Seed',0);
prevstream = RandStream.setDefaultStream(stream0);
diary('demo_survival_weibull.txt');
disp('Running: demo_survival_weibull')
demo_survival_weibull
fprintf('\n gp hyperparameters (gp_pak(gp)): \n \n') 
disp(gp_pak(gp))
diary off;
save('realValuesSurvival_weibull.mat', char(var(1)));
for i=2:length(var)
  save('realValuesSurvival_weibull.mat', char(var(i)), '-append');
end
drawnow;clear;close all;
