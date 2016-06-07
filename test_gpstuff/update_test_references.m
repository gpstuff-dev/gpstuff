% Run selected demos, compare results from demos to old results and save
% any error messages.

demos={'demo_binomial1', ...
  'demo_classific', 'demo_derivativeobs',  ...
  'demo_multinom', 'demo_neuralnetcov', ...
  'demo_periodic', 'demo_regression1', 'demo_regression_additive1', ...
  'demo_regression_hier', 'demo_regression_sparse1', 'demo_survival_aft'};

iter=0;
failed=0;
path=strrep(which('run_tests'), 'run_tests.m', '');
for ii=1:length(demos)
  iter=iter+1;
  setrandstream(0);  
  if exist('OCTAVE_VERSION','builtin')
    values.real=load([path 'octave/realValues_' strrep(demos{ii}, 'demo_', '')]);
  else
    values.real=load([path 'matlab/realValues_' strrep(demos{ii}, 'demo_', '')]);
  end
  field=fieldnames(values.real);
  fprintf('\nRunning demo: %s\n\n', demos{ii});
  try
    eval(demos{ii});
    delete([path 'realValues_' strrep(demos{ii}, 'demo_', '') '.mat'])
    for jj=1:length(field)
      save('-append', [path 'realValues_' strrep(demos{ii}, 'demo_', '') '.mat'], field{jj})
    end
  catch err
    fprintf('Failed %s\n s%', demos{ii}, err)
  end
  close all;
end
fprintf('Results saved to %s.\n', path);
