% Run selected demos, compare results from demos to old results and save
% any error messages.

demos={'demo_binomial1', ...
  'demo_classific', 'demo_derivativeobs',  ...
  'demo_multinom', 'demo_neuralnetcov', ...
  'demo_periodic', 'demo_regression1', 'demo_regression_additive1', ...
  'demo_regression_hier', 'demo_regression_sparse1', 'demo_survival_aft'};

invalid_demos={};
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
    for j=1:length(field)
      iter=iter+1;
      if mean(mean((abs(getfield(values.real, field{j}) - eval(field{j})))./getfield(values.real, field{j})))>1e-2
        failed=failed+1;
        invalid_demos(failed).name=demos{ii};
        invalid_demos(failed).paramName=field{j};
        invalid_demos(failed).paramValue_old=getfield(values.real, field{j});
        invalid_demos(failed).paramValue_new=eval(field{j});
        invalid_demos(failed).error='New values dont match with old saved values with relative tolerance of 1e-2.';
      end
    end
  catch err
    % Error while running a demo
    iter=iter+length(field);
    failed=failed+1;
    invalid_demos(failed).name=demos{ii};
    invalid_demos(failed).paramName='';
    invalid_demos(failed).paramValue_old=[];
    invalid_demos(failed).paramValue_new=[];
    % Save error structure to field error.
    invalid_demos(failed).error=err;    
    for j=1:length(field)
      failed=failed+1;
      invalid_demos(failed).name=demos{ii};
      invalid_demos(failed).paramName=field{j};
      invalid_demos(failed).paramValue_old=getfield(values.real, field{j});
      invalid_demos(failed).paramValue_new=[];
      invalid_demos(failed).error='Error while runnig the demo.';
    end
  end
  close all;
end
invalid_demos
fprintf('Failed %d of %d tests. Check struct invalid_demos for further details.\n',failed, iter);