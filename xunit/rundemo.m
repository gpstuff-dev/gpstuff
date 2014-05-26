function rundemo(name, varToSave, mode)
%RUNDEMO  Run a GPstuff demo and save the results.
%
%   Description
%     RUNDEMO(name, varToSave, mode) runs the GPstuff demo given in name
%     and save given variables into the folder 'testValues' or 'realValues'.
%     This function is used by test_* to compare the results into the
%     precomputed expected ones.
%
%   Parameters:
%     name
%       The name of the demo without the .m extension or the demo_ prefix,
%       e.g. 'binomial1'.
%     varToSave (optional)
%       Cell array of strings indicates the variables that are saved into
%       the desired folder. Giving string 'all' saves the whole work
%       space. String 'same' (default) looks the names of the saved
%       variables in the file 'realValues/<nameOfTheDemo>.mat' and saves
%       them. Empty array indicates that nothig is saved.
%     mode (optional)
%       String 'test' (default) or 'real' indicating which folder the
%       results are saved.
%
%   See also
%     TEST_*, DEMO_*
%
% Copyright (c) 2014 Tuomas Sivula

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% Handle parameters
% Save to 'testValues' by default
if nargin < 3
  mode = 'test';
end
% If varToSave is not defined, save the same as before
if nargin < 2
  varToSave = 'same';
end

% Validate mode parameter
if ~(strcmp(mode, 'test') || strcmp(mode, 'real'))
  error('Mode has to be ''test'' or ''real''')
end

% Close all figures
close all;

% Save required variables into the structure run_demo_data
run_demo_data.name = name;
run_demo_data.varToSave = varToSave;
run_demo_data.mode = mode;
% Change working directory
fpath = mfilename('fullpath');
run_demo_data.origPath = cd(fpath(1:end-length(mfilename)));
% Try to avoid differences in the results by setting a common random stream
run_demo_data.origStream = setrandstream(0);
% Hide figure windows
run_demo_data.origFigVisibility = get(0,'DefaultFigureVisible');
set(0,'DefaultFigureVisible','off')
clear name varToSave fpath mode

% Create the folder if it does not exist
if ~(exist([run_demo_data.mode 'Values/'], 'dir') == 7)
  mkdir([run_demo_data.mode 'Values/'])
end

% Run the demo and create log into the desired directory
try
  
  fprintf('---- Run demo_%s %s\n\n', ...
    run_demo_data.name, repmat('-',1,79-15-length(run_demo_data.name)));
  
  % Create log, if diary is not currently running
  if strcmp(get(0,'Diary'), 'off')
    run_demo_data.diary = 1;
    if exist([run_demo_data.mode 'Values/' run_demo_data.name '.txt'] ,'file')
      delete([run_demo_data.mode 'Values/' run_demo_data.name '.txt']);
    end
    diary([run_demo_data.mode 'Values/' run_demo_data.name '.txt']);
  end
  
  % Run the demo and measure time
  run_demo_data.timer_id = tic;
  run(['demo_' run_demo_data.name])
  % If variable gp exist, display hypreparameters
  if exist('gp', 'var')
    fprintf('\n gp hyperparameters: \n \n')
    disp(gp_pak(gp))
  end
  fprintf('Demo completed in %.3f minutes\n', toc(run_demo_data.timer_id)/60)
  
  % Close diary if logged
  if isfield(run_demo_data, 'diary')
    diary off;
  end

catch err
  % Error running the demo
  ME = MException('run_demo:DemoFailure', 'Could not run demo_%s', run_demo_data.name);
  ME = addCause(ME, err);
  % Restore changes and raise error
  close all;
  cd(run_demo_data.origPath)
  setrandstream(run_demo_data.origStream);
  set(0,'DefaultFigureVisible',run_demo_data.origFigVisibility)
  if isfield(run_demo_data, 'diary')
    diary off;
  end
  throw(ME)
end

% Save the results into the desired directory
try
  
  % Save variables
  if ~isempty(run_demo_data.varToSave)
    if iscellstr(run_demo_data.varToSave)
      % Save given variables
      save([run_demo_data.mode 'Values/' run_demo_data.name], run_demo_data.varToSave{:})
    elseif ischar(run_demo_data.varToSave) && strcmp(run_demo_data.varToSave, 'same')
      % Save the same variables as in the 'realValues/<nameOfTheDemo>.mat'
      finfo = whos(matfile(['realValues/' run_demo_data.name '.mat']));
      if ~isempty(finfo)
        save([run_demo_data.mode 'Values/' run_demo_data.name], finfo.name)
      else
        warning(['File realValues/' run_demo_data.name '.mat not found' ...
          'or file has no variables.'])
      end
    elseif ischar(run_demo_data.varToSave) && strcmp(run_demo_data.varToSave, 'all')
      % Save all variables
      save([run_demo_data.mode 'Values/' run_demo_data.name])
    else
      warning('Unsupported parameter varToSave, no variables saved.')
    end
  end
  
  % Save all figures into the desired folder
  for i = get(0,'children')'
    filename = [run_demo_data.mode 'Values/' run_demo_data.name '_fig' num2str(i) '.fig'];
    % First save the figure hidden
    saveas(i,filename)
    % Then change the visible-flag on
    f=load(filename,'-mat');
    n=fieldnames(f);
    f.(n{1}).properties.Visible='on';
    save(filename,'-struct','f')
  end
  % Close all the 'hidden' figures
  close all;
  
catch err
  % Error saving the variables
  ME = MException('run_demo:DemoResultSaveFailure', 'Could not save the results');
  ME = addCause(ME, err);
  % Restore changes and raise error
  close all;
  cd(run_demo_data.origPath)
  setrandstream(run_demo_data.origStream);
  set(0,'DefaultFigureVisible',run_demo_data.origFigVisibility)
  throw(ME)
end

fprintf('Results saved into the folder ''xunit/%sValues/''\n\n', run_demo_data.mode)

% Restore changes
cd(run_demo_data.origPath)
setrandstream(run_demo_data.origStream);
set(0,'DefaultFigureVisible',run_demo_data.origFigVisibility)

