function rundemo(name, varToSave, mode)
%RUNDEMO  Run a GPstuff demo and save the results.
%
%   Description
%     RUNDEMO(name, varToSave, mode) runs the GPstuff demo given in name
%     and save given variables into the folder 'testValues' or 
%     'realValues'. This function is used by test_* to compare the results
%     into the precomputed expected ones.
%     
%     The saved components are:
%       - desired workspace variables into <name>.mat
%       - command line output log into <name>.txt
%       - new created figures into <name>_fig#.fig
%     If diary is on before this function is called, it is left untouched
%     and the logfile is not created. Figures created before this function
%     are not touched.
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
%       them. Empty array indicates that nothig is saved. A function can be
%       applied into the variable before saving by providing a cell array
%       containing the name of the variable and the function handle (in
%       that order) in place of the name string in the cell array varToSave
%       e.g. varToSave = {'var1' {'var2' @(x)diag(x)} 'var3'} saves
%       variables var1, var2 = diag(var2) and var3. By providing a third
%       cell array string element, the name of the saved variable can be
%       changed, e.g. {'varOut' @(x)diag(x), 'varOrig'} saves variable
%       varOut = diag(varOrig).
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

% Save required variables into the structure run_demo_data
run_demo_data.name = name;
run_demo_data.varToSave = varToSave;
run_demo_data.mode = mode;
% Store original figures
run_demo_data.orig_figs =  get(0,'children');
% Change working directory
fpath = mfilename('fullpath');
run_demo_data.origPath = cd(fpath(1:end-length(mfilename)));
% Try to avoid differences in the results by setting a common random stream
run_demo_data.origStream = setrandstream(0);
% Hide figure windows
run_demo_data.origFigVisibility = get(0,'DefaultFigureVisible');
set(0,'DefaultFigureVisible','off')
% Check if diay is already used (no logging then)
if strcmp(get(0,'Diary'), 'off')
  run_demo_data.diary = 1;
else
  run_demo_data.diary = 0;
end
clear name varToSave fpath mode

% Create cleanup function to restore changes
run_demo_cleanupObj = onCleanup(@() restoreChanges(run_demo_data));

% Create the folder if it does not exist
if ~(exist([run_demo_data.mode 'Values/'], 'dir') == 7)
  mkdir([run_demo_data.mode 'Values/'])
end

% Run the demo and create log into the desired directory
try
  
  fprintf('---- Run demo_%s %s\n\n', ...
    run_demo_data.name, repmat('-',1,79-15-length(run_demo_data.name)));
  
  % Create log, if diary is not currently running
  if run_demo_data.diary
    if exist([run_demo_data.mode 'Values/' run_demo_data.name '.txt'] ...
        ,'file')
      delete([run_demo_data.mode 'Values/' run_demo_data.name '.txt']);
    end
    diary([run_demo_data.mode 'Values/' run_demo_data.name '.txt']);
  end
  
  % Run the demo and measure time
  run_demo_data.timer_id = tic;
  run(['demo_' run_demo_data.name])
  % If variable gp exist, display hypreparameters
  if exist('gp', 'var')
    run_demo_data.w = gp_pak(gp);
    if numel(run_demo_data.w) <= 10
      fprintf('\n gp hyperparameters: \n \n')
      disp(run_demo_data.w)
    end
  end
  fprintf('Demo completed in %.3f minutes\n', ...
    toc(run_demo_data.timer_id)/60)
  
  % Close diary at this point (if logged)
  if run_demo_data.diary
    diary off;
  end

catch err
  % Error running the demo
  % Here the exception could be included as a cause to a new exeption but
  % that makes the information related to this exeption harder to see.
  % ME = MException('run_demo:DemoFailure', 'Could not run demo_%s', ...
  %   run_demo_data.name);
  % ME = addCause(ME, err);
  % throw(ME)
  rethrow(err)
end

% Save the results into the desired directory
try
  
  % Save variables
  if ~isempty(run_demo_data.varToSave)
    if iscell(run_demo_data.varToSave)
      % Apply functions
      broken = 0;
      for i = 1:length(run_demo_data.varToSave)
        if iscell(run_demo_data.varToSave{i}) ...
            && length(run_demo_data.varToSave{i}) >= 2 ...
            && ischar(run_demo_data.varToSave{i}{1}) ...
            && isvarname(run_demo_data.varToSave{i}{1}) ...
            && isa(run_demo_data.varToSave{i}{2}, 'function_handle') ...
            && (length(run_demo_data.varToSave{i}) == 2 ...
            || (length(run_demo_data.varToSave{i}) == 3 ...
            && ischar(run_demo_data.varToSave{i}{3}) ...
            && isvarname(run_demo_data.varToSave{i}{3})) )
          if length(run_demo_data.varToSave{i}) == 2
            if exist(run_demo_data.varToSave{i}{1}, 'var')
              eval(sprintf('%s = run_demo_data.varToSave{i}{2}(%s);', ...
                run_demo_data.varToSave{i}{1}, ...
                run_demo_data.varToSave{i}{1}));
            else
              error('Variable %s not found', run_demo_data.varToSave{i}{1})
            end
          else
            if exist(run_demo_data.varToSave{i}{3}, 'var')
              eval(sprintf('%s = run_demo_data.varToSave{i}{2}(%s);', ...
                run_demo_data.varToSave{i}{1}, ...
                run_demo_data.varToSave{i}{3}));
            else
              error('Variable %s not found', run_demo_data.varToSave{i}{3})
            end
          end
          run_demo_data.varToSave{i} = run_demo_data.varToSave{i}{1};
        elseif ~ischar(run_demo_data.varToSave{i})
          warning('Unsupported parameter varToSave, no variables saved.')
          broken = 1;
          break
        end
      end
      if ~broken
        % Save given variables
        save([run_demo_data.mode 'Values/' run_demo_data.name], ...
          run_demo_data.varToSave{:})
      end
    elseif ischar(run_demo_data.varToSave) ...
        && strcmp(run_demo_data.varToSave, 'same')
      % Save the same variables as in the 'realValues/<nameOfTheDemo>.mat'
      finfo = whos(matfile(['realValues/' run_demo_data.name '.mat']));
      if ~isempty(finfo)
        save([run_demo_data.mode 'Values/' run_demo_data.name], finfo.name)
      else
        warning(['File realValues/' run_demo_data.name '.mat not found' ...
          'or file has no variables.'])
      end
    elseif ischar(run_demo_data.varToSave) ...
        && strcmp(run_demo_data.varToSave, 'all')
      % Save all variables
      save([run_demo_data.mode 'Values/' run_demo_data.name])
    else
      warning('Unsupported parameter varToSave, no variables saved.')
    end
  end
  
  % Save all figures into the desired folder
  for i = setdiff(get(0,'children'), run_demo_data.orig_figs)'
    if isprop(i, 'Number')
      filename = [run_demo_data.mode 'Values/' ...
        run_demo_data.name '_fig' num2str(i.Number) '.fig'];
    else
      filename = [run_demo_data.mode 'Values/' ...
        run_demo_data.name '_fig' num2str(i) '.fig'];
    end
    % First save the figure hidden
    saveas(i,filename)
    % Then change the visible-flag on
    % Note that the field hgM_070000 present in some newer Matlab versions
    % is discarded.
    f=load(filename,'-mat','hgS_070000');
    n=fieldnames(f);
    f.(n{1}).properties.Visible='on';
    save(filename,'-struct','f')
  end
  
catch err
  % Error saving the variables
  ME = MException('run_demo:DemoResultSaveFailure', ...
    'Could not save the results');
  ME = addCause(ME, err);
  throw(ME)
end

fprintf('Results saved into the folder ''tests/%sValues/''\n\n', ...
  run_demo_data.mode)

end


function restoreChanges(run_demo_data)
  % Ensure that changes are restored when exiting the function
  
  % Close all created 'hidden' figures
  for i = setdiff(get(0,'children'), run_demo_data.orig_figs)'
    close(i);
  end
  % Change directory
  cd(run_demo_data.origPath)
  % Set back previous random stream
  setrandstream(run_demo_data.origStream);
  % Set back original figure visibility
  set(0,'DefaultFigureVisible',run_demo_data.origFigVisibility)
  % Ensure that diary is closed (if used)
  if run_demo_data.diary
    diary off;
  end
end
