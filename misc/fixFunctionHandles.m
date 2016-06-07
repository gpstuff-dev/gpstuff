function gpo = fixFunctionHandles(gpo)
%FIXFUNCTIONHANDLES
%
%  Syntax:
%    gpo = fixFunctionHandles(gpo);
%  
%  Description:
%    Fix the function handles that are broken when  
%	 when octatve saves and loads gp structures
%
% Copyright (c) 2016 Markus Paasiniemi
%
% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

% list all subfields of the struct gpo
gpFields = fieldnamesRec(gpo); 
gpFields = cellstr(horzcat(repmat(['gpo.'],length(gpFields),1),char(gpFields)));

%Fix all function handles for all execpt *_rec
for k = 1:length(gpFields)
    
    %check if gpFields{k} is a function/subfunction handle
    if strcmp(typeinfo(eval(gpFields{k})),'function handle')
    	
    	spl2 = strsplit(functions(eval(gpFields{k})).file,{'\' '/','.'});
        fn = func2str(eval(gpFields{k}));
        sfn = strsplit(gpFields{k},'.'){end};
        
        %check if gpFields{k} is a function handle	
        if strcmp(spl2{end-1},fn) 
        	eval(strcat(gpFields{k},'=@',fn,';'));
        
        elseif strcmp(strsplit(gpFields{k},'.'){end},'ne')
        	eval(strcat(gpFields{k},'=',spl2{end-1},'("init",gpo).fh.ne;'));
        
        %gpFields{k} is a subfunction handle
        elseif isfield(eval(spl2{end-1}).fh,sfn) 
				eval(strcat(gpFields{k},'=',spl2{end-1},'.fh.',sfn,';'));            
		
		%gpFields{k} is a subfunction handle of type *_rec
		else
			eval(strcat(gpFields{k},'=',spl2{end-1},'.fh.recappend(',spl2{end-1},',',spl2{end-1},').fh.',sfn,';'));
		end
	end
end


% Helper function to recurse through 
% a struct and find all subfields
    function fNames = fieldnamesRec(s)
        
        fNames = fieldnames(s);
        for i = 1:length(fNames)
            
            if isstruct(s.(fNames{i}))
                newFields = fieldnamesRec(s.(fNames{i}));
                fNames(end+1:end+length(newFields)) = cellstr(horzcat(repmat([fNames{i} '.'], length(newFields), 1),char(newFields)));
            
            elseif iscell(s.(fNames{i}))
                for j = 1:length(s.(fNames{i}))
                
                    if isstruct(s.(fNames{i}){j})
                        newFields = fieldnamesRec(s.(fNames{i}){j});
                        fNames(end+1:end+length(newFields)) = cellstr(horzcat(repmat([fNames{i} '{' num2str(j) '}.'],length(newFields), 1),char(newFields)));
                    end
                end
            end
        end
    end

end