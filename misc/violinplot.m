function hh = violinplot(x,y,varargin)
%VIOLINPLOT Plot a vertical violinplot
%   
%  Description
%    VIOLINPLOT(X,Y,OPTIONS) Plot a violinplot of the data Y at
%      location given by X. X is 1xM vector and Y is NxM matrix. 
%      For each column of X and Y, one violinplot and median line
%      presenting the estimated distribution of Y is plotted. 
%      The density estimate is made using LGPDENS.
%  
%    H=VIOLINPLOT(X,Y,OPTIONS) returns graphics handles.
%
%    OPTIONS is optional parameter-value pair
%      color  - color used can be given as character or RGB value vector
%               color is used for edgecolor and lighter version for facecolor 
%
  
% Copyright (c) 2011 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.
  
  ip=inputParser;
  ip.FunctionName = 'VIOLINPLOT';
  ip.addRequired('x', @(x) isnumeric(x) && isvector(x));
  ip.addRequired('y', @(x) isnumeric(x) && size(x,1)>1);
  ip.addParamValue('color','k', @(x) ischar(x) || ...
                   (isnumeric(x) & isequal(size(x),[1 3])));
  ip.parse(x,y,varargin{:});
  x=ip.Results.x;
  y=ip.Results.y;
  color=ip.Results.color;

  [n,m]=size(y);
  for i1=1:m
    [p,~,yy]=lgpdens(y(:,i1),'gridn',200,...
                     'range',[mean(y(:,i1))-3*std(y(:,i1)) mean(y(:,i1))+3*std(y(:,i1))]);
    cp=cumsum(p./sum(p));
    qi05=binsgeq(cp,0.005);
    qi995=binsgeq(cp,0.995);
    p([1:qi05 qi995:end])=[];
    yy([1:qi05 qi995:end])=[];
    cp=cumsum(p./sum(p));
    qi5=binsgeq(cp,0.05);
    qi50=binsgeq(cp,0.5);
    qi95=binsgeq(cp,0.95);
    p=p./max(p)/5;
    % if color was character, this will be rgb vector
    hp(1,i1)=patch(x(i1)+[p; -p(end:-1:1)],[yy; yy(end:-1:1)],color);
    if i1==1
      color=get(hp(1,i1),'facecolor');
    end
    set(hp(1,i1),'edgecolor',color);
    set(hp(1,i1),'facecolor',color.*0.2+[.8 .8 .8])
    % median line
    h(1,i1)=line([x(i1)-p(qi50) x(i1)+p(qi50)],[yy(qi50) yy(qi50)],'color',color,'linewidth',1);
    % 5% line
    h(1,i1)=line([x(i1)-p(qi5) x(i1)+p(qi5)],[yy(qi5) yy(qi5)],'color',color,'linewidth',1);
    % 95% line
    h(1,i1)=line([x(i1)-p(qi95) x(i1)+p(qi95)],[yy(qi95) yy(qi95)],'color',color,'linewidth',1);
    % 90% interval
    %h(1,i1)=line([x(i1) x(i1)],[yy(qi5) yy(qi95)],'color',color,'linewidth',1);
  end
  if nargout>0
    hh=[hp; h];
    hh=hh(:);
  end
  