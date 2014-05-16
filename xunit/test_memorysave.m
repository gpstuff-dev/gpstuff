function test_suite = test_memorysave

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_MEMORYSAVE
%
% Copyright (c) 2011-2012 Ville Tolvanen

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

initTestSuite;


  function testDemo
    % Run demo
    disp('Running: demo_memorysave')
    demo_memorysave
    drawnow;clear;close all


