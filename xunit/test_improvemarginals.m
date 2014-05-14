function test_suite = test_improvemarginals

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_IMPROVEMARGINALS
%
% Copyright (c) 2011-2012 Ville Tolvanen

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

initTestSuite;


  function testDemo
    % Set random number stream so that the test failing isn't because
    % randomness. Run demo & save test values.
    prevstream=setrandstream(0);    
    disp('Running: demo_improvemarginals')
    demo_improvemarginals
    path = which('test_improvemarginals');
    path = strrep(path,'test_improvemarginals.m', 'testValues');
    if ~(exist(path, 'dir') == 7)
      mkdir(path)
    end
    path = strcat(path, '/testImprovemarginals');
    save(path, 'pc_ep', 'fvec_ep', 'pc_ep_pred', 'fvec_ep_pred', ...
      'pc_la', 'fvec_la', 'pc_la_pred', 'fvec_la_pred', ...
      'pc_la2', 'fvec_la2', 'pc_la_pred2', 'fvec_la_pred2');
    
    % Set back initial random stream
    setrandstream(prevstream);
    drawnow;clear;close all


% Test the corrected distributions with 5% tolerance.
        
  function test_EP_fact
    values.real = load('realValuesImprovemarginals.mat','pc_ep','fvec_ep','pc_ep_pred','fvec_ep_pred');
    values.test = load(strrep(which('test_improvemarginals.m'), 'test_improvemarginals.m', 'testValues/testImprovemarginals.mat'),'pc_ep','fvec_ep','pc_ep_pred','fvec_ep_pred');
    assertElementsAlmostEqual((values.real.pc_ep), (values.test.pc_ep), 'relative', 0.05);
    assertElementsAlmostEqual([values.real.fvec_ep(1), values.real.fvec_ep(end)], [values.test.fvec_ep(1), values.test.fvec_ep(end)], 'relative', 0.05);
    assertElementsAlmostEqual((values.real.pc_ep_pred), (values.test.pc_ep_pred), 'relative', 0.05);
    assertElementsAlmostEqual([values.real.fvec_ep_pred(1), values.real.fvec_ep_pred(end)], [values.test.fvec_ep_pred(1), values.test.fvec_ep_pred(end)], 'relative', 0.05);


  function test_Laplace_CM2
    values.real = load('realValuesImprovemarginals.mat','pc_la','fvec_la','pc_la_pred','fvec_la_pred');
    values.test = load(strrep(which('test_improvemarginals.m'), 'test_improvemarginals.m', 'testValues/testImprovemarginals.mat'),'pc_la','fvec_la','pc_la_pred','fvec_la_pred');
    assertElementsAlmostEqual((values.real.pc_la), (values.test.pc_la), 'relative', 0.05);
    assertElementsAlmostEqual([values.real.fvec_la(1), values.real.fvec_la(end)], [values.test.fvec_la(1), values.test.fvec_la(end)], 'relative', 0.05);
    assertElementsAlmostEqual((values.real.pc_la_pred), (values.test.pc_la_pred), 'relative', 0.05);
    assertElementsAlmostEqual([values.real.fvec_la_pred(1), values.real.fvec_la_pred(end)], [values.test.fvec_la_pred(1), values.test.fvec_la_pred(end)], 'relative', 0.05);


  function test_Laplace_fact
    values.real = load('realValuesImprovemarginals.mat','pc_la2','fvec_la2','pc_la_pred2','fvec_la_pred2');
    values.test = load(strrep(which('test_improvemarginals.m'), 'test_improvemarginals.m', 'testValues/testImprovemarginals.mat'),'pc_la2','fvec_la2','pc_la_pred2','fvec_la_pred2');
    assertElementsAlmostEqual((values.real.pc_la2), (values.test.pc_la2), 'relative', 0.1);
    assertElementsAlmostEqual([values.real.fvec_la2(1), values.real.fvec_la2(end)], [values.test.fvec_la2(1), values.test.fvec_la2(end)], 'relative', 0.1);
    assertElementsAlmostEqual((values.real.pc_la_pred2), (values.test.pc_la_pred2), 'relative', 0.1);
    assertElementsAlmostEqual([values.real.fvec_la_pred2(1), values.real.fvec_la_pred2(end)], [values.test.fvec_la_pred2(1), values.test.fvec_la_pred2(end)], 'relative', 0.1);

