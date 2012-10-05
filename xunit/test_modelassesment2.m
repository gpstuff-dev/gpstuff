function test_suite = test_modelassesment2

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_MODELASSESMENT2

% Copyright (c) 2011-2012 Ville Tolvanen

initTestSuite;


  function testDemo
    % Set random number stream so that test failing isn't because randomness.
    % Run demo & save test values.
    stream0 = RandStream('mt19937ar','Seed',0);
    if str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
      prevstream = RandStream.setDefaultStream(stream0);
    else
      prevstream = RandStream.setGlobalStream(stream0);
    end
    
    disp('Running: demo_modelassesment2')
    demo_modelassesment2
    path = which('test_modelassesment2.m');
    path = strrep(path,'test_modelassesment2.m', 'testValues');
    if ~(exist(path, 'dir') == 7)
      mkdir(path)
    end
    path = strcat(path, '/testModelAssesment2');
    save(path, 'DIC', 'DIC2', 'DIC_latent', 'p_eff', ...
      'p_eff2', 'p_eff_latent', 'p_eff_latent2', 'mlpd_cv');
    
    % Set back initial random stream
    if str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
      RandStream.setDefaultStream(prevstream);
    else
      RandStream.setGlobalStream(prevstream);
    end
    drawnow;clear;close all
        
  % Compare test values to real values.

  function testDICParameters
    values.real = load('realValuesModelAssesment2.mat', 'DIC');
    values.test = load(strrep(which('test_modelassesment2.m'), 'test_modelassesment2.m', 'testValues/testModelAssesment2.mat'), 'DIC');
    assertVectorsAlmostEqual(values.real.DIC, values.test.DIC, 'relative', 0.05);

        
  function testDICAll
    values.real = load('realValuesModelAssesment2.mat', 'DIC2');
    values.test = load(strrep(which('test_modelassesment2.m'), 'test_modelassesment2.m', 'testValues/testModelAssesment2.mat'), 'DIC2');
    assertVectorsAlmostEqual(values.real.DIC2, values.test.DIC2, 'relative', 0.05);

        
  function testDICLatent
    values.real = load('realValuesModelAssesment2.mat', 'DIC_latent');
    values.test = load(strrep(which('test_modelassesment2.m'), 'test_modelassesment2.m', 'testValues/testModelAssesment2.mat'), 'DIC_latent');
    assertVectorsAlmostEqual(values.real.DIC_latent, values.test.DIC_latent, 'relative', 0.05);

        
  function testPeffLatentsMarginalized
    values.real = load('realValuesModelAssesment2.mat', 'p_eff');
    values.test = load(strrep(which('test_modelassesment2.m'), 'test_modelassesment2.m', 'testValues/testModelAssesment2.mat'), 'p_eff');
    assertVectorsAlmostEqual(values.real.p_eff, values.test.p_eff, 'relative', 0.4);        
        
        
	function testPeffLatent
    values.real = load('realValuesModelAssesment2.mat', 'p_eff_latent');
    values.test = load(strrep(which('test_modelassesment2.m'), 'test_modelassesment2.m', 'testValues/testModelAssesment2.mat'), 'p_eff_latent');
    assertVectorsAlmostEqual(values.real.p_eff_latent, values.test.p_eff_latent, 'relative', 0.05);
        
        
	function testPeffLatent2
    values.real = load('realValuesModelAssesment2.mat', 'p_eff_latent2');
    values.test = load(strrep(which('test_modelassesment2.m'), 'test_modelassesment2.m', 'testValues/testModelAssesment2.mat'), 'p_eff_latent2');
    assertVectorsAlmostEqual(values.real.p_eff_latent2, values.test.p_eff_latent2, 'relative', 0.25);
        
        
	function testLogPredDensity10foldCV
    values.real = load('realValuesModelAssesment2.mat', 'mlpd_cv');
    values.test = load(strrep(which('test_modelassesment2.m'), 'test_modelassesment2.m', 'testValues/testModelAssesment2.mat'), 'mlpd_cv');
    assertVectorsAlmostEqual(values.real.mlpd_cv, values.test.mlpd_cv, 'relative', 0.05);