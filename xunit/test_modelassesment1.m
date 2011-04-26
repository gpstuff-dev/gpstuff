function test_suite = test_modelassesment1
initTestSuite;

% Set random number stream so that test failing isn't because randomness.
% Run demo % save test values.
    
    function testDemo
        stream0 = RandStream('mt19937ar','Seed',0);
        prevstream = RandStream.setDefaultStream(stream0);
        disp('Running: demo_modelassesment1')
        demo_modelassesment1
        path = which('test_modelassesment1.m');
        path = strrep(path,'test_modelassesment1.m', 'testValues/testModelAssesment1');
        save(path, 'DIC', 'DIC2', 'DIC_latent', ...
            'p_eff', 'p_eff2', 'p_eff_latent', 'p_eff_latent2', 'mlpd_cv', 'mrmse_cv');
        RandStream.setDefaultStream(prevstream);
        drawnow;clear;close all
        
 % Compare test values to real values.

    function testDICParameters
        values.real = load('realValuesModelAssesment1.mat','DIC');
        values.test = load('testValues/testModelAssesment1.mat','DIC');
        assertVectorsAlmostEqual(values.real.DIC, values.test.DIC, 'relative', 0.20);
        
        
    function testDICAll
        values.real = load('realValuesModelAssesment1.mat','DIC2');
        values.test = load('testValues/testModelAssesment1.mat','DIC2');
        assertVectorsAlmostEqual(values.real.DIC2, values.test.DIC2, 'relative', 0.20);
        
        
    function testDICLatent
        values.real = load('realValuesModelAssesment1.mat','DIC_latent');
        values.test = load('testValues/testModelAssesment1.mat','DIC_latent');
        assertVectorsAlmostEqual(values.real.DIC_latent, values.test.DIC_latent, 'relative', 0.20);
        
        
     function testPeffLatentMarginalized
        values.real = load('realValuesModelAssesment1.mat','p_eff');
        values.test = load('testValues/testModelAssesment1.mat','p_eff');
        assertVectorsAlmostEqual(values.real.p_eff, values.test.p_eff, 'relative', 0.20);
        

     function testPeffAll
        values.real = load('realValuesModelAssesment1.mat','p_eff2');
        values.test = load('testValues/testModelAssesment1.mat','p_eff2');
        assertVectorsAlmostEqual(values.real.p_eff2, values.test.p_eff2, 'relative', 0.20);      
        
        
     function testPeffLatent
        values.real = load('realValuesModelAssesment1.mat','p_eff_latent');
        values.test = load('testValues/testModelAssesment1.mat','p_eff_latent');
        assertVectorsAlmostEqual(values.real.p_eff_latent, values.test.p_eff_latent, 'relative', 0.20);
        
        
     function testPeffLatent2
        values.real = load('realValuesModelAssesment1.mat','p_eff_latent2');
        values.test = load('testValues/testModelAssesment1.mat','p_eff_latent2');
        assertVectorsAlmostEqual(values.real.p_eff_latent2, values.test.p_eff_latent2, 'relative', 0.20);
        
     
     function testLogPredDensity10foldCV
        values.real = load('realValuesModelAssesment1.mat', 'mlpd_cv');
        values.test = load('testValues/testModelAssesment1.mat', 'mlpd_cv');
        assertVectorsAlmostEqual(values.real.mlpd_cv, values.test.mlpd_cv, 'relative', 0.20);
        
        
     function testMeanSquaredError10foldCV
        values.real = load('realValuesModelAssesment1.mat', 'mrmse_cv');
        values.test = load('testValues/testModelAssesment1.mat', 'mrmse_cv');
        assertVectorsAlmostEqual(values.real.mrmse_cv, values.test.mrmse_cv, 'relative', 0.20);            
             
             
             