x = [1 2 3 4]';
y = [1 2 3 4]' + [0.0589   -0.2672    0.1429    0.3247]'; %[0.0327    0.0175   -0.0187    0.0726]';

plot(x,y,'*')
hold on
plot(x,y)
axis([0 5 0 5])

[n, nin] = size(x);
gpcf1 = gpcf_sexp('init', nin, 'lengthScale', 1, 'magnSigma2', 1);
gpcf2 = gpcf_noise('init', nin, 'noiseSigmas2', 1);   % 0.1^2

% Set the prior for the parameters of covariance functions 
gpcf2.p.noiseSigmas2 = sinvchi2_p({0.05^2 0.5});    % MUUTA t�ss� invgam_p saman n�k�iseksi kuin gpcf_sexp('set'...)
gpcf1.p.lengthScale = gamma_p({3 7});  
gpcf1.p.magnSigma2 = sinvchi2_p({0.05^2 0.5});

% sparse model. Set the inducing points to the GP
gp = gp_init('init', 'PIC_BLOCK', nin, 'regr', {gpcf1}, {gpcf2});
%gp = gp_init('init', 'PIC_BLOCK', nin, 'regr', {gpcf1}, {gpcf2}, 'jitterSigmas', 0.0);

% Set the inducing inputs
u=[2 3]';
plot(u, [0.1 0.1], 'r*')

index = {};
index{1} = [1 2]';
index{2} = [3 4]';

%index{1} = 1;index{2} = 2;index{3} = 3;index{4} = 4;
gp = gp_init('set', gp, 'X_u', u, 'blocks', {'manual', x, index});


% Check the gradient for PIC
% ================================

mask_block = [1 1 0 0 ; 
              1 1 0 0 ;
              0 0 1 1 ;
              0 0 1 1 ];
%mask_block = eye(4,4);
gp.mask = mask_block;

[K_ff, C_ff] = gp_trcov(gp, x);
K_uu = gp_trcov(gp, gp.X_u);
K_fu = gp_cov(gp, x, gp.X_u);
Q_ff = K_fu*(K_uu\K_fu');


Labl = mask_block.*(C_ff-Q_ff);

%Labl = diag(diag((C_ff-Q_ff)));
A = K_uu + K_fu'*inv(Labl)*K_fu;

c = inv(chol(A));

L = inv(Labl)*K_fu*c;

inv(Labl) - L*L'

inv(Labl) + inv(Labl)*K_fu*(c*c')*K_fu'*inv(Labl)

inv(Labl) + inv(Labl)*K_fu*inv(A)*K_fu'*inv(Labl)
hold on
inv(Q_ff + Labl)

b = t'*inv(Q_ff + mask.*(C_ff-Q_ff));
inv(Q_ff + mask.*(Cbl_ff-Q_ff)) - mask.*(Cbl_ff-Q_ff)




dKuu_l = 2*K_uu.*bsxfun(@minus,u,u').^2./gp.cf{1}.lengthScale.^3
%dKuu_l = dKuu_l + dKuu_l' - diag(diag(dKuu_l));
dKuf_l = 2*K_fu'.*bsxfun(@minus,u,x').^2./gp.cf{1}.lengthScale.^3
dKff_l = 2*K_ff.*bsxfun(@minus,x,x').^2./gp.cf{1}.lengthScale.^3
%dKff_l = dKff_l + dKff_l' - diag(diag(dKff_l));

[g gdata gprior] = gp_g(gp_pak(gp,'hyper'), gp, x, y, 'hyper');

gradcheck(gp_pak(gp,'hyper'), @gp_e, @gp_g, gp, x, y, 'hyper')

gradcheck(gp_pak(gp,'all'), @gp_e, @gp_g, gp, x, y, 'all')
