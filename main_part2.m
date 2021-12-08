clear all
close all
clc

% State equations
g = @(x) 1./(1 + exp(-x));
tr = @(coeff, x) coeff * g(x);
obs = @(coeff, x) coeff * x;

% State dimension
dx = 50;

% Time series length
T = 100;

% Conditions
var_x = 1;
var_y = 0.5;

% Sparsity
p_s = 0.3;

% Generate data
[A, C, x, y] = generate_states(T, dx, p_s, var_x, var_y, tr, obs);

M = 100;
N = dx;
I = 2;


% Initialize
x_samples = zeros(dx, T, M);
parfor m = 1:M
    for t = 1:2
        x_samples(:,t,m) = mvnrnd( y(:,t), var_x*eye(dx));
    end
end


psi = zeros(N,M);
lambda = zeros(N,M);

ln_alpha = zeros(1,M);
ln_beta = zeros(1,M);

Sig_inv = 1./50*eye(dx);
Sig = 50*eye(dx);
C_samples = zeros(dx,dx,M);
A_samples = ones(dx,dx,M);


% Start iterations
for i = 1 : I

    
    % Start MPF
    for t = 3 : T
        
        tic
        % Using N number of filters in parallel
        parfor n = 1 : N
           
            % Get proposal parameters
            [lambda(n,:), psi(n,:), C_temp] = optimal_proposal(dx, t, n, M, x_samples, A_samples, y, g, var_x, var_y, Sig_inv);

            % Propose samples
            x_temp(n,:) = mvnrnd(lambda(n,:), psi(n,:).*eye(M));
           

            % Store C_samples
            C_samples(n,:,:) = C_temp;

        end
        
        
        % Communicate states
        x_samples(:,t,:) = x_temp;

        % Computing weights using estimates of all N filters
        parfor n = 1 : N

            % Computing weights
            w = weights(t, M, n, y, x_samples, A_samples, C_samples, g, var_x, var_y, psi, lambda);

            % Resample
            idx_w = datasample(1:M, M, 'Weights', w);
            
            x_temp(n,:) = x_samples(n, t, idx_w);

            C_temp(n,:,:) = C_samples(n,:,idx_w);

        end

        % Set resampled states
        x_samples(:, t, :) = x_temp;
        C_samples = C_temp;

    end



    % Sample Topology
    for n = 1 : N
        for r = 1 : N

            for m = 1:M    

                % Computing alpha
                A_samples(n,r,m) = 1;
                idx = find(A_samples(n,:,m) == 1);
                ln_alpha(m) = compute_topology(dx, T, idx, m, n, g, x_samples, Sig, Sig_inv, var_x);

                % Computing beta
                A_samples(n,r,m) = 0;
                idx = find(A_samples(n,:,m) == 1);
                ln_beta(m) = compute_topology(dx, T, idx, m, n, g, x_samples, Sig, Sig_inv, var_x);


                % Scale Probabilities
                alpha(m) = exp(ln_alpha(m) - max([ln_alpha(m), ln_beta(m)]));
                beta(m) =  exp(ln_beta(m) - max([ln_alpha(m), ln_beta(m)]));
                

                % Sample
                prob = alpha(m)/(alpha(m) + beta(m));
                res = datasample([1, 0], 1, 'Weights', [prob, 1-prob]); 
                A_samples(n,r,m) = res;
            end


        end
    end

    parfor m = 1:M
        [~,~, fscore] = adj_eval(A, A_samples(:,:,m));
        fs(i,m) = fscore;
    end
    toc

end

% Get estimates
x_est = mean(x_samples, 3);
C_est = mean(C_samples, 3);
fs_est = mean(fs, 2);


figure(1);
n = datasample(1:dx, 1);
plot(x(n,:), 'k','LineWidth',2)
hold on
plot(x_est(n,:), 'b--','LineWidth',2)



figure(2)
r = datasample(1:dx, 1);
hist(squeeze(C_samples(n,r,:)))
hold on
scatter(C(n,r), 0, 100, 'r', 'filled')


save('dx50_5cagla05.mat', 'fs_est', 'x_est', 'C_est', 'C_samples', 'var_x', 'var_y','y','x','T', 'M','I');
