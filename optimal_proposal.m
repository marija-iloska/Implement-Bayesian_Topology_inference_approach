function [lambda, psi, C_samples] = optimal_proposal(dx, t, n, M, x_samples, A_samples, y, g, var_x, var_y, Sig_inv)


eta_n = zeros(dx,M);
eksi_n = zeros(1,M);

% @ t - 1
for m = 1 : M

    % Find genes that are connected
    idx = find(A_samples(n,:,m) == 1);
    G = g(x_samples(idx,:,:));

    U_inv = G(:,1:t-2,m) * G(:,1:t-2,m)'/var_x;

    % C at t-1
    C_n = inv(Sig_inv(idx, idx) + U_inv);

    % eta at t-1
    eta_n(idx,m) = C_n*( G(:,1:t-2,m)*squeeze(x_samples(n, 2:t-1, m))'/var_x );

    % eksi at t
    eksi_n(m) = var_x + G(:,t-1,m)'*C_n*G(:,t-1,m);

end

% nu at t
nu_n = sum(eta_n(idx,:).*squeeze(G(:,t-1,:)), 1);

% Proposal parameters at t
psi = var_y*eksi_n./(var_y + eksi_n);

lambda = eksi_n*y(n,t)./(eksi_n + var_y) + ...
    var_y*nu_n./(eksi_n + var_y);


% Store C_samples
C_samples = eta_n.*squeeze(A_samples(n,:,:));

end