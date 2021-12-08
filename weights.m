function [w] = weights(t, M, n, y, x_samples, A_samples, C_samples, g, var_x, var_y, psi, lambda)

ln_w = zeros(1,M);

for m = 1:M

    x_sq = squeeze(x_samples(n,t,m))';

    idx = find(A_samples(n,:,m) == 1);
    G = g(x_samples(idx,:,:));

    ln_w(m) = -0.5*log(2*pi*var_x*var_y./psi(n,m)) - 0.5/var_y*( y(n,t) - x_sq ).^2 - ...
        0.5/var_x*(x_sq - C_samples(n,idx,m)*squeeze(G(:,t-1,m)) ).^2 + ...
        0.5./psi(n,m).*(x_sq - lambda(n,m)).^2;

end

% Scale
w_un = exp(ln_w - max(ln_w));

% Normalize
w = w_un./sum(w_un);


end
