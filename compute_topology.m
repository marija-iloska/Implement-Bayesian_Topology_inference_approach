function [ln_prob] = compute_topology(dx, T, idx, m, n, g, x_samples, Sig, Sig_inv, var_x)


G = g(x_samples(idx,:,:));
U_inv = G(:,1:T-2,m) * G(:,1:T-2,m)'/var_x;
C_top = inv( Sig_inv(idx,idx) + U_inv);
C_top_inv = Sig_inv(idx,idx) + U_inv;

mu_new = C_top*G(:,1:T-2, m)* squeeze(x_samples(n, 2:T-1, m))'/var_x;

term1 = -0.5*(T-1)*log(2*pi*var_x) -0.5*dx*log(2*pi*det(Sig)) + 0.5*dx*log(2*pi*det(C_top));

term2 = -0.5*sum(x_samples(n,2:T-1,m).^2) + 0.5*mu_new'*C_top_inv*mu_new;

ln_prob = term1 + term2;

end