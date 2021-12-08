function [A, C, x, y] = generate_states(T, dx, p_s, var_x, var_y, tr, obs)


% Initialize coefficient and aadjacency matrices
%C = rand(dx, dx); % - 0.25;
C = unifrnd(-5, 5 , dx, dx);
A = ones(dx, dx);


for j = 1 : dx
    idx = datasample(1:dx, round(p_s*dx));
    A(j,idx) = 0;
end

C = C.*A;


% C = [3, 0, 0, -4.5 ; -2.9, 0 5 0; -6 4 0 0; 0 -3 2 0];

%  C = [0, 0, 0, 0.6, 0.7, 0, 1.9, 2.9; -0.1, 0, 0, 3.5, 0, -2.1, 0, 3.4; ...
%      -4.4, 0.9, -1.7, -0.3, 3.4, 0, 1.7, 0; 0, 0.5, 2.8, -3.7, 0.9, 0, 0, -3.1; ...
%      0, 0.2, 0, -2.6, -3.2, -0.1, -0.5, 4;  -0.5, -1.8, 0, 3.4, 1.4, 1.1, 0, -1.7; ...
%      -0.8, 0, 0, -3, 1.1, 0.4, 0, 0;  -0.3, 0, -1, 0, 0.1, 0, 0, 2.2];


%A = (C~=0);


% Generate the data
x(:,1) = rand(dx, 1);
y(:,1) = x(:,1) + mvnrnd(zeros(dx,1), var_y*eye(dx))';

for t = 2:T
    x(:,t) = tr(C, x(:,t-1)) + mvnrnd(zeros(dx,1), var_x*eye(dx))';
    y(:,t) = obs(1, x(:,t)) + mvnrnd(zeros(dx,1), var_y*eye(dx))';       
end

end
