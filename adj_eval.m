function [precision, recall, fscore] = adj_eval(A, A_hat)
precision = sum(sum(A_hat.*A))/sum(sum(A_hat==1));
recall = sum(sum(A_hat.*A))/(sum(sum(A==1)));
fscore = 2*(precision*recall/(precision+recall));
end

