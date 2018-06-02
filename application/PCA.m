function [w, lambda, T]= PCA(X):
	X -= mean(X, 1)
  XTX = X'*X
  [w,lambda] = eig(XTX)
  
end
