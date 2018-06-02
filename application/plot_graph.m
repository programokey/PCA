x = rand(50, 1)*6 + 2;
y = x*0.4 + 3 + rand(size(x));
scatter(x, y, "filled", "r")
hold
annotation('arrow',[5, 3],[2.5, 4]) 
annotation('arrow',[5, 3],[2.5, 6]) 
axis([[0, 10],[0, 10]])


