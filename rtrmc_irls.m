function [U, W, allstats] = rtrmc_irls(problem, opts, U0)

    % Smoothing parameter for IRLS and number of RTRMC calls allowed.
    epsilon = 0.01;
    n_irls_iter = 5;

    % Gather values from the problem description
    I = problem.I;
    J = problem.J;
	X = problem.X;
    C = problem.C;
    m = problem.m;
    n = problem.n;
    r = problem.r;
	lambda = problem.lambda;
    
    allstats = cell(n_irls_iter, 1);

    % Pick the standard initial guess if none is given.
    if nargin < 3 || isempty(U0)
        U0 = initialguess(problem);
    end

    % Solve the problem a first time
    [U, W, stats] = rtrmc(problem, opts, U0);
    allstats{1} = stats;
    disp([stats.RMSE]);

    for irls_iter = 2 : n_irls_iter
    
        UW = spmaskmult(U, W, I, J);
        newC = max(lambda, C./sqrt(abs(UW-X)+epsilon));
        newproblem = buildproblem(I, J, X, newC, m, n, r, lambda);
        
        % Copy the data for RMSE computation
        newproblem.Xmean = problem.Xmean;
        newproblem.Xtest = problem.Xtest;
        newproblem.Itest = problem.Itest;
        newproblem.Jtest = problem.Jtest;
%         newproblem.A = problem.A;
%         newproblem.B = problem.B;
        
        [U, W, stats] = rtrmc(newproblem, opts, U);
        allstats{irls_iter} = stats;
        disp([stats.RMSE]);
        
    end

end
