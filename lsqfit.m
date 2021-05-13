function [W, out2, out3, HW] = lsqfit(problem, U, in3, in4)
% [W, COMPUMEM] = LSQFIT(PROBLEM, U)
% [W, COMPUMEM] = LSQFIT(PROBLEM, U, COMPUMEM)
% [W, DW, COMPUMEM, HW] = LSQFIT(PROBLEM, U, H)
% [W, DW, COMPUMEM, HW] = LSQFIT(PROBLEM, U, H, COMPUMEM)
%
% Given a low-rank matrix completion problem structure PROBLEM and a guess
% at the column space U, returns a matrix W such that U*W is as close as
% possible, in the least-squares sense, to the matrix to be recovered.
%
% If H is provided, DW is returned and is the directional derivative
% of the mapping U -> W along H.
%
% COMPUMEM is an optional structure in which common computations between
% LSQFIT and RTRMC will be stored so as to reduce the amount of
% redundant computations.
%
% The algorithm implemented here is described in the following paper:
% 
% Nicolas Boumal and Pierre-Antoine Absil,
% RTRMC: A Riemannian trust-region method for low-rank matrix completion,
% Accepted at NIPS 2011, Granada (Spain), Dec. 2011.
%
% Nicolas Boumal, UCLouvain, May 19, 2011.
% http://perso.uclouvain.be/nicolas.boumal/RTRMC/
%
% SEE ALSO: buildproblem rtrmc

% Modified May 13, 2021 to return HW rather than save it in compumem.
    
    Hprovided = false;
    
    if nargin == 2
        compumem = struct();
    elseif nargin == 3
        % the third argument is either a compumem structure or a direction
        % H for the Hessian computation
        if isstruct(in3)
            compumem = in3;
        else
            H = in3;
            Hprovided = true;
            compumem = struct();
        end
    elseif nargin == 4
        H = in3;
        compumem = in4;
        Hprovided = true;
    else
        error('Wrong number of input parameters (need 2, 3 or 4)');
    end
    
    
	X = problem.X;
    C = problem.C;
    % m = problem.m;
    % n = problem.n;
    I = problem.I;
    J = problem.J;
    mask = problem.mask;
    Chat = problem.Chat;
	lambda = problem.lambda;
    
    % Factorization of the blocks of the block-diagonal, symmetric,
    % positive definite matrix A.
    if ~isfield(compumem, 'Achol')
        compumem.Achol = buildmatrix(problem, U);
    end
    Achol = compumem.Achol;

    % Solve LSQ system block by block.
    if ~isfield(compumem, 'W')
        rhs = multfullsparse(U.', (C.^2.*X), mask);
        compumem.W = blocksolve(Achol, rhs);
    end
    W = compumem.W;
    
    % Compute the directional derivative if it is required too.
    if Hprovided
        
        if ~isfield(compumem, 'UW')
            compumem.UW = spmaskmult(U, W, I, J);
        end
        UW = compumem.UW;
        
        if ~isfield(compumem, 'RU')
            compumem.RU = Chat.*(UW-X) - (lambda.^2)*X;
        end
        RU = compumem.RU;
        
        % Compumem shouldn't hold information relative to H, so we do not
        % store HW: we return it as an extra output.
        % It's possible that HW is computed more than once with the same H
        % and W: if so, it would be a point to improve.
        % (Comment and changes: May 13, 2021.)
        HW = spmaskmult(H, W, I, J);
        
        Q1 = multfullsparse(H.', RU, mask);
        Q2 = multfullsparse(U.', Chat.*HW, mask);
		
        dW = -blocksolve(Achol, Q1 + Q2);
        
        out2 = dW;
        out3 = compumem;
        
    else
        
        out2 = compumem;
        
    end
    
end

function Achol = buildmatrix(problem, U)
% Returns the Cholesky factors of the diagonal blocks of the symmetric,
% positive definite matrix A in a cell array.
% A is defined in the accompanying paper.
    
    Chat = problem.Chat;
    lambda = problem.lambda;
    % I = problem.I;
    % J = problem.J;
    % m = problem.m;
    % n = problem.n;
    
    % Achol = buildAchol(Chat, U, lambda.^2, I, J, uint32(n));

    mask = problem.mask;
    setsparseentries(mask, sqrt(Chat));
    
    Achol = spbuildmatrix(mask, U, lambda^2);
    
end

function X = blocksolve(R, B)
% R is a cell containing n upper triangular square matrices of size r.
% B is a matrix of size r x n. X is a matrix of size r x n where the i-th
% column is the solution of the equation A{i}x = B(:, i), with
% A{i} = R{i}.'*R{i}. This is: the R{i} matrices are the upper triangular
% Cholesky factors of the matrices A{i} (which we do not require/store).

    X = cholsolvecell(R, B);
    
end
