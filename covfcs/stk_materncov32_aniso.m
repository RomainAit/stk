% STK_MATERNCOV32_ANISO computes the anisotropic Matern covariance with nu=3/2
%
% CALL: k = stk_materncov32_aniso(param, x, y, diff)
%   param  = vector of parameters of size 1+d
%   x      = structure whose field 'a' contains the observed points.
%            x.a  is a matrix of size n x d, where n is the number of
%            points and d is the dimension of the factor space
%   y      = same as x
%   diff   = differentiation parameter
%
% STK_MATERNCOV32_ANISO computes a Matern covariance between two random vectors
% specified by the locations of the observations. This anisotropic
% covariance function has 2+d parameters, where d is the dimension of the
% factor space. They are defined as follows:
%    param(1)   = log(sigma^2) is the logarithm of the variance
%    param(1+i) = -log(rho(i)) is  the logarithm of  the inverse  of  the
%                 i^th range parameter
%
% If diff ~= -1, the function returns the derivative of the covariance wrt
% param(diff)

% Copyright Notice
%
%    Copyright (C) 2011, 2012 SUPELEC
%
%    Authors:   Julien Bect       <julien.bect@supelec.fr>
%               Emmanuel Vazquez  <emmanuel.vazquez@supelec.fr>
%
% Copying Permission Statement
%
%    This file is part of
%
%            STK: a Small (Matlab/Octave) Toolbox for Kriging
%               (http://sourceforge.net/projects/kriging)
%
%    STK is free software: you can redistribute it and/or modify it under
%    the terms of the GNU General Public License as published by the Free
%    Software Foundation,  either version 3  of the License, or  (at your
%    option) any later version.
%
%    STK is distributed  in the hope that it will  be useful, but WITHOUT
%    ANY WARRANTY;  without even the implied  warranty of MERCHANTABILITY
%    or FITNESS  FOR A  PARTICULAR PURPOSE.  See  the GNU  General Public
%    License for more details.
%
%    You should  have received a copy  of the GNU  General Public License
%    along with STK.  If not, see <http://www.gnu.org/licenses/>.

function k = stk_materncov32_aniso(param, x, y, diff)

persistent x0 y0 xs ys param0 D Kx_cache compute_Kx_cache

stk_narginchk(3, 4);

% default: compute the value (not a derivative)
if (nargin<4), diff = -1; end

% check consistency for the number of factors
dim = size(x.a, 2);
if (size(y.a, 2) ~= dim),
    stk_error('xi.a and yi.a have incompatible sizes.', 'InvalidArgument');
end
nb_params = dim + 1;
if (numel(param) ~= nb_params)
    stk_error('xi.a and param have incompatible sizes.', 'InvalidArgument');
end

% extract parameters from the "param" vector
Sigma2 = exp(param(1));
invRho = exp(param(2:end));

% check parameter values
if ~(Sigma2>0) || ~all(invRho>0),
    error('Incorrect parameter value.');
end

invRho = diag(invRho);

% check if all input arguments are the same as before
% (or if this is the first call to the function)
if isempty(x0) || isempty(y0) || isempty(param0) || ...
        ~isequal({x.a,y.a,param},{x0.a,y0.a,param0})
    % compute the distance matrix
    xs = x.a * invRho; ys = y.a * invRho;
    D = stk_distance_matrix(xs, ys);
    % save arguments for the next call
    x0 = x; y0 = y; param0 = param;
    % recomputation of Kx_cache is required
    compute_Kx_cache = true;
end

if diff == -1,
    %%% compute the value (not a derivative)
    k = Sigma2 * stk_sf_matern32(D, -1);
elseif diff == 1,
    %%% diff wrt param(1) = log(Sigma2)
    k = Sigma2 * stk_sf_matern32(D, -1);
elseif (diff >= 2) && (diff <= nb_params),
    %%% diff wrt param(diff) = - log(invRho(diff-1))
    ind = diff - 1;
    if compute_Kx_cache || isempty(Kx_cache)
        Kx_cache  = 1./(D+eps) .* (Sigma2 * stk_sf_matern32(D, 1));
        compute_Kx_cache = false;
    end
    nx = size(x.a,1); ny = size(y.a,1);
    k = (repmat(xs(:,ind),1,ny) - repmat(ys(:,ind)',nx,1)).^2 .* Kx_cache;
else
    stk_error('Incorrect value for the ''diff'' parameter.', 'InvalidArgument');
end

end % function


%%%%%%%%%%%%%
%%% tests %%%
%%%%%%%%%%%%%

%%
% 1D, 5x5

%!shared param x y
%!  dim = 1;
%!  model = stk_model('stk_materncov32_aniso', dim);
%!  param = model.param;
%!  x = stk_sampling_randunif(5, dim);
%!  y = stk_sampling_randunif(5, dim);

%!error stk_materncov32_aniso();
%!error stk_materncov32_aniso(param);
%!error stk_materncov32_aniso(param, x);
%!test  stk_materncov32_aniso(param, x, y);
%!test  stk_materncov32_aniso(param, x, y, -1);
%!error stk_materncov32_aniso(param, x, y, -1, pi^2);

%!error stk_materncov32_aniso(param, x, y, -2);
%!test  stk_materncov32_aniso(param, x, y, -1);
%!error stk_materncov32_aniso(param, x, y,  0);
%!test  stk_materncov32_aniso(param, x, y,  1);
%!test  stk_materncov32_aniso(param, x, y,  2);
%!error stk_materncov32_aniso(param, x, y,  3);
%!error stk_materncov32_aniso(param, x, y,  nan);
%!error stk_materncov32_aniso(param, x, y,  inf);

%%
% 3D, 4x10

%!shared dim param x y nx ny
%!  dim = 3;
%!  model = stk_model('stk_materncov32_aniso', dim);
%!  param = model.param;
%!  nx = 4; ny = 10;
%!  x = stk_sampling_randunif(nx,  dim);
%!  y = stk_sampling_randunif(ny, dim);

%!test
%!  K1 = stk_materncov32_aniso(param, x, y);
%!  K2 = stk_materncov32_aniso(param, x, y, -1);
%!  assert(isequal(size(K1), [nx ny]));
%!  assert(stk_isequal_tolabs(K1, K2));

%!test
%!  for i = 1:(dim+1),
%!    dK = stk_materncov32_aniso(param, x, y,  i);
%!    assert(isequal(size(dK), [nx ny]));
%!  end
