% STK_MODEL_GPPOSTERIOR constructs a posterior model

% Copyright Notice
%
%    Copyright (C) 2015 CentraleSupelec
%
%    Author:  Julien Bect  <julien.bect@centralesupelec.fr>

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

function M_post = stk_model_gpposterior (M_prior, xi, zi)

if nargin == 0
    
    M_post.prior_model  = [];
    M_post.dim          = [];
    M_post.input_data   = [];
    M_post.output_data  = [];
    M_post.kreq         = [];
    
elseif nargin == 3
    
    % Backward compatiblity: accept model structures with missing lognoisevariance
    if (~ isfield (M_prior, 'lognoisevariance')) ...
            || (isempty (M_prior.lognoisevariance))
        M_prior.lognoisevariance = - inf;
    end
    
    % Prepare the lefthand side of the KRiging EQuation
    if iscell (xi)
        % WARNING: experimental HIDDEN feature, use at your own risk !!!
        kreq = xi{2}; % already computed, I hope you know what you're doing ;-)
        xi = xi{1};
    else
        kreq = stk_kreq_qr (M_prior, xi);
    end
    
    n = size (xi, 1);
    
    % Check the size of z_obs
    if ~ (isempty (zi) || isequal (size (zi), [n 1]))
        stk_error (['z_obs must either be empty or have the ' ...
            'same number of rows as x_obs.'], 'IncorrectSize');
    end
    
    M_post.prior_model  = M_prior;
    M_post.dim          = size (xi, 2);
    M_post.input_data   = double (xi);
    M_post.output_data  = double (zi);
    M_post.kreq         = kreq;
    
else
    
    stk_error ('Incorrect number of input arguments.', 'SyntaxError');
    
end % if

M_post = class (M_post, 'stk_model_gpposterior');

end % function stk_model_gpposterior


%!shared M_prior, x_obs, z_obs
%! x_obs = (linspace (0, pi, 15))';
%! z_obs = sin (x_obs);
%!
%! M_prior = stk_model ('stk_materncov32_iso');
%! M_prior.order = 0; % this is currently the default, but better safe than sorry
%! M_prior.param = log ([1.0; 2.1]);

%!test  M_post = stk_model_gpposterior ();
%!test  M_post = stk_model_gpposterior (M_prior, x_obs, z_obs);
%!error M_post = stk_model_gpposterior (M_prior, x_obs, [z_obs; z_obs]);
%!error M_post = stk_model_gpposterior (M_prior, x_obs, [z_obs; z_obs], 3.441);

%!test % hidden feature
%! kreq = stk_kreq_qr (M_prior, x_obs);
%! M_post = stk_model_gpposterior (M_prior, {x_obs, kreq}, z_obs);