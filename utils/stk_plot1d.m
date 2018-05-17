% STK_PLOT1D is a convenient plot function for 1D kriging predictions
%
% CALL: stk_plot1d (XI, ZI, XT, ZT, ZP)
%
%    plots the evaluation points (XI, ZI), the "true function" with values
%    ZT on the grid XT, and a representation of the prediction ZP on the
%    same grid XT: the kriging prediction (posterior mean) surrounded by a
%    shaded area corresponding to 95% pointwise confidence intervals.
%
%    It is possible to omit plotting either the observations (XI, ZI) or
%    the true function ZT by providing empty matrices.
%
% CALL: stk_plot1d (XI, ZI, XT, ZT, ZP, ZSIM)
%
%    also plots a set ZSIM of samplepaths.
%
% CALL: stk_plot1d (H_AXES, ...)
%
%    plots into existing axes with axis handle H_AXES.
%
% CALL: H_PLOT = stk_plot1d (...)
%
%   returns the handles of the drawing in a structure.

% Copyright Notice
%
%    Copyright (C) 2015-2018 CentraleSupelec
%    Copyright (C) 2011-2014 SUPELEC
%
%    Authors:  Julien Bect       <julien.bect@centralesupelec.fr>
%              Emmanuel Vazquez  <emmanuel.vazquez@centralesupelec.fr>
%              Remi Stroh        <remi.stroh@lne.fr>

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

function h_plot = stk_plot1d (varargin)

% Extract axis handle (if it is present)
[h_axes, varargin] = stk_plot_getaxesarg (varargin{:});

if nargout < 1
    stk_plot1d_ (h_axes, varargin{:});
    % Do not display the handles if they are not asked (graphical option)
else
    h_plot = stk_plot1d_ (h_axes, varargin{:});
end

end % function


function h = stk_plot1d_ (h_axes, xi, zi, xt, zt, zp, zsim)

has_zt_arg   = (nargin > 4) && (~ isempty (zt));
has_zp_arg   = (nargin > 5) && (~ isempty (zp));
has_zsim_arg = (nargin > 6) && (~ isempty (zsim));

if (nargin > 3) && (~ isempty (xt))
    [xt, idx_sort] = sort (xt);
end

h = struct ('truth', [], 'obs', [], 'pred', [], 'ci', [], 'sim', []);

% Shaded area representing pointwise confidence intervals
if has_zp_arg
    zp = zp(idx_sort, :);
    h.ci = stk_plot_shadedci (h_axes, xt, zp);
    hold on;
end

% Plot sample paths
if has_zsim_arg
    if isa (zsim, 'stk_dataframe')
        % Prevents automatic creation of a legend by @stk_dataframe/plot
        zsim.colnames = {};
    end
    h.sim = plot (h_axes, xt, zsim(idx_sort, :), ...
        '-',  'LineWidth', 1, 'Color', [0.39, 0.47, 0.64]);
    set (h.sim(1), 'DisplayName', 'Samplepaths');
    hold on;
end

% Ground truth
if has_zt_arg
    h.truth = plot (h_axes, xt, zt(idx_sort, :), '--', 'LineWidth', 3, ...
        'Color', [0.39, 0.47, 0.64], 'DisplayName', 'True function');
    hold on;
end

% Kriging predictor (posterior mean)
if has_zp_arg
    h.pred = plot (h_axes, xt, zp.mean, 'LineWidth', 3, ...
        'Color', [0.95 0.25 0.3], 'DisplayName', 'Posterior mean');
    hold on;
end

% Evaluations
if ~ isempty (zi)
    h.obs = plot (h_axes, xi, zi, 'ko', 'MarkerSize', 6, ...
        'MarkerFaceColor', 'k', 'DisplayName', 'Observations');
end

hold off;  set (gca, 'box', 'off');

end % function
