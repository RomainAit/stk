% STK_RESCALE rescales a dataset from one box to another

% Copyright Notice
%
%    Copyright (C) 2012-2014 SUPELEC
%
%    Author:  Julien Bect  <julien.bect@supelec.fr>

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

function [y, a, b] = stk_rescale (x, box1, box2)

if nargin > 3,
   stk_error ('Too many input arguments.', 'TooManyInputArgs');
end

% make sure that box1 is an stk_hrect object
if isempty (box1)
    box1 = stk_hrect (size (x, 2));
else
    box1 = stk_hrect (box1);
end

% call @stk_hrect/stk_rescale
[y, a, b] = stk_rescale (x, box1, box2);

end % function stk_rescale


%!shared x
%! x = rand (10, 4);
%! y = stk_rescale (x, [], []);
%! assert (stk_isequal_tolabs (x, y));

%!test
%! y = stk_rescale(0.5, [], [0; 2]);
%! assert (stk_isequal_tolabs (y, 1.0));

%!test
%! y = stk_rescale (0.5, [0; 1], [0; 2]);
%! assert (stk_isequal_tolabs (y, 1.0));

%!test
%! y = stk_rescale (0.5, [0; 2], []);
%! assert (stk_isequal_tolabs (y, 0.25));

%!test
%! y = stk_rescale (0.5, [0; 2], [0; 1]);
%! assert (stk_isequal_tolabs (y, 0.25));
