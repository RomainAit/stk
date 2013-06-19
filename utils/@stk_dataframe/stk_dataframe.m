% STK_DATAFRAME constructs a dataframe

% Copyright Notice
%
%    Copyright (C) 2013 SUPELEC
%
%    Author: Julien Bect  <julien.bect@supelec.fr>

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

function x = stk_dataframe(x0, colnames, rownames)

stk_narginchk(0, 3);

if nargin == 0  % default constructor
    x = struct('data', zeros(0, 1), 'vnames', {{}}, 'rownames', {{}});    
else    
    x = struct('data', x0, 'vnames', {{}}, 'rownames', {{}});
end   

x = class(x, 'stk_dataframe');

if nargin >= 2,
    x = stk_set_colnames(x, colnames);
end

if nargin >= 3,
    x = stk_set_rownames(x, rownames);
end

end % function stk_dataframe

%!shared x y

%!test % default constructor
%! x = stk_dataframe();   

%!test
%! y = stk_dataframe(rand(3, 2));
%! assert (isa (y, 'stk_dataframe') && isequal(size(y), [3 2]))

%!test
%! y = stk_dataframe(rand(3, 2), {'x', 'y'});
%! assert (isa (y, 'stk_dataframe') && isequal(size(y), [3 2]))

%!test
%! y = stk_dataframe(rand(3, 2), {'x', 'y'}, {'a', 'b', 'c'});
%! assert (isa (y, 'stk_dataframe') && isequal(size(y), [3 2]))