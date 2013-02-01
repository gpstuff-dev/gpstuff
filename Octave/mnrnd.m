## Copyright (C) 2012  Arno Onken
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Function File} {@var{x} =} mnrnd (@var{n}, @var{p})
## @deftypefnx {Function File} {@var{x} =} mnrnd (@var{n}, @var{p}, @var{s})
## Generate random samples from the multinomial distribution.
##
## @subheading Arguments
##
## @itemize @bullet
## @item
## @var{n} is the first parameter of the multinomial distribution. @var{n} can
## be scalar or a vector containing the number of trials of each multinomial
## sample. The elements of @var{n} must be non-negative integers.
##
## @item
## @var{p} is the second parameter of the multinomial distribution. @var{p} can
## be a vector with the probabilities of the categories or a matrix with each
## row containing the probabilities of a multinomial sample. If @var{p} has
## more than one row and @var{n} is non-scalar, then the number of rows of
## @var{p} must match the number of elements of @var{n}.
##
## @item
## @var{s} is the number of multinomial samples to be generated. @var{s} must
## be a non-negative integer. If @var{s} is specified, then @var{n} must be
## scalar and @var{p} must be a vector.
## @end itemize
##
## @subheading Return values
##
## @itemize @bullet
## @item
## @var{x} is a matrix of random samples from the multinomial distribution with
## corresponding parameters @var{n} and @var{p}. Each row corresponds to one
## multinomial sample. The number of columns, therefore, corresponds to the
## number of columns of @var{p}. If @var{s} is not specified, then the number
## of rows of @var{x} is the maximum of the number of elements of @var{n} and
## the number of rows of @var{p}. If a row of @var{p} does not sum to @code{1},
## then the corresponding row of @var{x} will contain only @code{NaN} values.
## @end itemize
##
## @subheading Examples
##
## @example
## @group
## n = 10;
## p = [0.2, 0.5, 0.3];
## x = mnrnd (n, p);
## @end group
##
## @group
## n = 10 * ones (3, 1);
## p = [0.2, 0.5, 0.3];
## x = mnrnd (n, p);
## @end group
##
## @group
## n = (1:2)';
## p = [0.2, 0.5, 0.3; 0.1, 0.1, 0.8];
## x = mnrnd (n, p);
## @end group
## @end example
##
## @subheading References
##
## @enumerate
## @item
## Wendy L. Martinez and Angel R. Martinez. @cite{Computational Statistics
## Handbook with MATLAB}. Appendix E, pages 547-557, Chapman & Hall/CRC, 2001.
##
## @item
## Merran Evans, Nicholas Hastings and Brian Peacock. @cite{Statistical
## Distributions}. pages 134-136, Wiley, New York, third edition, 2000.
## @end enumerate
## @end deftypefn

## Author: Arno Onken <asnelt@asnelt.org>
## Description: Random samples from the multinomial distribution

function x = mnrnd (n, p, s)

  # Check arguments
  if (nargin == 3)
    if (! isscalar (n) || n < 0 || round (n) != n)
      error ("mnrnd: n must be a non-negative integer");
    endif
    if (! isvector (p) || any (p < 0 | p > 1))
      error ("mnrnd: p must be a vector of probabilities");
    endif
    if (! isscalar (s) || s < 0 || round (s) != s)
      error ("mnrnd: s must be a non-negative integer");
    endif
  elseif (nargin == 2)
    if (isvector (p) && size (p, 1) > 1)
      p = p';
    endif
    if (! isvector (n) || any (n < 0 | round (n) != n) || size (n, 2) > 1)
      error ("mnrnd: n must be a non-negative integer column vector");
    endif
    if (! ismatrix (p) || isempty (p) || any (p < 0 | p > 1))
      error ("mnrnd: p must be a non-empty matrix with rows of probabilities");
    endif
    if (! isscalar (n) && size (p, 1) > 1 && length (n) != size (p, 1))
      error ("mnrnd: the length of n must match the number of rows of p");
    endif
  else
    print_usage ();
  endif

  # Adjust input sizes
  if (nargin == 3)
    n = n * ones (s, 1);
    p = repmat (p(:)', s, 1);
  elseif (nargin == 2)
    if (isscalar (n) && size (p, 1) > 1)
      n = n * ones (size (p, 1), 1);
    elseif (size (p, 1) == 1)
      p = repmat (p, length (n), 1);
    endif
  endif
  sz = size (p);

  # Upper bounds of categories
  ub = cumsum (p, 2);
  # Make sure that the greatest upper bound is 1
  gub = ub(:, end);
  ub(:, end) = 1;
  # Lower bounds of categories
  lb = [zeros(sz(1), 1) ub(:, 1:(end-1))];

  # Draw multinomial samples
  x = zeros (sz);
  for i = 1:sz(1)
    # Draw uniform random numbers
    r = repmat (rand (n(i), 1), 1, sz(2));
    # Compare the random numbers of r to the cumulated probabilities of p and
    # count the number of samples for each category
    x(i, :) =  sum (r <= repmat (ub(i, :), n(i), 1) & r > repmat (lb(i, :), n(i), 1), 1);
  endfor
  # Set invalid rows to NaN
  k = (abs (gub - 1) > 1e-6);
  x(k, :) = NaN;

endfunction

%!test
%! n = 10;
%! p = [0.2, 0.5, 0.3];
%! x = mnrnd (n, p);
%! assert (size (x), size (p));
%! assert (all (x >= 0));
%! assert (all (round (x) == x));
%! assert (sum (x) == n);

%!test
%! n = 10 * ones (3, 1);
%! p = [0.2, 0.5, 0.3];
%! x = mnrnd (n, p);
%! assert (size (x), [length(n), length(p)]);
%! assert (all (x >= 0));
%! assert (all (round (x) == x));
%! assert (all (sum (x, 2) == n));

%!test
%! n = (1:2)';
%! p = [0.2, 0.5, 0.3; 0.1, 0.1, 0.8];
%! x = mnrnd (n, p);
%! assert (size (x), size (p));
%! assert (all (x >= 0));
%! assert (all (round (x) == x));
%! assert (all (sum (x, 2) == n));
