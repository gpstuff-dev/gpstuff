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
## @deftypefn {Function File} {@var{y} =} mnpdf (@var{x}, @var{p})
## Compute the probability density function of the multinomial distribution.
##
## @subheading Arguments
##
## @itemize @bullet
## @item
## @var{x} is vector with a single sample of a multinomial distribution with
## parameter @var{p} or a matrix of random samples from multinomial
## distributions. In the latter case, each row of @var{x} is a sample from a
## multinomial distribution with the corresponding row of @var{p} being its
## parameter.
##
## @item
## @var{p} is a vector with the probabilities of the categories or a matrix
## with each row containing the probabilities of a multinomial sample.
## @end itemize
##
## @subheading Return values
##
## @itemize @bullet
## @item
## @var{y} is a vector of probabilites of the random samples @var{x} from the
## multinomial distribution with corresponding parameter @var{p}. The parameter
## @var{n} of the multinomial distribution is the sum of the elements of each
## row of @var{x}. The length of @var{y} is the number of columns of @var{x}.
## If a row of @var{p} does not sum to @code{1}, then the corresponding element
## of @var{y} will be @code{NaN}.
## @end itemize
##
## @subheading Examples
##
## @example
## @group
## x = [1, 4, 2];
## p = [0.2, 0.5, 0.3];
## y = mnpdf (x, p);
## @end group
##
## @group
## x = [1, 4, 2; 1, 0, 9];
## p = [0.2, 0.5, 0.3; 0.1, 0.1, 0.8];
## y = mnpdf (x, p);
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
## Description: PDF of the multinomial distribution

function y = mnpdf (x, p)

  # Check arguments
  if (nargin != 2)
    print_usage ();
  endif

  if (! ismatrix (x) || any (x(:) < 0 | round (x(:) != x(:))))
    error ("mnpdf: x must be a matrix of non-negative integer values");
  endif
  if (! ismatrix (p) || any (p(:) < 0))
    error ("mnpdf: p must be a non-empty matrix with rows of probabilities");
  endif

  # Adjust input sizes
  if (! isvector (x) || ! isvector (p))
    if (isvector (x))
      x = x(:)';
    endif
    if (isvector (p))
      p = p(:)';
    endif
    if (size (x, 1) == 1 && size (p, 1) > 1)
      x = repmat (x, size (p, 1), 1);
    elseif (size (x, 1) > 1 && size (p, 1) == 1)
      p = repmat (p, size (x, 1), 1);
    endif
  endif
  # Continue argument check
  if (any (size (x) != size (p)))
    error ("mnpdf: x and p must have compatible sizes");
  endif

  # Count total number of elements of each multinomial sample
  n = sum (x, 2);
  # Compute probability density function of the multinomial distribution
  t = x .* log (p);
  t(x == 0) = 0;
  y = exp (gammaln (n+1) - sum (gammaln (x+1), 2) + sum (t, 2));
  # Set invalid rows to NaN
  k = (abs (sum (p, 2) - 1) > 1e-6);
  y(k) = NaN;

endfunction

%!test
%! x = [1, 4, 2];
%! p = [0.2, 0.5, 0.3];
%! y = mnpdf (x, p);
%! assert (y, 0.11812, 0.001);

%!test
%! x = [1, 4, 2; 1, 0, 9];
%! p = [0.2, 0.5, 0.3; 0.1, 0.1, 0.8];
%! y = mnpdf (x, p);
%! assert (y, [0.11812; 0.13422], 0.001);
