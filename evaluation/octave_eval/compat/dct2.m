function B = dct2(A, m, n)
% DCT2  2-D discrete cosine transform, MATLAB-compatible (Octave compat shim).
%
%   Octave's image package lists dct2 but does not implement it; SSEQ
%   (fecal.m) needs it.  This is MATLAB's definition: the orthonormal DCT-II
%   along the columns, then along the rows,
%
%       B(p,q) = w(p) w(q) sum_m sum_n A(m,n) cos(pi(2m-1)(p-1)/2M) cos(pi(2n-1)(q-1)/2N)
%       w(1) = 1/sqrt(K), w(k>1) = sqrt(2/K),
%
%   computed as C_M * A * C_N.'.  dct2(A, [M N]) / dct2(A, M, N) pad with
%   zeros or crop to M x N first, as in MATLAB.  Agrees with MATLAB's
%   FFT-based dct2 to rounding error (~1e-15 relative).

  A = double(A);
  if nargin == 2
    n = m(2); m = m(1);
  end
  if nargin >= 2
    T = zeros(m, n);
    mm = min(m, size(A, 1)); nn = min(n, size(A, 2));
    T(1:mm, 1:nn) = A(1:mm, 1:nn);
    A = T;
  end
  [M, N] = size(A);
  B = dct_matrix(M) * A * dct_matrix(N).';
end


function C = dct_matrix(K)
  [j, k] = meshgrid(0:K-1, 0:K-1);       % k: frequency (row), j: sample (column)
  C = cos(pi * (2*j + 1) .* k / (2*K));
  C(1, :) = C(1, :) / sqrt(K);
  C(2:end, :) = C(2:end, :) * sqrt(2/K);
end
