function y = betapdf(x, a, b)
  % Octave shim: beta PDF (statistics package not available for Octave 10).
  y = zeros(size(x));
  k = (x >= 0) & (x <= 1);
  y(k) = exp((a-1).*log(x(k)) + (b-1).*log1p(-x(k)) - betaln(a, b));
end
