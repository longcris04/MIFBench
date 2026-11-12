function out = blkproc(A, blockSize, border, fun, varargin)
  % Octave shim for MATLAB's removed blkproc(), backed by blockproc().
  if nargin < 4
    fun = border;           % blkproc(A, [m n], fun) form
    border = [0 0];
  end
  if any(border ~= 0)
    error('blkproc shim: nonzero border not supported');
  end
  % Octave's blockproc hands the block itself to the callback; MATLAB's
  % hands a block_struct. Accept either so the same shim works both ways.
  out = blockproc(A, blockSize, @(b) fun(blkproc_data(b), varargin{:}));
end

function d = blkproc_data(b)
  if isstruct(b), d = b.data; else d = b; end
end
