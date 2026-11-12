function mif_eval_list(list_file, out_file)
% MIF_EVAL_LIST  Score fused images with the metrics of Run_Foder_Updated.m.
%
%   mif_eval_list(LIST_FILE, OUT_FILE)
%
%   LIST_FILE  text file, one image per line, tab separated:
%                  name <TAB> path_1 <TAB> path_2 <TAB> path_3
%              path_1 = functional source as stored (CT / PET / SPECT, RGB allowed),
%              path_2 = MRI source, path_3 = fused grey image -- the same roles as
%              path_1 / path_2 / path_3 in Run_Foder_Updated.m.
%   OUT_FILE   CSV, one row per image, columns Num, Name and the 27 numeric
%              columns of Run_Foder_Updated.m (MLI ... metricZhao, x1, x2, Time).
%
%   The per-image computation is copied from Run_Foder_Updated.m line by line:
%   path_1 goes through rgb2gray when it has three channels, path_2 through
%   ConvertRGBtoYUV (Y channel), and every metric takes the uint8 images
%   im1 / im2 / imf in the same argument order.  y(20) (metricZhao) is -1 and
%   y(21), y(22) (x1, x2) are 0 there, and Time is the empty tic/toc, so those
%   four columns carry no information; they are kept so the columns line up
%   with the MATLAB output.  The one addition: a metric that raises an error
%   gives NaN for that image (and a message on stderr) instead of stopping
%   the whole list.

  here = fileparts(mfilename('fullpath'));
  root = fileparts(here);
  addpath(genpath(fullfile(root, 'indexes')));
  if exist('OCTAVE_VERSION', 'builtin')
    % Octave only (see octave_eval/README.md); later addpath calls take precedence.
    addpath(fullfile(root, 'octave_compat'));   % niqe, blkproc, freqspace, normcdf, ...
    addpath(fullfile(here, 'compat'));          % dct2 (missing from Octave's image package)
    addpath(fullfile(here, 'fast'));            % bit-identical vectorised fmi, Qabf, MI, NCC
    pkg load image
    warning('off', 'all');
  end

  fid = fopen(list_file, 'r');
  if fid < 0
    error('mif_eval_list: cannot open %s', list_file);
  end
  rows = {};
  while true
    line = fgetl(fid);
    if ~ischar(line), break; end
    if isempty(strtrim(line)), continue; end
    rows{end+1} = strsplit(line, sprintf('\t'));  %#ok<AGROW>
  end
  fclose(fid);

  headers = {'Num', 'Name', 'MLI', 'SD', 'Entropy', 'AG', 'Qabf', 'VIFF', 'FMI', 'MI', ...
             'Xydeas', 'Qp', 'Q', 'Qw', 'Qe', 'NIQE', 'SSEQ', 'FSIM', 'SSIM', 'TMQI', ...
             'PSNR', 'Chen', 'CB', 'metricWang', 'NCIE', 'metricZhao', 'x1', 'x2', 'Time'};

  tmp_file = [out_file '.part'];
  out = fopen(tmp_file, 'w');
  fprintf(out, '%s', headers{1});
  fprintf(out, ',%s', headers{2:end});
  fprintf(out, '\n');

  for k = 1:numel(rows)
    r = rows{k};
    name = r{1};
    filename1 = r{2};
    filename2 = r{3};
    filename3 = r{4};

    A = im2double(imread(filename1));
    img2 = im2double(imread(filename2));
    imgf = im2double(imread(filename3));

    if size(A, 3) > 1
      A = rgb2gray(A);
    end

    % Convert RGB to YUV
    B_YUV = ConvertRGBtoYUV(img2);
    BB = B_YUV(:, :, 1);

    tic
    time = toc;

    F = imgf;
    v = safe(@() DanhGiaAnh(F), 4, name, 'DanhGiaAnh');
    v1 = zeros(1, 4);
    v1(1) = v(1);
    v1(2) = v(2);
    v1(3) = v(3);
    v1(4) = v(4);

    im1 = im2uint8(A);
    im2 = im2uint8(BB);
    imf = im2uint8(F);

    y = zeros(1, 22);
    y(1) = safe(@() Qabf(im1, im2, imf), 1, name, 'Qabf');
    y(2) = safe(@() VIFF(im1, im2, imf), 1, name, 'VIFF');
    y(3) = safe(@() fmi(im1, im2, imf, 'edge', 3), 1, name, 'fmi');
    y(4) = safe(@() mutual_information(im1, im2, imf, 256), 1, name, 'mutual_information');
    y(5) = safe(@() metricXydeas(im1, im2, imf), 1, name, 'metricXydeas');
    y(6) = safe(@() Qp_ABF(im1, im2, imf), 1, name, 'Qp_ABF');
    y(7) = safe(@() metricPeilla(im1, im2, imf, 1), 1, name, 'metricPeilla1');
    y(8) = safe(@() metricPeilla(im1, im2, imf, 2), 1, name, 'metricPeilla2');
    y(9) = safe(@() metricPeilla(im1, im2, imf, 3), 1, name, 'metricPeilla3');
    y(10) = safe(@() niqe(imf), 1, name, 'niqe');
    y(11) = safe(@() SSEQ(cat(3, im1, im2, imf)), 1, name, 'SSEQ');
    y(12) = safe(@() FSIM(im1, imf), 1, name, 'FSIM');
    y(13) = safe(@() MS_SSIM(im1, im2, imf), 1, name, 'MS_SSIM');
    y(14) = safe(@() TMQI(cat(3, im1, im1, im1), cat(3, imf, imf, imf)), 1, name, 'TMQI');
    y(15) = safe(@() psnr(imf, im1), 1, name, 'psnr');
    y(16) = safe(@() metricChen(im1, im2, imf), 1, name, 'metricChen');
    y(17) = safe(@() metricChenBlum(im1, im2, imf), 1, name, 'metricChenBlum');
    y(18) = safe(@() metricWang(im1, im2, imf), 1, name, 'metricWang');
    y(19) = safe(@() Q_NCIE(im1, im2, imf), 1, name, 'Q_NCIE');
    y(20) = -1;  % metricZhao(imresize(im1,2), imresize(im2,2), imresize(imf,2)) is disabled in Run_Foder_Updated.m

    fprintf(out, '%d,%s', k, name);
    fprintf(out, ',%.10g', [v1, y, time]);
    fprintf(out, '\n');
    fflush(out);
  end
  fclose(out);
  movefile(tmp_file, out_file, 'f');
end


function value = safe(fn, n, name, label)
  try
    value = double(fn());
    if numel(value) < n
      value(end+1:n) = NaN;
    end
    value = value(1:n);
  catch err
    fprintf(2, 'mif_eval_list: %s failed on %s: %s\n', label, name, err.message);
    value = NaN(1, n);
  end
end
