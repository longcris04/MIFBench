function Mutual_information=mutual_information(grey_matrixA,grey_matrixB,grey_matrixF,grey_level)
% MUTUAL_INFORMATION  Vectorised copy of indexes/mutual_information.m (Octave speed-up).
%
%   Same result bit for bit: the histograms the original counts pixel by pixel
%   are counted with accumarray (exact integer counts), then the original
%   formulas are applied unchanged.  Kept on purpose: "grey_matrix+1" is done
%   on the uint8 input, which saturates at 255, so grey levels 254 and 255
%   share a bin -- exactly as in the original.

HA=entropy_fusion(grey_matrixA,grey_level);
HB=entropy_fusion(grey_matrixB,grey_level);
HF=entropy_fusion(grey_matrixF,grey_level);
HFA=Hab(grey_matrixF,grey_matrixA,grey_level);
HFB=Hab(grey_matrixF,grey_matrixB,grey_level);
MIFA=HA+HF-HFA;
MIFB=HB+HF-HFB;
Mutual_information=MIFA+MIFB;
end


function entropyR=entropy_fusion(grey_matrix,grey_level)
grey_matrix=grey_matrix+1;
counter = accumarray(double(grey_matrix(:)), 1, [grey_level 1]).';
total= sum(counter(:));
index = find(counter~=0);
p = counter/total;
entropyR= sum(sum(-p(index).*log2(p(index))));
end


function HabR=Hab(grey_matrixA,grey_matrixB,grey_level)
grey_matrixA=grey_matrixA+1;
grey_matrixB=grey_matrixB+1;
counter = accumarray([double(grey_matrixA(:)), double(grey_matrixB(:))], 1, [256 256]);
total= sum(counter(:));
index = find(counter~=0);
p = counter/total;
HabR = sum(sum(-p(index).*log2(p(index))));
end
