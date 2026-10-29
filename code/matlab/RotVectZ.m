function vr = RotVectZ(v, fi_d)
%
% The MIT License (MIT)
%
% Copyright (c) 2023 Roman Szewczyk
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
%
%
% Function rotates vector v around versor vers by angle fi
% consistient version
%
% v - input vector (3,1)
% vers - versor for rotation (around) (3,1)
% fi - rotation angle (1)  - in degrees
%
% https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
%

vers = [0; 0; 1];

fi=fi_d./180.*pi;     % in radians

vers=vers./norm(vers); % to be sure

vr=vers.*dot(vers,v)+cos(fi).*cross(cross(vers,v),vers)+sin(fi).*cross(vers,v);

end
