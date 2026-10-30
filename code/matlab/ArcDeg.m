function res = ArcDeg(x,y)
%
% The MIT License (MIT)
%
% Copyright (c) 2026, Anna Ostaszewska-Liżewska, Roman Szewczyk
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
% DESCRIPTION:
% function calculates the angle of (x,y) vector
% res - output in deg
%

R=sqrt(x.*x+y.*y);
if R==0
  res = NaN;
  fprintf('\n\n *** ArcDeg: Error R=0, x=%f, y=%f *** \n\n',x,y);
  return;
end
%

if ((x>=0) && (y>=0))                 % the first quarter
  res = atan(y./x);
elseif ((x<0) && (y>=0))              % the second quarter
  res = pi-atan(abs(y)./abs(x));
elseif ((x<0) && (y<0))               % the third quarter
  res = pi+atan(abs(y)./abs(x));
elseif ((x>=0) && (y<0))              % the fourth quarter
  res = 2.*pi-atan(abs(y)./abs(x));
else                                  % something went wrong
  res = NaN;
  fprintf('\n\n *** ArcDeg: NaN Error, x=%f, y=%f *** \n\n',x,y);
end

res = res *180./pi;

end
