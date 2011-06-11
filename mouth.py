#!/usr/bin/env python

from pylab import *
from scikits.audiolab import Sndfile, play
from scipy.signal import lfilter
from itertools import izip

def nonorm_spectral_centroid(d, n=None):
    if n==None: n = len(d)
    f = abs(fft(d,n))[:n/2]
    return sum(f * linspace(0,1,n/2))

def spectral_centroid(d, n=None):
    if n==None: n = len(d)
    f = abs(fft(d,n))[:n/2]
    return sum(f/sum(f) * linspace(0,1,n/2))

def chunks(d, hopsize, size):
    for i in xrange(len(d)/hopsize):
        yield d[i*hopsize:i*hopsize+size]

def findedges(amp, dcent, ta=0.2, td=2, n=16):
    state = 0
    for a, d in izip(amp, dcent):
        if state == 0:
            if d > td and a > ta:
                yield 1
                state = 1
                c = 0
            else:
                yield 0
            continue
        if state == 1:
            if d < td and a < ta:
                c += 1
            else:
                c = 0
            if c > n:
                state = 0
            yield 0

def get_edgeffts(data, edges, n=32):
    for i, e in enumerate(edges):
        if e == 1:
            yield abs(fft(data[i*hopsize-n:i*hopsize])[:n/2])

def get_sounds(data, edges, n=32, hopsize=16):
    for i, e in enumerate(edges):
        if e == 1:
            yield data[i*hopsize-n:i*hopsize-n+4000]

mouth = [s.read_frames(s.nframes) for s in [Sndfile('mouth.ogg')]][0]

hopsize = 16
fftsize = 256
hoptime = arange(len(mouth)/hopsize)*hopsize

centroids = map(nonorm_spectral_centroid,chunks(mouth, hopsize, fftsize))
dcentroids = lfilter([1,-1],[1],centroids)
edges = array(list(findedges(abs(mouth[::hopsize]),dcentroids, 0.2, 2, 32)))
edgeffts = list(get_edgeffts(mouth, edges))
sounds = list(get_sounds(mouth, edges))
sounds = reduce(lambda x,y: concatenate((x,y)), sounds)

figure(1)
clf()
subplot(6,1,1)
plot(mouth)
subplot(6,1,2)
plot(abs(mouth))
subplot(6,1,3)
plot(hoptime, centroids)
subplot(6,1,4)
plot(hoptime, dcentroids)
subplot(6,1,5)
plot(hoptime, edges)
subplot(6,1,6)
[plot(f/max(f)) for f in edgeffts]
draw()

play(sounds)

show()
