#!/usr/bin/env python

import sys
from pylab import *
from scikits.audiolab import Sndfile, play
from sklearn.decomposition import PCA

fn = "dnb-165-clicktrack.ogg"
sound = [s.read_frames(s.nframes) for s in [Sndfile(fn)]][0]
samplerate = Sndfile(fn).samplerate

start = 64147  # determined in audacity

mouth = sound[start:,0]
clik = sound[start:,1]

BPM = 165
beatwidth = int(samplerate*60 / BPM)
width = beatwidth / 8

def make_pieces(width, audio):
    pieces = []
    features = []
    for i in xrange(len(audio)/width):
        piece = audio[width*i:width*(i+1)]
        pieces.append(piece)
        feat = log10(fft(piece)[:width/32]+1)
        features.append(feat)
    return array(pieces), array(features)

def plot_pieces_and_features(pieces, features):
    x = 0
    for p,f in zip(pieces, features):
        subplot(211)
        plot(arange(x, x+width), p)
        subplot(212)
        plot(logspace(0,log10(width),len(f))-1+x, f)
        x += len(p)

def pca_features(features):
    fig = figure(2)
    pca = PCA()
    transf = pca.fit_transform(features)
    artists = []
    for i in range(len(features)):
        a, = plot(transf[i,0], transf[i,1],
                 'o', markersize=7, color='blue',
                  alpha=0.5, picker = 3)
        artists.append(a)

    def on_pick(event):
        try:
            i = artists.index(event.artist)
            print i
            play(pieces[i])
        except:
            return True
        return True
    fig.canvas.mpl_connect('pick_event', on_pick)

pieces, features = make_pieces(width, mouth)
#figure(1)
#plot_pieces_and_features(pieces, features)
pca_features(features)

show()
