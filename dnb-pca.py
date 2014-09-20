#!/usr/bin/env python

import sys
from pylab import *
from scikits.audiolab import Sndfile, play
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
        feat = log10(abs(fft(piece)[:width/32]+1))
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

def pca_features_interactive(features):
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
    show()

def pca_features(features):
    fig = figure(2)
    pca = PCA()
    transf = pca.fit_transform(features)
    return transf[:,0:2]

def do_kmeans(data, clusters=2, show=False):
    kmeans = KMeans(n_clusters = clusters)
    fit = kmeans.fit_predict(data)
    for c in range(clusters):
        d = array([x for x,k in zip(data,fit) if k==c]).T
        if show:
            plot(d[0], d[1], 'o', alpha=0.5)
    return fit

def plot_pieces_and_clusters(pieces, clusters):
    n_clusters = max(clusters)+1
    colors = [cm.jet(float(x)/n_clusters) for x in range(n_clusters)]
    m = max([p.max() for p in pieces])
    n = min([p.min() for p in pieces])
    for i in xrange(len(pieces)):
        plot(arange(width*i, width*(i+1)), pieces[i], 'k-', alpha=0.5)
        fill_between([width*i, width*(i+1)], [m]*2, [n]*2,
                     color = colors[clusters[i]], alpha = 0.5)

pieces, features = make_pieces(width, mouth)
#figure(1)
#plot_pieces_and_features(pieces, features)
#pca_features_interactive(features)
#clusters = do_kmeans(pca_features(features), clusters=6)
clusters = do_kmeans(features, clusters=6)

plot_pieces_and_clusters(pieces, clusters)

show()
