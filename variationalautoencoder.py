# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 12:26:22 2022

@author: shini
"""

import numpy
import matplotlib.pyplot as mpl

class variationalneuron:
    
    def __init__(self, dendrites, activation = 'linear'):
        self.bias = 1
        self.dendrites = 2 * numpy.random.random(dendrites) - 1
        self.activation = activation
        self.axons = 1
    
    def fire(self, neurotransmitters):
        firescore = []
        if len(self.dendrites.shape) == 1:
            for i in range(neurotransmitters.shape[0] - self.dendrites.shape[0] + 1):
                thisscore = self.bias
                for l in range(self.dendrites.shape[0]):
                    thisscore = thisscore + neurotransmitters[i + l] * self.dendrites[l]
                firescore.append(thisscore)
        elif len(self.dendrites.shape) == 2:
            for i in range(neurotransmitters.shape[0] - self.dendrites.shape[0] + 1):
                thisrow = []
                for j in range(neurotransmitters.shape[1] - self.dendrites.shape[1] + 1):
                    thisscore = self.bias
                    for l in range(self.dendrites.shape[0]):
                        for m in range(self.dendrites.shape[1]):
                            thisscore = thisscore + neurotransmitters[i + l][j + m] * self.dendrites[l][m]
                    thisrow.append(thisscore)
                if len(thisrow) == 1:
                    firescore.append(thisrow[0])
                else:
                    firescore.append(thisrow)
        elif len(self.dendrites.shape) == 3:
            for i in range(neurotransmitters.shape[0] - self.dendrites.shape[0] + 1):
                thiscol = []
                for j in range(neurotransmitters.shape[1] - self.dendrites.shape[1] + 1):
                    thisrow = []
                    for k in range(neurotransmitters.shape[2] - self.dendrites.shape[2] + 1):
                        thisscore = self.bias
                        for l in range(self.dendrites.shape[0]):
                            for m in range(self.dendrites.shape[1]):
                                for n in range(self.dendrites.shape[2]):
                                    thisscore = thisscore + neurotransmitters[i + l][j + m][k + n] * self.dendrites[l][m][n]
                        thisrow.append(thisscore)
                    if len(thisrow) == 1:
                        thiscol.append(thisrow[0])
                    else:
                        thiscol.append(thisrow)
                if len(thiscol) == 1:
                    firescore.append(thiscol[0])
                else:
                    firescore.append(thiscol)
        firescore = numpy.array(firescore)
        if len(firescore.shape) > 1 and firescore.shape[0] == 1:
            firescore = firescore[0]
        if self.activation == 'linear':
            return firescore
        elif self.activation == 'sigmoid':
            return 1 / (1 + numpy.exp(-1 * firescore))
        elif self.activation == 'relu':
            return numpy.maximum(firescore, numpy.zeros(firescore.shape))
    
    def setweights(self, newbias, weightarray):
        self.bias = newbias
        if len(self.dendrites.shape) == 1:
            for i in range(self.dendrites.shape[0]):
                self.dendrites[i] = weightarray[i]
        elif len(self.dendrites.shape) == 2:
            for i in range(self.dendrites.shape[0]):
                for j in range(self.dendrites.shape[1]):
                    self.dendrites[i][j] = weightarray[i][j]
        elif len(self.dendrites.shape) == 3:
            for i in range(self.dendrites.shape[0]):
                for j in range(self.dendrites.shape[1]):
                    for k in range(self.dendrites.shape[2]):
                        self.dendrites[i][j][k] = weightarray[i][j][k]
    
    def updateweights(self, error, inputs, learnrate):
        newerror = numpy.zeros(inputs.shape)
        if numpy.isscalar(error):
            self.bias = self.bias + error * learnrate
            if self.activation == 'sigmoid':
                sigmoidderivative = 0
                if len(self.dendrites.shape) == 1:
                    for i in range(self.dendrites.shape[0]):
                        sigmoidderivative = sigmoidderivative + self.dendrites[i] * inputs[i]
                elif len(self.dendrites.shape) == 2:
                    for i in range(self.dendrites.shape[0]):
                        for j in range(self.dendrites.shape[1]):
                            sigmoidderivative = sigmoidderivative + self.dendrites[i][j] * inputs[i][j]
                elif len(self.dendrites.shape) == 3:
                    for i in range(self.dendrites.shape[0]):
                        for j in range(self.dendrites.shape[1]):
                            for k in range(self.dendrites.shape[2]):
                                sigmoidderivative = sigmoidderivative + self.dendrites[i][j][k] * inputs[i][j][k]
                sigmoidderivative = numpy.exp(-1 * sigmoidderivative) / (1 + numpy.exp(-1 * sigmoidderivative))**2
            if len(self.dendrites.shape) == 1:
                for i in range(self.dendrites.shape[0]):
                    if self.activation == 'sigmoid':
                        newerror[i] = self.dendrites[i] * sigmoidderivative * error
                        self.dendrites[i] = self.dendrites[i] + error * sigmoidderivative * inputs[i] * learnrate
                    else:
                        newerror[i] = self.dendrites[i] * error
                        self.dendrites[i] = self.dendrites[i] + error * inputs[i] * learnrate
            elif len(self.dendrites.shape) == 2:
                for i in range(self.dendrites.shape[0]):
                    for j in range(self.dendrites.shape[1]):
                        if self.activation == 'sigmoid':
                            newerror[i][j] = self.dendrites[i][j] * sigmoidderivative * error
                            self.dendrites[i][j] = self.dendrites[i][j] + error * sigmoidderivative * inputs[i][j] * learnrate
                        else:
                            newerror[i][j] = self.dendrites[i][j] * error
                            self.dendrites[i][j] = self.dendrites[i][j] + error * inputs[i][j] * learnrate
            elif len(self.dendrites.shape) == 3:
                for i in range(self.dendrites.shape[0]):
                    for j in range(self.dendrites.shape[1]):
                        for k in range(self.dendrites.shape[2]):
                            if self.activation == 'sigmoid':
                                newerror[i][j][k] = self.dendrites[i][j][k] * sigmoidderivative * error
                                self.dendrites[i][j][k] = self.dendrites[i][j][k] + error * sigmoidderivative * inputs[i][j][k] * learnrate
                            else:
                                newerror[i][j][k] = self.dendrites[i][j][k] * error
                                self.dendrites[i][j][k] = self.dendrites[i][j][k] + error * inputs[i][j][k] * learnrate
        else:
            dendritegradient = numpy.zeros(self.dendrites.shape)
            if len(error.shape) == 1:
                for i in range(error.shape[0]):
                    if len(error.shape) == len(self.dendrites.shape): # This assumes that the input will have the same dimensions as the output or be greater by 1
                        if self.activation == 'sigmoid':
                            sigmoidderivative = 0
                            for j in range(self.dendrites.shape[0]):
                                sigmoidderivative = sigmoidderivative + self.dendrites[j] * inputs[i + j]
                            sigmoidderivative = numpy.exp(-1 * sigmoidderivative) / (1 + numpy.exp(-1 * sigmoidderivative))**2
                        for j in range(self.dendrites.shape[0]):
                            if self.activation == 'sigmoid':
                                newerror[i + j] = newerror[i + j] + self.dendrites[j] * sigmoidderivative * error[i]
                                dendritegradient[j] = dendritegradient[j] + error[i] * sigmoidderivative * inputs[i + j] * learnrate
                            else:
                                newerror[i + j] = newerror[i + j] + self.dendrites[j] * error[i]
                                dendritegradient[j] = dendritegradient[j] + error[i] * inputs[i + j] * learnrate
                    else:
                        if self.activation == 'sigmoid':
                            sigmoidderivative = 0
                            for k in range(self.dendrites.shape[0]):
                                for j in range(self.dendrites.shape[1]):
                                    sigmoidderivative = sigmoidderivative + self.dendrites[k][j] * inputs[k][i + j]
                            sigmoidderivative = numpy.exp(-1 * sigmoidderivative) / (1 + numpy.exp(-1 * sigmoidderivative))**2
                        for k in range(self.dendrites.shape[0]):
                            for j in range(self.dendrites.shape[1]):
                                if self.activation == 'sigmoid':
                                    newerror[k][i + j] = newerror[k][i + j] + self.dendrites[k][j] * sigmoidderivative * error[i]
                                    dendritegradient[k][j] = dendritegradient[k][j] + error[i] * sigmoidderivative * inputs[k][i + j] * learnrate
                                else:
                                    newerror[k][i + j] = newerror[k][i + j] + self.dendrites[k][j] * error[i]
                                    dendritegradient[k][j] = dendritegradient[k][j] + error[i] * inputs[k][i + j] * learnrate
            elif len(error.shape) == 2:
                for i in range(error.shape[0]):
                    for j in range(error.shape[1]):
                        if len(error.shape) == len(self.dendrites.shape):
                            if self.activation == 'sigmoid':
                                sigmoidderivative = 0
                                for k in range(self.dendrites.shape[0]):
                                    for l in range(self.dendrites.shape[1]):
                                        sigmoidderivative = sigmoidderivative + self.dendrites[k][l] * inputs[i + k][j + l]
                                sigmoidderivative = numpy.exp(-1 * sigmoidderivative) / (1 + numpy.exp(-1 * sigmoidderivative))**2
                            for k in range(self.dendrites.shape[0]):
                                for l in range(self.dendrites.shape[1]):
                                    if self.activation == 'sigmoid':
                                        newerror[i + k][j + l] = newerror[i + k][j + l] + self.dendrites[k][l] * sigmoidderivative * error[i][j]
                                        dendritegradient[k][l] = dendritegradient[k][l] + error[i][j] * sigmoidderivative * inputs[i + k][j + l] * learnrate
                                    else:
                                        newerror[i + k][j + l] = newerror[i + k][j + l] + self.dendrites[k][l] * error[i][j]
                                        dendritegradient[k][l] = dendritegradient[k][l] + error[i][j] * inputs[i + k][j + l] * learnrate
                        else:
                            if self.activation == 'sigmoid':
                                sigmoidderivative = 0
                                for m in range(self.dendrites.shape[0]):
                                    for k in range(self.dendrites.shape[1]):
                                        for l in range(self.dendrites.shape[2]):
                                            sigmoidderivative = sigmoidderivative + self.dendrites[m][k][l] * inputs[m][i + k][j + l]
                                sigmoidderivative = numpy.exp(-1 * sigmoidderivative) / (1 + numpy.exp(-1 * sigmoidderivative))**2
                            for m in range(self.dendrites.shape[0]):
                                for k in range(self.dendrites.shape[1]):
                                    for l in range(self.dendrites.shape[2]):
                                        if self.activation == 'sigmoid':
                                            newerror[m][i + k][j + l] = newerror[m][i + k][j + l] + self.dendrites[m][k][l] * sigmoidderivative * error[i][j]
                                            dendritegradient[m][k][l] = dendritegradient[m][k][l] + error[i][j] * sigmoidderivative * inputs[m][i + k][j + l] * learnrate
                                        else:
                                            newerror[m][i + k][j + l] = newerror[m][i + k][j + l] + self.dendrites[m][k][l] * error[i][j]
                                            dendritegradient[m][k][l] = dendritegradient[m][k][l] + error[i][j] * inputs[m][i + k][j + l] * learnrate
            self.dendrites = self.dendrites + dendritegradient
        return newerror

class variationalnetwork:
    
    def __init__(self, encoderlayers, encoderdendrites, encoderactivations, decoderlayers, decoderdendrites, decoderactivations):
        self.encoderblueprint = encoderlayers
        self.encoder = []
        for i in range(len(encoderlayers) - 1):
            thislayer = []
            if len(encoderlayers[i]) == 1:
                for j in range(encoderlayers[i][0]):
                    thislayer.append(variationalneuron(encoderdendrites[i], activation = encoderactivations[i]))
            elif len(encoderlayers[i]) == 2:
                for j in range(encoderlayers[i][0]):
                    thisrow = []
                    for k in range(encoderlayers[i][1]):
                        thisrow.append(variationalneuron(encoderdendrites[i], activation = encoderactivations[i]))
                    thislayer.append(thisrow)
            elif len(encoderlayers[i]) == 3:
                for j in range(encoderlayers[i][0]):
                    thisrow = []
                    for k in range(encoderlayers[i][1]):
                        thiscol = []
                        for l in range(encoderlayers[i][2]):
                            thiscol.append(variationalneuron(encoderdendrites[i], activation = encoderactivations[i]))
                        thisrow.append(thiscol)
                    thislayer.append(thisrow)
            self.encoder.append(thislayer)
        self.encodermeans = []
        self.encodervarnc = []
        if len(encoderlayers[len(encoderlayers) - 1]) == 1:
            for i in range(encoderlayers[len(encoderlayers) - 1][0]):
                self.encodermeans.append(variationalneuron(encoderdendrites[len(encoderdendrites) - 1], activation = encoderactivations[len(encoderactivations) - 1]))
                self.encodervarnc.append(variationalneuron(encoderdendrites[len(encoderdendrites) - 1], activation = encoderactivations[len(encoderactivations) - 1]))
        elif len(encoderlayers[len(encoderlayers) - 1]) == 2:
            for i in range(encoderlayers[len(encoderlayers) - 1][0]):
                meanrow = []
                varncrow = []
                for j in range(encoderlayers[len(encoderlayers) - 1][1]):
                    meanrow.append(variationalneuron(variationalneuron(encoderdendrites[len(encoderdendrites) - 1], activation = encoderactivations[len(encoderactivations) - 1])))
                    varncrow.append(variationalneuron(variationalneuron(encoderdendrites[len(encoderdendrites) - 1], activation = encoderactivations[len(encoderactivations) - 1])))
                self.encodermeans.append(meanrow)
                self.encodervarnc.append(varncrow)
        elif len(encoderlayers[len(encoderlayers) - 1]) == 3:
            for i in range(encoderlayers[len(encoderlayers) - 1][0]):
                meanrow = []
                varncrow = []
                for j in range(encoderlayers[len(encoderlayers) - 1][1]):
                    meancol = []
                    varnccol = []
                    for k in range(encoderlayers[len(encoderlayers) - 1][2]):
                        meancol.append(variationalneuron(variationalneuron(encoderdendrites[len(encoderdendrites) - 1], activation = encoderactivations[len(encoderactivations) - 1])))
                        varnccol.append(variationalneuron(variationalneuron(encoderdendrites[len(encoderdendrites) - 1], activation = encoderactivations[len(encoderactivations) - 1])))
                    meanrow.append(meancol)
                    varncrow.append(varnccol)
                self.encodermeans.append(meanrow)
                self.encodervarnc.append(varncrow)
        self.decoderblueprint = decoderlayers
        self.decoder = []
        for i in range(len(decoderlayers)):
            thislayer = []
            if len(decoderlayers[i]) == 1:
                for j in range(decoderlayers[i][0]):
                    thislayer.append(variationalneuron(decoderdendrites[i], activation = decoderactivations[i]))
            elif len(decoderlayers[i]) == 2:
                for j in range(decoderlayers[i][0]):
                    thisrow = []
                    for k in range(decoderlayers[i][1]):
                        thisrow.append(variationalneuron(decoderdendrites[i], activation = decoderactivations[i]))
                    thislayer.append(thisrow)
            elif len(decoderlayers[i]) == 3:
                for j in range(decoderlayers[i][0]):
                    thisrow = []
                    for k in range(decoderlayers[i][1]):
                        thiscol = []
                        for l in range(decoderlayers[i][2]):
                            thiscol.append(variationalneuron(decoderdendrites[i], activation = decoderactivations[i]))
                        thisrow.append(thiscol)
                    thislayer.append(thisrow)
            self.decoder.append(thislayer)
    
    def reportstructure(self):
        for i in range(len(self.encoder)):
            print(f'Encoder layer {i + 1}')
            if len(self.encoderblueprint[i]) == 1:
                print(f'Dimensions: [{len(self.encoder[i])}]')
                print(f'Neural input: {self.encoder[i][0].dendrites.shape}')
                print(f'Neuron activation: {self.encoder[i][0].activation}')
            elif len(self.encoderblueprint[i]) == 2:
                print(f'Dimensions: [{len(self.encoder[i])}, {len(self.encoder[i][0])}]')
                print(f'Neural input: {self.encoder[i][0][0].dendrites.shape}')
                print(f'Neuron activation: {self.encoder[i][0][0].activation}')
            elif len(self.encoderblueprint[i]) == 3:
                print(f'Dimensions: [{len(self.encoder[i])}, {len(self.encoder[i][0])}, {len(self.encoder[i][0][0])}]')
                print(f'Neural input: {self.encoder[i][0][0][0].dendrites.shape}')
                print(f'Neuron activation: {self.encoder[i][0][0][0].activation}')
        print(f'Final encoder layer:')
        if len(self.encoderblueprint[len(self.encoderblueprint) - 1]) == 1:
            print(f'Dimensions: [{len(self.encodermeans)}]')
            print(f'Neural input: {self.encodermeans[0].dendrites.shape}')
            print(f'Neuron activation: {self.encodermeans[0].activation}')
        elif len(self.encoderblueprint[len(self.encoderblueprint) - 1]) == 2:
            print(f'Dimensions: [{len(self.encodermeans)}, {len(self.encodermeans[0])}]')
            print(f'Neural input: {self.encodermeans[0][0].dendrites.shape}')
            print(f'Neuron activation: {self.encodermeans[0][0].activation}')
        elif len(self.encoderblueprint[len(self.encoderblueprint) - 1]) == 3:
            print(f'Dimensions: [{len(self.encodermeans)}, {len(self.encodermeans[0])}, {len(self.encodermeans[0][0])}]')
            print(f'Neural input: {self.encodermeans[0][0][0].dendrites.shape}')
            print(f'Neuron activation: {self.encodermeans[0][0][0].activation}')
        for i in range(len(self.decoder)):
            print(f'Decoder layer {i + 1}')
            if len(self.decoderblueprint[i]) == 1:
                print(f'Dimensions: [{len(self.decoder[i])}]')
                print(f'Neural input: {self.decoder[i][0].dendrites.shape}')
                print(f'Neuron activation: {self.decoder[i][0].activation}')
            elif len(self.decoderblueprint[i]) == 2:
                print(f'Dimensions: [{len(self.decoder[i])}, {len(self.decoder[i][0])}]')
                print(f'Neural input: {self.decoder[i][0][0].dendrites.shape}')
                print(f'Neuron activation: {self.decoder[i][0][0].activation}')
            elif len(self.decoderblueprint[i]) == 3:
                print(f'Dimensions: [{len(self.decoder[i])}, {len(self.decoder[i][0])}, {len(self.decoder[i][0][0])}]')
                print(f'Neural input: {self.decoder[i][0][0][0].dendrites.shape}')
                print(f'Neuron activation: {self.decoder[i][0][0][0].activation}')
    
    def encode(self, data, training = False):
        encoded = data
        if training:
            datatrail = []
        for i in range(len(self.encoder)):
            thisdata = []
            if len(self.encoderblueprint[i]) == 1:
                for j in range(len(self.encoder[i])):
                    signal = self.encoder[i][j].fire(encoded)
                    if len(signal.shape) == 1 and signal.shape[0] == 1:
                        thisdata.append(signal[0])
                    else:
                        thisdata.append(signal)
            elif len(self.encoderblueprint[i]) == 2:
                for j in range(len(self.encoder[i])):
                    thisrow = []
                    for k in range(len(self.encoder[i][j])):
                        signal = self.encoder[i][j][k].fire(encoded)
                        if len(signal.shape) == 1 and signal.shape[0] == 1:
                            thisrow.append(signal[0])
                        else:
                            thisrow.append(signal)
                    if len(thisrow) == 0:
                        thisdata.append(thisrow[0])
                    else:
                        thisdata.append(thisrow)
            elif len(self.encoderblueprint[i]) == 3:
                for j in range(len(self.encoder[i])):
                    thiscol = []
                    for k in range(len(self.encoder[i][j])):
                        thisrow = []
                        for l in range(len(self.encoder[i][j][k])):
                            signal = self.encoder[i][j][k][l].fire(encoded)
                            if len(signal.shape) == 1 and signal.shape[0] == 1:
                                thisrow.append(signal[0])
                            else:
                                thisrow.append(signal)
                        if len(thisrow) == 1:
                            thiscol.append(thisrow[0])
                        else:
                            thiscol.append(thisrow)
                    if len(thiscol) == 1:
                        thisdata.append(thiscol[0])
                    else:
                        thisdata.append(thiscol)
            encoded = numpy.array(thisdata)
            if training:
                datatrail.append(numpy.array(thisdata))
        encodedmeans = []
        encodedvarnc = []
        for i in range(len(self.encodermeans)):
            thismean = self.encodermeans[i].fire(encoded)
            thisvarnc = self.encodervarnc[i].fire(encoded)
            if len(thismean.shape) == 1 and thismean.shape[0] == 1:
                encodedmeans.append(thismean[0])
            else:
                encodedmeans.append(thismean)
            if len(thisvarnc.shape) == 1 and thisvarnc.shape[0] == 1:
                encodedvarnc.append(thisvarnc[0])
            else:
                encodedvarnc.append(thisvarnc)
        if training:
            datatrail.append([numpy.array(encodedmeans), numpy.array(encodedvarnc)])
            return datatrail
        else:
            return [numpy.array(encodedmeans), numpy.array(encodedvarnc)]
    
    def decode(self, data, training = False):
        decoded = data
        if training:
            datatrail = []
        for i in range(len(self.decoder)):
            thisdata = []
            if len(self.decoderblueprint[i]) == 1:
                for j in range(len(self.decoder[i])):
                    signal = self.decoder[i][j].fire(decoded)
                    if len(signal.shape) == 1 and signal.shape[0] == 1:
                        thisdata.append(signal[0])
                    else:
                        thisdata.append(signal)
            elif len(self.decoderblueprint[i]) == 2:
                for j in range(len(self.decoder[i])):
                    thisrow = []
                    for k in range(len(self.decoder[i][j])):
                        signal = self.decoder[i][j][k].fire(decoded)
                        if len(signal.shape) == 1 and signal.shape[0] == 1:
                            thisrow.append(signal[0])
                        else:
                            thisrow.append(signal)
                    if len(thisrow) == 1:
                        thisdata.append(thisrow[0])
                    else:
                        thisdata.append(thisrow)
            elif len(self.decoderblueprint[i]) == 3:
                for j in range(len(self.decoder[i])):
                    thiscol = []
                    for k in range(len(self.decoder[i][j])):
                        thisrow = []
                        for l in range(len(self.decoder[i][j][k])):
                            signal = self.decoder[i][j][k][l].fire(decoded)
                            if len(signal.shape) == 1 and signal.shape[0] == 1:
                                thisrow.append(signal[0])
                            else:
                                thisrow.append(signal)
                        if len(thisrow) == 1:
                            thiscol.append(thisrow[0])
                        else:
                            thiscol.append(thisrow)
                    if len(thiscol) == 1:
                        thisdata.append(thiscol[0])
                    else:
                        thisdata.append(thiscol)
            decoded = numpy.array(thisdata)
            if training:
                datatrail.append(numpy.array(thisdata))
        if training:
            return datatrail
        else:
            return decoded
    
    def reparameterize(self, mean, logvariance):
        newsample = []
        for i in range(len(mean)):
            newsample.append(mean[i] + numpy.random.normal() * numpy.exp(logvariance[i] * .5))
        return numpy.array(newsample)
    
    def think(self, thought):
        memory = self.encode(thought)
        hazy = self.reparameterize(memory[0], memory[1])
        idea = self.decode(hazy)
        return idea
    
    def backpropagateerror(self, neurons, layershape, error, inputs, learnrate):
        newgradient = numpy.zeros(inputs.shape)
        if len(layershape) == 1:
            for i in range(len(neurons)):
                newgradient = newgradient + neurons[i].updateweights(error[i], inputs, learnrate)
        elif len(layershape) == 2:
            for i in range(len(neurons)):
                for j in range(len(neurons[i])):
                    newgradient = newgradient + neurons[i][j].updateweights(error[i][j], inputs, learnrate)
        elif len(layershape) == 3:
            for i in range(len(neurons)):
                for j in range(len(neurons[i])):
                    for k in range(len(neurons[i][j])):
                        newgradient = newgradient + neurons[i][j][k].updateweights(error[i][j][k], inputs, learnrate)
        return newgradient
    
    def train(self, trainset, testset, learnrate, maxiterations = 500):
        iteration = 0
        iterationscore = 0
        while iteration < maxiterations:
            for i in range(len(trainset)):
                encoded = self.encode(trainset[i], training = True)
                recoded = self.reparameterize(encoded[len(encoded) - 1][0], encoded[len(encoded) - 1][1])
                decoded = self.decode(recoded, training = True)
                decodergradient = numpy.zeros(decoded[len(decoded) - 1].shape)
                if len(decoded[len(decoded) - 1].shape) == 1:
                    for j in range(decoded[len(decoded) - 1].shape[0]):
                        if decoded[len(decoded) - 1][j] >= 0:
                            decodergradient[j] = (1 / (numpy.exp(-1 * decoded[len(decoded) - 1][j]) + 1)) - trainset[i][j]
                        else:
                            decodergradient[j] = (numpy.exp(decoded[len(decoded) - 1][j]) / (1 + numpy.exp(decoded[len(decoded) - 1][j]))) - trainset[i][j]
                elif len(decoded[len(decoded) - 1].shape) == 2:
                    for j in range(decoded[len(decoded) - 1].shape[0]):
                        for k in range(decoded[len(decoded) - 1].shape[1]):
                            if decoded[len(decoded) - 1][j][k] >= 0:
                                decodergradient[j][k] = (1 / (numpy.exp(-1 * decoded[len(decoded) - 1][j][k]) + 1)) - trainset[i][j][k]
                            else:
                                decodergradient[j][k] = (numpy.exp(decoded[len(decoded) - 1][j][k]) / (1 + numpy.exp(decoded[len(decoded) - 1][j][k]))) - trainset[i][j][k]
                elif len(decoded[len(decoded) - 1].shape) == 3:
                    for j in range(decoded[len(decoded) - 1].shape[0]):
                        for k in range(decoded[len(decoded) - 1].shape[1]):
                            for l in range(decoded[len(decoded) - 1].shape[2]):
                                if decoded[len(decoded) - 1][j][k][l] >= 0:
                                    decodergradient[j][k][l] = (1 / (numpy.exp(-1 * decoded[len(decoded) - 1][j][k][l]) + 1)) - trainset[i][j][k][l]
                                else:
                                    decodergradient[j][k][l] = (numpy.exp(decoded[len(decoded) - 1][j][k][l]) / (1 + numpy.exp(decoded[len(decoded) - 1][j][k][l]))) - trainset[i][j][k][l]
                decodererror = self.backpropagateerror(self.decoder[len(self.decoder) - 1], self.decoderblueprint[len(self.decoderblueprint) - 1], decodergradient, decoded[len(decoded) - 2], learnrate)
                decodergradient = decodererror
                for j in range(1, len(self.decoder) - 1):
                    decodererror = self.backpropagateerror(self.decoder[len(self.decoder) - j - 1], self.decoderblueprint[len(self.decoderblueprint) - j - 1], decodergradient, decoded[len(decoded) - j - 2], learnrate)
                    decodergradient = decodererror
                decodererror = self.backpropagateerror(self.decoder[0], self.decoderblueprint[0], decodergradient, recoded, learnrate)
                inducednoise = (recoded - encoded[len(encoded) - 1][0]) / encoded[len(encoded) - 1][1]
                meangradient = decodererror - encoded[len(encoded) - 1][0]
                logvargradient = numpy.zeros(decodererror.shape)
                for j in range(encoded[len(encoded) - 1][1].shape[0]):
                    if encoded[len(encoded) - 1][1][j] >= 350:
                        logvargradient[j] = (decodererror[j] * inducednoise[j]) + (1 - numpy.exp(2 * 350))
                    else:
                        logvargradient[j] = (decodererror[j] * inducednoise[j]) + (1 - numpy.exp(2 * encoded[len(encoded) - 1][1][j]))
                meanerror = self.backpropagateerror(self.encodermeans, self.encoderblueprint[len(self.encoderblueprint) - 1], meangradient, encoded[len(encoded) - 2], learnrate)
                logvarerror = self.backpropagateerror(self.encodervarnc, self.encoderblueprint[len(self.encoderblueprint) - 1], logvargradient, encoded[len(encoded) - 2], learnrate)
                totalgradient = meanerror + logvarerror
                for j in range(len(self.encoder) - 1):
                    totalerror = self.backpropagateerror(self.encoder[len(self.encoder) - j - 1], self.encoderblueprint[len(self.encoderblueprint) - j - 2], totalgradient, encoded[len(encoded) - j - 3], learnrate)
                    totalgradient = totalerror
                totalerror = self.backpropagateerror(self.encoder[0], self.encoderblueprint[0], totalgradient, trainset[i], learnrate)
            previousscore = iterationscore
            iterationscore = 0
            for i in range(len(testset)):
                encoded = self.encode(testset[i])
                recoded = self.reparameterize(encoded[0], encoded[1])
                decoded = self.decode(recoded)
                iterationscore = iterationscore + numpy.sum(numpy.max(numpy.array([decoded, numpy.zeros(decoded.shape)])) - decoded * testset[i] + numpy.log(1 + numpy.exp(-1 * numpy.abs(decoded))))
                for j in range(encoded[len(encoded) - 1][1].shape[0]):
                    if encoded[len(encoded) - 1][1][j] > 350:
                        iterationscore = iterationscore - .5 * (1 + 2 * encoded[len(encoded) - 1][1][j] + encoded[len(encoded) - 1][0][j]**2 + numpy.exp(2 * 350))
                    else:
                        iterationscore = iterationscore - .5 * (1 + 2 * encoded[len(encoded) - 1][1][j] + encoded[len(encoded) - 1][0][j]**2 + numpy.exp(2 * encoded[len(encoded) - 1][1][j]))
            print([iterationscore, previousscore])
            iteration = iteration + 1

if __name__ == '__main__':
    simpleimage = .1 * numpy.random.random((128, 128))
    for j in range(simpleimage.shape[0]):
        for i in range(simpleimage.shape[1]):
            simpleimage[j][i] = simpleimage[j][i] + numpy.cos(i * numpy.pi / 4) + numpy.cos((i + j * 3**.5) * numpy.pi / 8) + numpy.cos((i - j * 3**.5) * numpy.pi / 8)
    verticaledgelinear = variationalneuron((2, 2), activation = 'linear')
    verticaledgesigmoid = variationalneuron((2, 2), activation = 'sigmoid')
    verticaledgerelu = variationalneuron((2, 2), activation = 'relu')
    verticaledgelinear.setweights(0, numpy.array([[-1, 1], [-1, 1]]))
    verticaledgesigmoid.setweights(0, numpy.array([[-1, 1], [-1, 1]]))
    verticaledgerelu.setweights(0, numpy.array([[-1, 1], [-1, 1]]))
    linearedges = verticaledgelinear.fire(simpleimage)
    sigmoidedges = verticaledgesigmoid.fire(simpleimage)
    reluedges = verticaledgerelu.fire(simpleimage)
    mpl.figure()
    fig, ax = mpl.subplots(2, 2)
    ax[0][0].imshow(simpleimage)
    ax[0][1].imshow(linearedges)
    ax[1][0].imshow(sigmoidedges)
    ax[1][1].imshow(reluedges)