import torch

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math



class StandardNN:
    #default hyper parameters#####################
    layerSize = 30
    learningRate = .023
    bias = .0000001
    runs = 2000
    reservoirSparseness = .9
    bypassReservoir = True
    ######################################


    def __init__(self,layerSizeArg,learningRateArg,biasArg,runsArg,reservoirSparsenessArg,bypassReservoirArg):
        self.layerSize = layerSizeArg
        self.learningRate = learningRateArg
        self.bias = biasArg
        self.runs = runsArg
        self.reservoirSparseness = reservoirSparsenessArg
        self.bypassReservoir = bypassReservoirArg

    def train(self):
        layerSize = self.layerSize
        learningRate = self.learningRate
        bias = self.bias
        runs = self.runs
        reservoirSparseness = self.reservoirSparseness
        bypassReservoir = self.bypassReservoir

        #initialize counter
        count = 0

        #create sparse reservoir
        reservoirTensorSeed = torch.rand(layerSize,layerSize)
        reservoirTensor = torch.where(reservoirTensorSeed>(1-reservoirSparseness),reservoirTensorSeed,0)
        reservoirProjection = torch.rand(1,layerSize)*bias
        preReservoirInputTensor = torch.rand(1,layerSize)*bias

        raw_input_tensor = torch.rand(1,10)

        target_tensor = torch.rand(1,layerSize)*.8
        initialZeroTensor = torch.tensor(torch.zeros(1,layerSize),dtype=torch.float32)

        input_tensor2 = torch.tensor(initialZeroTensor,dtype=torch.float32)
        weight_tensor2 = torch.rand(layerSize,layerSize)-.5

        input_tensor1 = torch.tensor(initialZeroTensor,dtype=torch.float32)
        weight_tensor1 = torch.rand(layerSize,layerSize)-.5

        input_tensori = torch.tensor(initialZeroTensor,dtype=torch.float32)
        weight_tensori= torch.rand(layerSize,layerSize)-.5

        ones_square_tensor = torch.ones(layerSize,layerSize)
        ones_line_tensor = torch.ones(1,layerSize)



        while count < runs:

            #temporary input tensor
            dataTensor = torch.rand(1,10)

            if bypassReservoir == False:
                #embed low dimensional raw input tensor into higher dimensional preReservoirInputTensor
                preReservoirInputTensor[0][0] = preReservoirInputTensor[0][0] + dataTensor[0][0]
                preReservoirInputTensor[0][1] = preReservoirInputTensor[0][1] + dataTensor[0][1]
                preReservoirInputTensor[0][2] = preReservoirInputTensor[0][2] + dataTensor[0][2]
                preReservoirInputTensor[0][3] = preReservoirInputTensor[0][3] + dataTensor[0][3]
                preReservoirInputTensor[0][4] = preReservoirInputTensor[0][4] + dataTensor[0][4]
                preReservoirInputTensor[0][5] = preReservoirInputTensor[0][5] + dataTensor[0][5]
                preReservoirInputTensor[0][6] = preReservoirInputTensor[0][6] + dataTensor[0][6]
                preReservoirInputTensor[0][7] = preReservoirInputTensor[0][7] + dataTensor[0][7]
                preReservoirInputTensor[0][8] = preReservoirInputTensor[0][8] + dataTensor[0][8]
                preReservoirInputTensor[0][9] = preReservoirInputTensor[0][9] + dataTensor[0][9]

                #embed time-varying inputs into reservoir projection
                reservoirProjection = torch.tanh(torch.matmul((preReservoirInputTensor + reservoirProjection+bias),reservoirTensor))+reservoirProjection

                #normalize reservoir projection
                reservoirProjection = reservoirProjection/torch.sum(reservoirProjection)
                input_tensori = reservoirProjection

            else:
                #embed low dimensional raw input tensor into higher dimensional zeros tensor
                input_tensori = initialZeroTensor

                input_tensori[0][0] = input_tensori[0][0] + dataTensor[0][0]
                input_tensori[0][1] = input_tensori[0][1] + dataTensor[0][1]
                input_tensori[0][2] = input_tensori[0][2] + dataTensor[0][2]
                input_tensori[0][3] = input_tensori[0][3] + dataTensor[0][3]
                input_tensori[0][4] = input_tensori[0][4] + dataTensor[0][4]
                input_tensori[0][5] = input_tensori[0][5] + dataTensor[0][5]
                input_tensori[0][6] = input_tensori[0][6] + dataTensor[0][6]
                input_tensori[0][7] = input_tensori[0][7] + dataTensor[0][7]
                input_tensori[0][8] = input_tensori[0][8] + dataTensor[0][8]
                input_tensori[0][9] = input_tensori[0][9] + dataTensor[0][9]

            #feedforward
            output_tensori = torch.matmul(input_tensori,weight_tensori)
            activated_tensori = torch.tanh(output_tensori + bias)
            input_tensor1 = activated_tensori

            output_tensor1 = torch.matmul(input_tensor1,weight_tensor2)
            activated_tensor1 = torch.tanh(output_tensor1 + bias)
            input_tensor2 = activated_tensor1

            output_tensor2 = torch.matmul(input_tensor2,weight_tensor2)
            activated_tensor2 = torch.tanh(output_tensor2 + bias)

            #calculate error
            total_error2 = torch.sum(torch.pow(torch.subtract(activated_tensor2,target_tensor),2))/2
            print(total_error2)

            #calculate activation derivates######
            #layer 2
            a = torch.exp(-output_tensor2)
            b = torch.exp(output_tensor2)
            c = sum(a,b)
            d = torch.pow(c,2)
            delta_activation_tensor2 = 4/d

            #layer 1
            r = torch.exp(-output_tensor2)
            s = torch.exp(output_tensor2)
            t = sum(r,s)
            u = torch.pow(t,2)
            delta_activation_tensor1 = 4/u

            #layer i
            h = torch.exp(-output_tensor1)
            i = torch.exp(output_tensor1)
            j = sum(h,i)
            k = torch.pow(j,2)
            delta_activation_tensori = 4/k
            #####################################

            #calculate output error derivatives
            delta_error_tensor2 = torch.subtract(activated_tensor2,target_tensor,alpha=1)
            flattened_L2_weight_tensor = torch.matmul(ones_line_tensor,weight_tensor2)
            flattened_L1_weight_tensor = torch.matmul(ones_line_tensor,weight_tensor1)

            #calculate layer 2 weight gradient tensor
            activation_and_error_deltas2 = torch.multiply(delta_error_tensor2,delta_activation_tensor2)
            input_tensor_transpose2 = torch.transpose(input_tensor2,0,1)
            weight_gradient_tensor_from_inputs2 = torch.multiply(input_tensor_transpose2,ones_square_tensor)
            complete_weight_gradient_tensor2 = torch.multiply(torch.multiply(activation_and_error_deltas2,weight_gradient_tensor_from_inputs2),learningRate)

            #calculate layer 1 weight gradient tensor
            activation_and_error_deltas1a = torch.multiply(flattened_L2_weight_tensor,delta_activation_tensor1)
            activation_and_error_deltas1b = torch.multiply(activation_and_error_deltas1a,activation_and_error_deltas2)
            input_tensor_transpose1 = torch.transpose(input_tensor1,0,1)
            weight_gradient_tensor_from_inputs1 = torch.multiply(input_tensor_transpose1,ones_square_tensor)
            complete_weight_gradient_tensor1 = torch.multiply(torch.multiply(activation_and_error_deltas1b,weight_gradient_tensor_from_inputs1),learningRate)

            #calculate layer i weight gradient tensor
            activation_and_error_deltasia = torch.multiply(flattened_L1_weight_tensor,delta_activation_tensori)
            activation_and_error_deltasib = torch.multiply(activation_and_error_deltasia,activation_and_error_deltas1b)
            input_tensor_transposei = torch.transpose(input_tensori,0,1)
            weight_gradient_tensor_from_inputsi = torch.multiply(input_tensor_transposei,ones_square_tensor)
            complete_weight_gradient_tensori = torch.multiply(torch.multiply(activation_and_error_deltasib,weight_gradient_tensor_from_inputsi),learningRate)

            #update weight tensors
            weight_tensor2 = torch.subtract(weight_tensor2,complete_weight_gradient_tensor2,alpha=1)
            weight_tensor1 = torch.subtract(weight_tensor1,complete_weight_gradient_tensor1,alpha=1)
            weight_tensori = torch.subtract(weight_tensori,complete_weight_gradient_tensori,alpha=1)

            count = count + 1




nn = StandardNN(30,.023,.0000001,2000,.9,True)
nn.train()











