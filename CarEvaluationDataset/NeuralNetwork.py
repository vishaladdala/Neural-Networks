import math
import random
import sys
def nameOfFile():
    print('filename:');
    file = sys.stdin.readline().strip()
    fop = open(file,'r');
    return fop;
def readFile(file):
    vals=list()
    for line in file:
        fs=[float(f) for f in line.split(",")]
        #fp=[int(g[-1]) for g in line.split(",")]
        vals.append(fs)
        #vals.append(fp)
    return vals;
def initialize_network(n_inputs,n_hidden_layers,n_hidden,n_outputs):
    network=list()
    i=0
    for val in n_hidden:
        hidden_layer=[{'weights':[random.random() for i in range(n_inputs[i]+1)]} for l in range(val)]
        network.append(hidden_layer)
        i+=1
    output_layer=[{'weights':[random.random() for i in range(n_inputs[-1]+1)]} for i in range(n_outputs)]
    #network.append(hidden_layer)
    network.append(output_layer)
    return network
def initialize_network_test(n_inputs,n_hidden_test,n_output_test,n_outputs):
    network=list()
    i=0
    hidden_layers=[]
    output_layer=[]
    for val in n_inputs:
        i+=1
        for l in range(val-1):
            hidden_layers.append({'weights':n_hidden_test[i]})
            i+=1
        network.append(hidden_layers)
    
        
    #for
    j=0
    #for value in n_outputs:
    for m in range(n_outputs):
        output_layer.append({'weights':n_output_test[j]})
        j+=1
    #output_layer=[{'weights':n_output_test[j]}for m in range(n_outputs)]
    network.append(output_layer)
    return network
def activate(weights,inputs):
    activation=weights[-1]
    for i in range(len(weights)-1):
        activation+=weights[i]*inputs[i]
    return activation
def transfer(activation):
    return 1.0/(1.0+math.exp(-activation))
def transfer_derivative(output):
    return output*(1.0-output)
def forward_propagate(network,row):
    inputs=row
    for layer in network:
        new_inputs=[]
        for neuron in layer:
            activation=activate(neuron['weights'],inputs)
            neuron['output']=transfer(activation)
            new_inputs.append(neuron['output'])
        inputs=new_inputs
    return inputs
def back_propagate_error(network,expected):
    for i in reversed(range(len(network))):
        layer=network[i]
        errors=list()
        if i!=len(network)-1:
            for j in range(len(layer)):
                error=0.0
                for neuron in network[i+1]:
                    error+=(neuron['weights'][j]*neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron=layer[j]
                errors.append(expected[j]-neuron['output'])
        for j in range(len(layer)):
            neuron=layer[j]
            neuron['delta']=errors[j]*transfer_derivative(neuron['output'])
def update_weights(network,row,l_rate):
    for i in range(len(network)):
        inputs=row[:-1]
        if i!=0:
            inputs=[neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j]+=l_rate*neuron['delta']*inputs[j]
            neuron['weights'][-1]+=l_rate*neuron['delta']
def train_network(network,train,l_rate,n_epoch,n_outputs,et):
    #sum_error=0.0
    for epoch in range(n_epoch):
        sum_error=0.0
        if(sum_error==et):
            break
        else:
            for row in train:
                outputs=forward_propagate(network,row)
                expected=[0 for i in range(n_outputs)]
                expected[int(row[-1])]=1
                sum_error+=sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                back_propagate_error(network,expected)
                update_weights(network,row,l_rate)
    print('train error=%.3f'%(sum_error/len(train)))
def predict(network,row):
    outputs=forward_propagate(network,row)
    return outputs.index(max(outputs))
def main():
    print("enter name of dataset")
    fin = nameOfFile()
    dataset=readFile(fin)
    train=input("enter the training percent")
    tp=int(train)
    i=int((tp/100)*len(dataset))
    #print(i)
    error_tolerance=input("enter error tolerance")
    et=float(error_tolerance)
    epoch=input("enter number of epoches")
    e=int(epoch)
    dataset_train=dataset[0:i+1]
    dataset_test=dataset[i+1:len(dataset)]
    n_input=len(dataset_train[0])-1
    n_inputs=[]
    n_inputs.append(n_input)
    n_hidden=[]
    n_hidden_test=[]
    n_output_test=[]
    hidden=input('enter number of hidden layers')
    n_hidden_layers=int(hidden)
    for val in range(n_hidden_layers):
        nb=input('enter neurons in each layer')
        value=int(nb)
        n_hidden.append(value)
        n_inputs.append(value)
    n_outputs=len(set([row[-1] for row in dataset]))
    network=initialize_network(n_inputs,n_hidden_layers,n_hidden,n_outputs)
    train_network(network,dataset_train,0.5,e,n_outputs,et)
    i=1
    j=1
    for layer in network:
        if(layer==network[-1]):
            print("output layer")
        else:
            print("hidden layer",i)
        for val in layer:
             print("neuron",j)
             #print(layer.index(val),end=" ")
             print(val['weights'])
             if(layer==network[-1]):
                 n_output_test.append(val['weights'])
             else:
                n_hidden_test.append(val['weights'])
             j+=1
        i+=1
        j=1
    #print(n_hidden_test)
    #print(n_output_test)
    #print(n_inputs[1:])
    #print(initialize_network_test(n_inputs[1:],n_hidden_test,n_output_test,n_outputs))
    for layer in network:
        for val in layer:
            del val['delta']
            del val['output']
    network_test=list()
    for layer in network:
        network_test.append(layer)
    #print(network_test)
    test_error=0
    accuracy=0
    for row in dataset_test:
        prediction=predict(network_test,row)
        actual=row[-1]
        if(prediction!=actual):
            test_error+=1
        else:
            accuracy+=1
        #print('Expected=%d, Got=%d' % (row[-1], prediction))
    print("Test Error is")
    print((test_error)/len(dataset_test))
    print("Accuracy for test model")
    print((accuracy/len(dataset_test))*100)
main()
