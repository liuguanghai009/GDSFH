function net = convertdln(net)

lgraph = net.layerGraph();

lgraph = removeLayers(lgraph,'softmax');

lgraph = removeLayers(lgraph,'classoutput');

net = dlnetwork(lgraph);

